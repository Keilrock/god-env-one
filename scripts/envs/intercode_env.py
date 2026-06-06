"""InterCode-Bash (NL2Bash) GRPO training environment.

This is the *training-time* counterpart to ``validator/evaluation/eval_intercode.py``.
It follows the rollout contract of ``liar_dice_env.py`` (same public function names,
same return-dict shape, ``CurriculumScheduler`` from ``shared_env``) but adapts it
for InterCode's in-process bash executor.

Two deliberate deviations from the OpenSpiel game envs (liars_dice / leduc / gin):

  1. NO HTTP env-server. The official evaluator runs ``LocalBashEnv`` in-process
     against global managed paths (/testbed, /system, /workspace, /backup), and
     runs tasks SEQUENTIALLY because those paths are shared. We mirror that here:
     ``init_env_pool`` is skipped entirely and ``_dispatch`` is a sequential loop,
     NOT a ``thread_pool`` fan-out. Running episodes concurrently would corrupt
     each other's filesystem (same reason the evaluator is serial).

  2. REWARD IS A 1:1 PORT of ``eval_intercode.LocalBashEnv._get_reward`` — terminal
     3-component continuous reward, no intermediate shaping. ``LocalBashEnv``, the
     ReAct prompt/demo, the action parser, and the NL2Bash asset loader below are
     copied verbatim from the evaluator so training-reward == eval-reward and the
     policy cannot overfit to a proxy signal.

The only genuinely new code is the chat-multi-turn rollout (``_run_episode`` /
``_dispatch``), which reuses the existing ``generate_rollout_completions`` +
action-masking machinery instead of the evaluator's single-string ReAct loop.

NOTE (DDP): because managed paths are global to the process tree, multi-GPU DDP
ranks sharing one container would interfere. The evaluator is single-process; this
env inherits the same constraint. Run intercode training with one rank per host fs.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path

from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    rollout_reward_func,  # re-exported for callers (env_configs registry)
)


# ---------------------------------------------------------------------------
# Constants (mirrored from eval_intercode.py so behaviour matches the evaluator)
# ---------------------------------------------------------------------------

_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN     = 5000

# --- Evaluator-equivalent knobs (same env vars / defaults as eval_intercode) ---
DEFAULT_INTERCODE_DATA_ROOT = Path("/intercode_data")
DEFAULT_INTERCODE_FS_ROOT   = Path("/intercode_fs")

DEFAULT_ACTION_TIMEOUT_SECONDS = 30
DEFAULT_OBS_TRUNCATE_CHARS     = 350
DEFAULT_MAX_TURNS              = 10   # eval's DEFAULT_MAX_TURNS — curriculum ramps toward this

DEFAULT_SCORING_MODE = "continuous"
VALID_SCORING_MODES  = {"continuous", "binary"}

# Same env-var contract as the evaluator. Default "continuous" → reward ∈ [0.01, 1.0].
# Reading the identical env var keeps the training reward on the exact same code path
# as eval no matter how the deployment is configured.
SCORING_MODE = os.getenv("INTERCODE_SCORING_MODE", DEFAULT_SCORING_MODE).strip().lower()
assert SCORING_MODE in VALID_SCORING_MODES, f"invalid INTERCODE_SCORING_MODE={SCORING_MODE!r}"

ALL_MANAGED_PATHS = ("/testbed", "/system", "/workspace", "/backup")
PATHS_PER_FS: dict[int, tuple[str, ...]] = {
    1: ("/testbed",),
    2: ("/system",),
    3: ("/workspace", "/backup"),
    4: (),  # filesystem-agnostic
}

_INTERCODE_RANGE_START = GAMES_TO_TASK_ID_RANGE["intercode"][0]


# ===========================================================================
# === PORTED VERBATIM FROM eval_intercode.py ================================
# Everything between this banner and the next one is a faithful copy of the
# evaluator. Do NOT add training-side shaping here — it must stay 1:1 so the
# reward the policy trains against is the reward it is scored on.
# ===========================================================================

# --- NL2Bash dataset mapping ----------------------------------------------

def _load_data(data_root: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for fs in (1, 2, 3, 4):
        path = data_root / f"nl2bash_fs_{fs}.json"
        out[fs] = json.loads(path.read_text())
    return out


def _compute_fs_ranges(data: dict[int, list[dict]]) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    cursor = 1
    for fs in (1, 2, 3, 4):
        n = len(data[fs])
        ranges.append((fs, cursor, cursor + n - 1))
        cursor += n
    return ranges


def _map_task_id(global_id: int, ranges: list[tuple[int, int, int]]) -> tuple[int, int]:
    for fs, start, end in ranges:
        if start <= global_id <= end:
            return fs, global_id - start
    total = ranges[-1][2]
    raise ValueError(f"task id {global_id} out of range; valid range is 1..{total}")


@dataclass(frozen=True)
class InterCodeAssets:
    data: dict[int, list[dict]]
    ranges: list[tuple[int, int, int]]
    snapshot_root: Path

    @property
    def total_tasks(self) -> int:
        return self.ranges[-1][2] if self.ranges else 0


def load_intercode_assets(
    data_root: "Path | str | None" = None,
    snapshot_root: "Path | str | None" = None,
) -> InterCodeAssets:
    data_path = (
        Path(data_root)
        if data_root is not None
        else Path(os.getenv("INTERCODE_DATA_ROOT", str(DEFAULT_INTERCODE_DATA_ROOT)))
    )
    snapshot_path = (
        Path(snapshot_root)
        if snapshot_root is not None
        else Path(os.getenv("INTERCODE_FS_ROOT", str(DEFAULT_INTERCODE_FS_ROOT)))
    )
    if not data_path.exists():
        raise RuntimeError(f"NL2Bash data not found at {data_path}; image may be misbuilt")
    if not snapshot_path.exists():
        raise RuntimeError(f"InterCode fs snapshots not found at {snapshot_path}; image may be misbuilt")

    data = _load_data(data_path)
    ranges = _compute_fs_ranges(data)
    # Guard against an empty/misbuilt dataset. With total==0 the prompt->task fold
    # in _map_prompt_to_task does `% total` (ZeroDivisionError, caught per-episode →
    # every intercode episode silently skips). That makes training *look* successful
    # while learning nothing. Fail loud here instead.
    total = ranges[-1][2] if ranges else 0
    if total <= 0:
        per_fs = {fs: len(data[fs]) for fs in (1, 2, 3, 4)}
        raise RuntimeError(
            f"intercode assets empty — check fs/data mount "
            f"(data_path={data_path}, per_fs_counts={per_fs}, total={total})"
        )
    return InterCodeAssets(data=data, ranges=ranges, snapshot_root=snapshot_path)


class LocalBashEnv:
    """In-process, docker-free analogue of intercode.envs.BashEnv for NL2Bash.

    Verbatim port of eval_intercode.LocalBashEnv — reset/_restore_fs/_capture_state/
    _exec_action/step/_get_reward are byte-for-byte equivalent so the training reward
    equals the eval reward.
    """

    def __init__(self, fs_version: int, entries: list[dict], snapshot_root: Path):
        self.fs_version = fs_version
        self.entries = entries
        self.managed_paths = PATHS_PER_FS[fs_version]
        self.snapshot_tar = snapshot_root / f"fs{fs_version}.tar"
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self.action_executed = False
        self.query: "str | None" = None
        self.gold: "str | None" = None
        self._snapshot_state: "dict[str, tuple] | None" = None
        self._agent_state: "dict[str, tuple] | None" = None
        self._eval_state: "dict[str, tuple] | None" = None

    def reset(self, index: int) -> str:
        record = self.entries[index]
        self.query = record["query"]
        self.gold = record.get("gold", "") or ""
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self._restore_fs()
        self._snapshot_state = self._capture_state()
        return self.query

    def _restore_fs(self) -> None:
        # Wipe ALL managed paths (not just this variant's) so leftovers from a
        # previous task can't leak into the current one — important for fs_4
        # which has no managed paths of its own.
        for p in ALL_MANAGED_PATHS:
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        if not self.managed_paths or not self.snapshot_tar.exists():
            return
        try:
            subprocess.run(
                ["tar", "-xpf", str(self.snapshot_tar), "-C", "/"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"failed to restore fs_{self.fs_version} snapshot: "
                f"{exc.stderr.decode('utf-8', errors='replace')}"
            )

    def _capture_state(self) -> dict[str, tuple]:
        state: dict[str, tuple] = {}
        for root_path in self.managed_paths:
            if not os.path.exists(root_path):
                continue
            for cur, dirs, files in os.walk(root_path):
                for name in dirs:
                    full = os.path.join(cur, name)
                    try:
                        st = os.lstat(full)
                        state[full] = ("<DIR>", st.st_mode)
                    except OSError:
                        state[full] = ("<ERR>", 0)
                for name in files:
                    full = os.path.join(cur, name)
                    try:
                        if os.path.islink(full):
                            state[full] = ("<LINK>", os.readlink(full))
                        else:
                            h = hashlib.md5()
                            with open(full, "rb") as fh:
                                for chunk in iter(lambda: fh.read(65536), b""):
                                    h.update(chunk)
                            st = os.lstat(full)
                            state[full] = (h.hexdigest(), st.st_size)
                    except OSError:
                        state[full] = ("<ERR>", 0)
        return state

    @staticmethod
    def _simplify_path(current: str, changed: str) -> str:
        """Resolve a `cd` argument against the current workdir — matches BashEnv."""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""
        path: list[str] = []
        for seg in (current + "/" + changed).split("/"):
            if seg == "..":
                if path:
                    path.pop()
            elif seg and seg != ".":
                path.append(seg)
        return "/" + "/".join(path)

    def _exec_action(self, action: str) -> None:
        is_cd = action.startswith("cd")
        new_path: str | None = None
        if is_cd and "cd " in action:
            cd_arg = action[action.index("cd ") + 3:].strip()
            new_path = self._simplify_path(self.workdir, cd_arg)
            action = f"cd {new_path}"
        try:
            res = subprocess.run(
                ["/bin/bash", "-c", action],
                cwd="/" if is_cd else (self.workdir or "/"),
                capture_output=True,
                timeout=DEFAULT_ACTION_TIMEOUT_SECONDS,
            )
            stdout = res.stdout.decode("utf-8", errors="replace")
            stderr = res.stderr.decode("utf-8", errors="replace")
            self.observation = stdout + (stderr if not stdout else "")
            self.action_executed = res.returncode == 0
            if is_cd and self.action_executed and new_path is not None:
                self.workdir = new_path
        except subprocess.TimeoutExpired:
            self.observation = "Command timed out"
            self.action_executed = False
        except Exception:
            self.observation = "Malformed command"
            self.action_executed = False

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if action.startswith("submit"):
            reward, info = self._get_reward()
            info["action_executed"] = True
            return self.observation, reward, True, info
        self._exec_action(action)
        return self.observation, 0.0, False, {"action_executed": self.action_executed}

    def _get_reward(self) -> tuple[float, dict]:
        # Snapshot end-state of the agent's filesystem before running gold.
        self._agent_state = self._capture_state()

        # Run the gold command in a freshly-restored filesystem.
        self._restore_fs()
        gold_obs = ""
        corrupt_gold = False
        if self.gold:
            try:
                res = subprocess.run(
                    ["/bin/bash", "-c", self.gold],
                    cwd="/",
                    capture_output=True,
                    timeout=DEFAULT_ACTION_TIMEOUT_SECONDS,
                )
                gold_obs = (
                    res.stdout.decode("utf-8", errors="replace")
                    + res.stderr.decode("utf-8", errors="replace")
                )
            except Exception:
                corrupt_gold = True
        self.observation_eval = gold_obs
        self._eval_state = self._capture_state()

        snapshot = self._snapshot_state or {}
        agent_changed = self._changed_paths(snapshot, self._agent_state or {})
        eval_changed = self._changed_paths(snapshot, self._eval_state or {})

        diff_miss = eval_changed - agent_changed
        diff_extra = agent_changed - eval_changed
        diff_same = agent_changed & eval_changed

        common_changes_total = len(diff_same)
        common_changes_correct = sum(
            1
            for path in diff_same
            if (self._agent_state or {}).get(path) == (self._eval_state or {}).get(path)
        )
        agent_obs = self.observation or ""
        gold_obs = self.observation_eval or ""

        # Part 1: filesystem-state diff size, smoothed via erf (matches upstream).
        p1 = round(0.33 * (1 - math.erf(len(diff_miss) + len(diff_extra))), 2)

        # Part 2: of the paths both agent and gold modified, what fraction match?
        if common_changes_total:
            p2 = round(0.33 * (common_changes_correct / common_changes_total), 2)
        else:
            p2 = 0.33

        # Part 3: TF-IDF cosine on agent vs gold stdout; falls back to exact match.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([agent_obs, gold_obs])
            sim = float((tfidf * tfidf.T).toarray()[0][1])
        except Exception:
            sim = 1.0 if agent_obs == gold_obs else 0.0
        p3 = round(0.33 * sim, 2)

        continuous_reward = 0.01 + p1 + p2 + p3

        # Binary pass criteria for each component:
        #   - fs diff:    no missing or extra changes vs gold
        #   - content:    every commonly-changed path is byte-identical
        #   - answer:     stdout matches gold after whitespace normalization
        fs_pass = (len(diff_miss) == 0) and (len(diff_extra) == 0)
        content_pass = (common_changes_total == 0) or (common_changes_correct == common_changes_total)
        answer_pass = " ".join(agent_obs.split()) == " ".join(gold_obs.split())
        all_pass = fs_pass and content_pass and answer_pass
        binary_reward = 1.0 if all_pass else 0.0

        reward = binary_reward if SCORING_MODE == "binary" else continuous_reward
        info = {
            "scoring_mode": SCORING_MODE,
            "file_diff": p1,
            "file_changes": p2,
            "answer_similarity": p3,
            "continuous_reward": continuous_reward,
            "binary_reward": binary_reward,
            "fs_pass": fs_pass,
            "content_pass": content_pass,
            "answer_pass": answer_pass,
            "diff_miss": list(diff_miss),
            "diff_extra": list(diff_extra),
            "corrupt_gold": corrupt_gold,
        }
        return reward, info

    @staticmethod
    def _changed_paths(before: dict[str, tuple], after: dict[str, tuple]) -> set[str]:
        keys = set(before) | set(after)
        return {k for k in keys if before.get(k) != after.get(k)}


# --- ReAct prompt + parser (inlined from princeton-nlp/intercode) ----------

REACT_INIT_MSG = """Interact with a Bourne Shell system using Bash queries to answer a question.
Use interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be two types:
(1) execute, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Here are some examples.
"""

DEMO_BASH_REACT = """Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Thought 1: I can use the `xxd` command to get a hexdump of the /textbed/textfile7.txt file
Action 1: execute[xxd -p -l 16 /testbed/textfile7.txt]
Observation 1: /bin/bash: line 1: xxd: command not found
Thought 2: Seems like xxd is not available. I can try to use `od` (octal dump) command to get a hexadecimal dump.
Action 2: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt]
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Thought 3: This command is showing the hexadecimal representation of the first 16 bytes, but it is not a single value.
I should try stitching together the values into one continuous string.
Action 3: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' ']
Observation 3: 546578742066696c6520372077697468
Thought 4: I've configured
Action 4: submit
Question: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
Thought 1: Find all files with the `.java` extension in `testbed/`, then calculate the md5hash, sort them,
filter out duplicates, and only pick the first line of the hash.
Action 1: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -d | cut -d' ' -f1]
Observation 1:
Thought 2: Seems like this did not produce any results. I may have to use another command instead of cut
to pick out the hashes from the output string
Action 2: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -Dw 32]
Observation 2: f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello1.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/dir1/AnotherHello.java
Thought 3: This shows the output hashes and they have the same values, indicating that these files are duplicates.
However, the file names are also shown, which are not needed.
Action 3: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | cut -c 35-]
Observation 3: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 4: This shows the file names exclusively, and no longer shows the hashes. It seems that the cut
command argument may not be the best choice for selecting file names.
Action 4: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $2}']
Observation 4: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 5: I use the awk command instead, but instead of printing out the hashes, it still prints out the file
names. I should select a different part of the output string instead of `$2`
Action 5: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $1}']
Observation 5: f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
Thought 6: This prints out identical hashes, and based on previous observations, I know that these are hashes of
duplicates `.java` files from the `testbed/` directory. This should be correct. I will submit.
Action 6: submit
Question: print disk usage in human readable format of files or folders in /workspace
Thought 1: The `du` command is useful for printing out disk usage of a specific directory. I can use this to
display this information for the `workspace` directory
Action 1: execute[du /workspace]
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Thought 2: The default `du` command gives storage in a non-human readble font. I can use the -h option
of the du command to print storage size with bytes.
Action 2: execute[du -h /workspace]
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Thought 3: This gives me storage information for every folder under the workspace directory, but
I only need the storage for just the `workspace/` directory. The `-s` option should help with this.
Action 3: execute[du -sh /workspace]
Observation 3: 100K\t/workspace
Thought 4: This shows data usage in human readable format for the `workspace` directory. I am finished.
Action 4: submit
Question: Count all the lines of all php files in the /testbed directory recursively
Thought 1: I should find the paths to all php files in the testbed directory, then apply the word
count command to each path.
Action 1: execute[find /testbed -name "*.php" | xargs wc -l]
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Thought 2: This shows me too much information, I only need the total number of lines. I should add up
the lines together and output a single number.
Action 2: execute[find /testbed -name "*.php" -exec wc -l {} + | awk '{total += $1} END{print total}']
Observation 2: 4
Thought 3: This total is wrong, it doesn't match the previous observation, where total is 2. I only
need to apply the word count command.
Action 3: execute[find /testbed -name "*.php" -type f -exec cat {} + | wc -l]
Observation 3: 2
Thought 4: The value is 2, which matches the initial observation that the total lines of php files in the
testbed directory is 2. I can submit.
Action 4: submit
Question: Create a hello.txt file in the /testbed directory and add the text "Hello world" to it.
Thought 1: I can first create a `hello.txt` file in the `testbed/` directory
Action 1: touch testbed/hello.txt
Observation 1:
Thought 2: I should check that the file was created successfully.
Action 2: execute[ls testbed/]
Observation 2: dir1/
dir2/
dir3/
hello.txt
files.txt
Thought 3: I can now add the "Hello world" text to the hello.txt file
Action 3: execute[echo Hello world > hello.txt]
Observation 3:
Thought 4: I should check that the text was written successfully to the hello.txt file.
Action 4: execute[cat testbed/hello.txt]
Observation 4: Hello world
Thought 5: The hello.txt file has been created successfully in the testbed/ directory, and it contains
the Hello World text. I can submit.
Action 5: submit
"""

_REACT_ACTION_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


def _parse_action(action: str) -> tuple[str, bool]:
    if action == "submit":
        return action, True
    matches = _REACT_ACTION_RE.findall(action)
    if matches:
        return matches[0], True
    return action, False


# ===========================================================================
# === END VERBATIM PORT =====================================================
# Everything below is training-side adaptation (chat rollout + masking).
# ===========================================================================


def _extract_action(completion_text: str, turn: int) -> tuple[str, bool]:
    """Pull the bash action out of a chat-mode assistant turn.

    The evaluator generates a constrained ``Thought N:``/``Action N:`` string and
    splits on the literal markers. In chat mode the assistant returns one free-form
    message, so we isolate the ``Action {turn}:`` segment (falling back to a bare
    ``Action:`` or the whole message), strip any hallucinated trailing
    ``Observation`` block, then defer to the verbatim ``_parse_action``.
    """
    m = re.search(rf"Action\s*{turn}\s*:\s*(.*)", completion_text, re.DOTALL)
    if m is None:
        m = re.search(r"Action\s*:\s*(.*)", completion_text, re.DOTALL)
    segment = (m.group(1) if m else completion_text).strip()

    # Cut anything the model emitted past its own action (e.g. a faked Observation).
    segment = re.split(r"\n\s*Observation\s*\d*\s*:", segment, maxsplit=1)[0].strip()

    if _REACT_ACTION_RE.search(segment):
        # _parse_action's regex grabs the execute[...] body (DOTALL).
        return _parse_action(segment)
    # Tolerate "submit" with surrounding whitespace / trailing text.
    if segment == "submit" or segment.split("\n", 1)[0].strip() == "submit":
        return "submit", True
    return segment, False


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry.

    InterCode has no hint mechanism, so hint_prob is pinned to 0. The schedule only
    ramps the per-episode turn budget from ``initial_max_turn`` up to the evaluator's
    DEFAULT_MAX_TURNS (10).
    """
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=DEFAULT_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.0,
        final_hint_prob=0.0,
        warmup_rollouts=512,
    )


# ---------------------------------------------------------------------------
# Module-level state (shared between full and last rollout functions)
# ---------------------------------------------------------------------------

_state: dict = {}


def _ensure_initialized(trainer) -> None:
    """Load NL2Bash assets + curriculum once per process (no-op afterwards).

    Unlike the OpenSpiel game envs, there is NO ``init_env_pool`` call: InterCode
    runs in-process against global managed paths, so there is no HTTP server pool to
    warm up. We only need the dataset/snapshot assets and the curriculum.
    """
    if _state.get("initialized"):
        return

    assets = load_intercode_assets()
    curriculum = _curriculum_factory(trainer.args)
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    _log_rank = os.environ.get("LOG_RANK", "0")
    if _log_rank == "all" or str(rank) == _log_rank:
        print(
            f"[CURRICULUM] InterCode initialized: initial_max_turn={trainer.args.initial_max_turn}, "
            f"final_max_turn={DEFAULT_MAX_TURNS}, rollouts_per_stage={trainer.args.rollouts_per_stage}, "
            f"total_tasks={assets.total_tasks}, scoring_mode={SCORING_MODE}"
        )

    _state.update(
        initialized=True,
        rank=rank,
        assets=assets,
        curriculum=curriculum,
    )


def _map_prompt_to_task(prompt: str, assets: InterCodeAssets) -> tuple[int, int]:
    """Map a training prompt (task id in the intercode range) to (fs_version, local_idx).

    The dataset feeds prompts as ``str(task_id)`` sampled from
    ``GAMES_TO_TASK_ID_RANGE["intercode"]`` (800M block). InterCode itself only has
    ``assets.total_tasks`` real tasks (global ids 1..total), so we fold the large
    training id back into that 1..total space deterministically, then reuse the
    evaluator's ``_map_task_id``.
    """
    gid = int(prompt)
    total = assets.total_tasks
    intercode_global = (gid - _INTERCODE_RANGE_START) % total + 1
    return _map_task_id(intercode_global, assets.ranges)


# ---------------------------------------------------------------------------
# Core episode runner (shared between full-prompt and last-prompt variants)
# ---------------------------------------------------------------------------

def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt: bool,
    assets: InterCodeAssets,
    trainer,
    tokenizer,
    current_max_turn: int,
) -> tuple[int, "dict | None"]:
    """Run one InterCode ReAct episode as a chat conversation.

    Token accumulation / action masking mirror ``liar_dice_env._run_episode``:
    mask=1 over LLM completion tokens, mask=0 over injected observation tokens.
    The reward is the SINGLE terminal reward from ``LocalBashEnv`` at submit — no
    intermediate shaping (so it equals the eval reward exactly).
    """
    gid = int(prompt)
    try:
        fs_version, local_idx = _map_prompt_to_task(prompt, assets)
        env = LocalBashEnv(fs_version, assets.data[fs_version], assets.snapshot_root)
        query = env.reset(local_idx)
    except Exception as exc:
        traceback.print_exc()
        print(f"Failed to reset InterCode env (task {gid}): {exc}")
        return index, None

    # --- Full-prompt accumulation state (only used when use_full_prompt=True) ---
    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: "list[int] | None"   = None

    # Last-prompt fallback (overwritten every loop iteration in use_full_prompt=False mode)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    # --- Episode state ---
    done            = False
    terminal_reward = 0.0
    turn_number     = 0
    invalid_count   = 0

    system_prompt = REACT_INIT_MSG + DEMO_BASH_REACT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Question: {query}"},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        display_turn = turn_number + 1  # ReAct examples are 1-indexed

        rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (identical scheme to liar_dice_env) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # Re-tokenising the full conversation can shift earlier token IDs
                    # (BPE tokenisers are not context-free). Skip delta mask this turn.
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask for this turn."
                    )
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- Parse + execute action ---
        action_parsed, is_code = _extract_action(completion_text, display_turn)
        if not is_code:
            observation = (
                "Error executing query: Your last `execute` action did not "
                "contain bash code"
            )
            invalid_count += 1
        else:
            observation, reward, done_step, _info = env.step(action_parsed)
            if done_step:
                terminal_reward = reward
                done = True

        if isinstance(observation, str) and len(observation) > DEFAULT_OBS_TRUNCATE_CHARS:
            observation = observation[:DEFAULT_OBS_TRUNCATE_CHARS]

        messages.append({"role": "user", "content": f"Observation {display_turn}: {observation}"})
        turn_number += 1

    # --- Terminal reward: force a submit if the agent never did (mirrors eval) ---
    if not done:
        _, terminal_reward, _, _ = env.step("submit")

    train_reward = float(terminal_reward)

    _metric_line = (
        "[ID:{:<6} fs:{} Done:{} T:{:>2d} | Reward:{:>6.3f} | Inv:{:<2}]".format(
            str(gid)[:6], fs_version, int(done), turn_number, train_reward, invalid_count,
        )
    )

    # --- Build result ---
    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    train_reward,
            "done":           done,
            "metric_line":    _metric_line,
            "messages":       messages,
        }
    else:
        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         train_reward,
            "final_score":    train_reward,
            "done":           done,
            "metric_line":    _metric_line,
            "messages":       messages,
        }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    """Common dispatch + aggregation logic for both rollout variants.

    SEQUENTIAL by design: InterCode's managed paths (/testbed, /system, ...) are
    global, so episodes cannot run concurrently without corrupting each other's
    filesystem — the same reason the official evaluator runs tasks one at a time.
    There is therefore no ``thread_pool`` fan-out here (cf. liar_dice_env._dispatch).
    """
    _ensure_initialized(trainer)

    curriculum       = _state["curriculum"]
    assets           = _state["assets"]
    current_max_turn = curriculum.get_max_turn()
    tokenizer        = trainer.processing_class

    _log_rank = os.environ.get("LOG_RANK", "0")
    _should_log = _log_rank == "all" or str(_state["rank"]) == _log_rank
    if _should_log:
        print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}")

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "done": False, "metric_line": "[FALLBACK]", "messages": []}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "done": False, "metric_line": "[FALLBACK]", "messages": []}
    )

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        assets=assets,
        trainer=trainer,
        tokenizer=tokenizer,
        current_max_turn=current_max_turn,
    )

    results: list[dict] = []
    for i, p in enumerate(prompts):
        try:
            _, res = run(i, p)
        except Exception as exc:
            traceback.print_exc()
            print(f"Episode failed (task {p}): {exc}")
            res = None
        results.append(res if res is not None else _fallback)

    curriculum.step(len(prompts))

    finished   = sum(1 for r in results if r.get("done", False))
    avg_return = sum(r["reward"] for r in results) / len(results) if results else 0

    _log_trajectories = bool(os.environ.get("LOG_TRAJECTORIES"))
    _batch_lines = [f"[BATCH] Finished: {finished}/{len(results)}, AvgReturn: {avg_return:.3f}"]
    for r in results:
        line = r.get("metric_line", "")
        if _log_trajectories:
            line += "\n" + json.dumps(r.get("messages", []))
        _batch_lines.append(line)
    print("\n".join(_batch_lines), flush=True)

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs":       [r["logprobs"]       for r in results],
        "env_rewards":    [r["reward"]         for r in results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in results]
    return out


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Sequential rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Sequential rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
