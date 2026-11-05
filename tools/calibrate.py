# /// script
# requires-python = ">=3.14"
# ///
"""Tune renza's engine start-up settings on this machine by measuring decode speed.

    uv run tools/calibrate.py [--dry-run] -- RENZA-COMMAND ... -- ENGINE-FLAGS ...

The command is the full renza command line, its engine part after its own `--`. The tool starts it once per
measurement (the candidate's flags replaced, or added if absent; the rest as given), waits for /v1/models, sends
greedy streamed chat completions with thinking off, and stops the server before the next start. URL, port and API
key come from the command's --host, --port and --api-key.

The tuned flags are removed from the command first, so every setting is measured against the engine's own default
(after an engine change, the old calibration is not the baseline). Steps 1-4 each change one flag in isolation:
every candidate is the command with no tuned flag plus that flag, against that command as measured in the first
start (reused). A climb measures both neighbours of its starting value, then one step at a time in the faster
direction while it improves: a step faster than the best so far continues; one not faster but within 2% of the best
(FLAT_BAND) gets one more step, so a flat first step does not end the climb; a step more than 2% below the best, or
the second step in a row that is not faster, ends it.
  1. --pool-workers, the PCIe share on automatic: from the engine's own count (in the first start's log) in steps
     of 1.
  2. --pcie-frac: the engine's PcieAuto sets the share from the CPU pool's speed and the link; from the share it
     settled on in the first start, fixed shares in steps of 0.05.
  3. --spec-min-p: 0.5 first, the middle of its useful range (up to 0.2 it cuts next to nothing, at 1.0 drafting
     is off), then in 0.1 steps between 0.1 and 0.9, all against the engine's 0.0.
  4. The adaptive expert tier, as three quantities per decode window, as one mechanism (each climb at the tier
     values chosen so far): how often it adapts (--adapt-every, steps of 1), how many experts it swaps per window
     (--adapt-swaps = that rate times --adapt-every, steps of 8 per window) and how much of the usage counts it
     keeps per window (--adapt-decay = that memory to the power --adapt-every, steps of 0.02). Each climb holds the
     other two per-window quantities; they start at the engine's (24 swaps and 0.7 ** (1/4) per window).
  5. Combination: the fastest value of steps 1-3 and the tier's result, each when it beat the default, sorted by
     that gain, added one at a time to the defaults; each addition is kept if the set is faster with it than
     without, else dropped. The first addition is its own start (the defaults plus that value), reused.
  6. If the combination changed anything: it against the command with no tuned flag, a start each, back to back. It
     is kept only when it beats the defaults by more than 3% (MIN_GAIN), else the result is the engine's defaults:
     start-ups, the adaptive tier and the OS move single measurements by a few percent, and this is the one decision
     that changes the configuration. Steps 1-5 only choose what to measure next.

Decode tok/s of one request: its completion tokens minus one over the time from the first streamed text to the last.
The measured rounds are the turns of three conversations, as an agent's session grows: round N sends turn N of each
conversation with the earlier turns and the model's own answers as history, so no start answers the same text twice
(answering it again lets the expert tier learn those very texts) and the server reuses each conversation's prefix.
Every start goes through the same turns, so starts compare. The warm-ups use prompts of their own. A round is the
median over the conversations; a candidate's value is the median over its rounds.

It prints each step's candidates (every round's tok/s and the median), what each step kept, the share PcieAuto
settled on and any move of the share during a start's measured rounds, the confirmation, where each default comes
from, the starts and the wall time, and as the last line the tuned engine flags to use in place of any the command
had (empty: the engine's defaults). It writes no file.
--dry-run prints every command a run would start, with its step, and starts nothing: it cannot know the results, so
it assumes the most starts (every candidate improves) and a share settled at a native pack's start-up default.
"""
import argparse
import json
import math
import re
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

MIN_GAIN = 0.03
MIN_WORKERS = 2
FLAT_BAND = 0.02                   # a step this close below the best so far is flat, not worse (see Climb)
# fixed shares around PcieAuto's, in its own resolution (it rounds to 0.05); 1.0: every missed expert over PCIe
PCIE_STEP = 0.05
# --spec-min-p from the engine's 0.0 (the window always takes every draft guess) up in 0.1 steps; at 1.0 the window
# practically never extends, which is drafting off (engine.cpp, the window's T)
SPEC_START, SPEC_STEP, SPEC_MAX = 0.5, 0.1, 0.9
ROUNDS = 3                         # the turns of each conversation
FIRST_WARM_UPS = 2                 # rounds before measuring, after the first start and after each later one
LATER_WARM_UPS = 1
# warm-up rounds of every start with the share on automatic (no --pcie-frac), so both sides of a comparison are
# measured after PcieAuto settled: it settles after 240 decode windows (48 + 96 + 96, engine.cpp PcieAuto::window); a
# window gives at most --spec tokens, so at the reference --spec 4 a 128-token answer can take as few as 32 windows:
# 8 requests, three rounds of the three prompts
SETTLE_WARM_UPS = 3
# starts per side in the final confirmation: at 512 tokens two starts of one setting differed by under 0.2% (first
# GPU run), far below MIN_GAIN
CONFIRM_STARTS = 1
# the adaptive tier: every --adapt-every decode windows it swaps up to --adapt-swaps of the most-routed missing
# experts into VRAM, then multiplies the usage counts by --adapt-decay (engine.cpp)
ADAPT_FLAGS = ("--adapt-every", "--adapt-swaps", "--adapt-decay")
TIER_RATE_STEP = 8                 # swaps per window (32 swaps at every 4 windows)
TIER_MEMORY_STEP = 0.02            # usage kept per window
TIER_MEMORY_MIN, TIER_MEMORY_MAX = 0.5, 0.99   # 1.0 would never forget
# answers measured: at 128 tokens the rounds of one start spread by 4-10%, at 512 by under 1% (first GPU run)
MAX_NEW = 512
WARM_UP_MAX_NEW = 128              # warm-ups only need decode windows and warm caches
MIN_TOKENS = 8                     # a request with no more completion tokens than this gives no sample
# the measured conversations: one user turn per round
CONVERSATIONS = (
    ("Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
     "Now make it merge any number of sorted lists with a heap, and update the tests.",
     "Add type hints and explain the time complexity of both versions."),
    ("Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
     "How does a heat pump that heats a house differ, and why can it deliver more heat than the electricity it uses?",
     "Write a short comparison of the two as a table, one row per main component."),
    ("List twelve European capitals with one sentence about each.",
     "Now group those capitals by the river or coast they lie on, where one applies.",
     "Pick three of them and plan a five-day train trip between them, with travel times."),
)
WARM_UP_PROMPTS = (
    "Describe how a hash table handles collisions.",
    "Summarise the causes of the French Revolution in one paragraph.",
    "Write a shell one-liner that counts the lines of every .py file in a tree, and explain it.",
)
DEFAULT = "default"
CHOSEN = "chosen"
READY_TIMEOUT = 900
REQUEST_TIMEOUT = 600
POLL_SECONDS = 1.0
LOG_TAIL = 5
# engine.cpp's defaults for the flags that print no resolved value at start-up
ENGINE_DEFAULTS = {"--spec-min-p": "0.0", "--adapt-every": "4", "--adapt-swaps": "96", "--adapt-decay": "0.7"}
# engine.cpp's log: PcieAuto's update, then the start-up share (probe, failed probe, canonical pack) and pool size
PCIE_SETTLED = re.compile(r"PCIe share: .*-> pcie_frac ([\d.]+)")
PCIE_START = re.compile(r"pcie_frac (?:default )?([\d.]+)")
POOL_WORKERS = re.compile(r"(\d+) expert-pool workers")
TUNED = ("--pcie-frac", "--spec-min-p", "--pool-workers", *ADAPT_FLAGS)
# --dry-run's assumptions: any count from 6 up has three smaller candidates; 0.55 is a native pack's start-up share
DRY_RUN_WORKERS = 8
DRY_RUN_LOG = ["strata generate: PCIe probe: (dry run) -> pcie_frac 0.55 (default 0.55)",
               f"strata generate: {DRY_RUN_WORKERS} expert-pool workers + the host thread"]
DRY_RUN_GAIN = 1.1
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def flag_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def with_flags(args: list[str], settings: dict[str, str]) -> list[str]:
    """`args` with each `flag value` of `settings`: the value replaced where the flag is, else appended."""
    out = list(args)
    for flag, value in settings.items():
        if flag in out[:-1]:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    return out


def without_flags(args: list[str], flags) -> list[str]:
    """`args` without each of `flags` and its value."""
    out, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def fastest(trials: dict[str, dict]) -> str:
    """The label of the fastest measurement; the default unless one is faster."""
    top = max(trials, key=lambda label: median(trials[label]["rates"]))
    return top if median(trials[top]["rates"]) > median(trials[DEFAULT]["rates"]) else DEFAULT


class Climb:
    """Whether a climb goes on after a step: faster than the best so far, or the first step in a row that is not
    faster but within FLAT_BAND of the best."""

    def __init__(self, best: float):
        self.best = best
        self.flat = False

    def goes_on(self, rate: float) -> bool:
        if rate > self.best:
            self.best, self.flat = rate, False
            return True
        if not self.flat and rate >= self.best * (1.0 - FLAT_BAND):
            self.flat = True
            return True
        return False


def climb_step(runner, engine: list[str], base: dict, flag: str, center: float, step: float, low: float,
               high: float, label, coupled=None, measure_center: bool = False) -> dict:
    """`base`, a measurement at `center`, against `flag` at `center - step` and at `center + step`, then one `step` at
    a time in the faster direction while it improves (`Climb`), within `low`..`high`; the winner is the fastest
    (`fastest`). `label` formats a value as the flag takes it; `coupled`, if given, maps a value to other
    flags set with it; with `measure_center`, `center` itself is measured first, as a candidate."""
    began = time.monotonic()
    trials = {DEFAULT: base}

    def measure(value: float) -> dict:
        settings = base["settings"] | {flag: label(value)} | (coupled(value) if coupled else {})
        trial = runner.measure(flag.lstrip("-"), label(value), with_flags(engine, settings), ROUNDS, MAX_NEW)
        trials[label(value)] = trial | {"settings": settings}
        return trials[label(value)]

    def rate(trial: dict) -> float:
        return median(trial["rates"])

    def inside(value: float) -> bool:
        return low <= round(value, 3) <= high

    best = rate(base)
    if measure_center:
        best = max(best, rate(measure(center)))
    sides = {d: measure(center + d) for d in (step, -step) if inside(center + d)}
    if sides:
        direction = max(sides, key=lambda d: rate(sides[d]))
        climb = Climb(best)
        value, last = center + direction, sides[direction]
        while climb.goes_on(rate(last)) and inside(value + direction):
            value += direction
            last = measure(value)
    winner = fastest(trials)
    return {"name": flag, "trials": trials, "winner": winner, "seconds": time.monotonic() - began}


def settled_share(log: list[str]) -> tuple[float | None, str | None]:
    """PcieAuto's last share in a start's log and its line, else the start-up share's."""
    for pattern, lines in ((PCIE_SETTLED, reversed(log)), (PCIE_START, log)):
        for line in lines:
            if m := pattern.search(line):
                return float(m[1]), line.strip()
    return None, None


def median(samples: list[float]) -> float:
    return statistics.median(samples) if samples else 0.0


def beats(samples: list[float], than: list[float]) -> bool:
    return median(samples) > median(than) * (1.0 + MIN_GAIN)


def pick(measured: dict[str, list[float]]) -> str:
    """The best candidate, or the default unless the best beats it by more than MIN_GAIN."""
    med = {k: median(v) for k, v in measured.items() if v}
    if DEFAULT not in med:
        return DEFAULT
    top = max(med, key=med.get)
    return top if beats(measured[top], measured[DEFAULT]) else DEFAULT


class Server:
    """Starts renza per measurement and measures it over HTTP."""

    def __init__(self, server_args: list[str], url: str, api_key: str | None):
        self.server_args = server_args
        self.url = url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.starts = 0
        self.ready_starts = 0

    def measure(self, step: str, label: str, engine_args: list[str], rounds: int, max_new: int) -> dict:
        """A fresh start's rounds (tok/s each; none if it did not start), its log and PcieAuto's lines in the rounds."""
        began = time.monotonic()
        self.starts += 1
        log: list[str] = []
        proc = subprocess.Popen(self.server_args + ["--"] + engine_args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        reader = threading.Thread(target=log.extend, args=(proc.stdout,), daemon=True)
        reader.start()
        rates: list[float] = []
        measured_from = 0
        try:
            model = self.wait_ready(proc)
            if model is not None:
                warm_ups = FIRST_WARM_UPS if self.ready_starts == 0 else LATER_WARM_UPS
                if "--pcie-frac" not in engine_args:
                    warm_ups = max(warm_ups, SETTLE_WARM_UPS)
                for _ in range(warm_ups):
                    for prompt in WARM_UP_PROMPTS:
                        self.decode(model, [{"role": "user", "content": prompt}], WARM_UP_MAX_NEW)
                self.ready_starts += 1
                measured_from = len(log)
                rates = self.conversations(model, rounds, max_new)
        finally:
            proc.terminate()
            proc.wait()
            reader.join()
        if rates:
            print(f"  [{self.starts}] {step} {label}: {' '.join(f'{r:.1f}' for r in rates)} tok/s "
                  f"({time.monotonic() - began:.0f} s with the start)", flush=True)
        else:
            tail = "\n      ".join(line.rstrip() for line in log[-LOG_TAIL:])
            print(f"  [{self.starts}] {step} {label}: the server did not start, candidate dropped\n      {tail}",
                  flush=True)
        measured = log[measured_from:] if rates else []
        moved = [line.strip() for line in measured if PCIE_SETTLED.search(line)]
        return {"rates": rates, "starts": [self.starts], "log": log, "moved": moved}

    def wait_ready(self, proc: subprocess.Popen) -> str | None:
        """The served model's id once /v1/models lists it; None if the server exits or times out first."""
        deadline = time.monotonic() + READY_TIMEOUT
        request = urllib.request.Request(self.url + "/v1/models", headers=self.headers)
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with OPENER.open(request, timeout=POLL_SECONDS * 5) as response:
                    models = json.load(response).get("data")
                if models:
                    return models[0]["id"]
            except (OSError, ValueError):
                pass
            time.sleep(POLL_SECONDS)
        return None

    def conversations(self, model: str, rounds: int, max_new: int) -> list[float]:
        """One rate per round: the median over the conversations of that round's turn."""
        histories: list[list[dict]] = [[] for _ in CONVERSATIONS]
        rates = []
        for turn in range(rounds):
            round_rates = []
            for history, turns in zip(histories, CONVERSATIONS):
                history.append({"role": "user", "content": turns[turn]})
                rate, answer = self.decode(model, history, max_new)
                history.append({"role": "assistant", "content": answer})
                if rate is not None:
                    round_rates.append(rate)
            rates.append(median(round_rates))
        return rates

    def decode(self, model: str, messages: list[dict], max_new: int) -> tuple[float | None, str]:
        """Decode tok/s of one streamed request (None if too short) and the answer's text."""
        body = {"model": model, "messages": messages, "stream": True,
                "temperature": 0, "max_tokens": max_new, "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": True}}
        request = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                         self.headers | {"Content-Type": "application/json"})
        first = last = None
        tokens = 0
        answer = []
        with OPENER.open(request, timeout=REQUEST_TIMEOUT) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line.removeprefix("data: "))
                if chunk.get("usage"):
                    tokens = chunk["usage"]["completion_tokens"]
                deltas = [c.get("delta", {}) for c in chunk.get("choices", [])]
                answer += [d["content"] for d in deltas if d.get("content")]
                if any(d.get("content") or d.get("reasoning_content") for d in deltas):
                    last = time.monotonic()
                    first = first if first is not None else last
        if tokens <= MIN_TOKENS or first is None or last <= first:
            return None, "".join(answer)
        return (tokens - 1) / (last - first), "".join(answer)


class DryRun:
    """Prints each command instead of starting it; each start is faster than every earlier one, so all steps run."""

    def __init__(self, server_args: list[str]):
        self.server_args = server_args
        self.starts = 0

    def measure(self, step: str, label: str, engine_args: list[str], rounds: int, max_new: int) -> dict:
        self.starts += 1
        print(f"{self.starts:3d} {step} {label}: {shlex.join(self.server_args + ['--'] + engine_args)}")
        return {"rates": [DRY_RUN_GAIN ** self.starts] * rounds, "starts": [self.starts], "log": DRY_RUN_LOG,
                "moved": []}


def defaults(log: list[str]) -> dict[str, tuple[str | None, str]]:
    """Each tuned flag's default and its source: the engine's own, or the log of a start with no tuned flag."""
    out = {flag: (value, "engine default") for flag, value in ENGINE_DEFAULTS.items()}
    for flag, pattern, source in (("--pool-workers", POOL_WORKERS, "start-up log"),
                                  ("--pcie-frac", PCIE_START, "start-up log, then PcieAuto (automatic)")):
        found = [m[1] for line in log if (m := pattern.search(line))]
        out[flag] = (found[0], source) if found else (None, "unknown: not in the start-up log")
    return out


def combine(runner, engine: list[str], base: dict, picks: list[dict]) -> dict | None:
    """`picks` (measurements faster than the defaults), highest gain first, added one at a time to the defaults
    (`base`); an addition stays if the set is faster with it than without. None without picks."""
    began = time.monotonic()
    picks = sorted(picks, key=lambda trial: median(trial["rates"]), reverse=True)
    if not picks:
        return None

    def label(settings: dict[str, str]) -> str:
        return "+ " + shlex.join(sum(settings.items(), ()))

    current = picks[0]
    trials = {DEFAULT: base, label(current["settings"]): current}
    winner = label(current["settings"])
    for pick_ in picks[1:]:
        settings = current["settings"] | pick_["settings"]
        trial = runner.measure("combine", label(pick_["settings"]), with_flags(engine, settings), ROUNDS, MAX_NEW)
        trials[label(pick_["settings"])] = trial | {"settings": settings}
        if median(trial["rates"]) > median(current["rates"]):
            current, winner = trials[label(pick_["settings"])], label(pick_["settings"])
    return {"name": "combination, highest isolated gain first", "trials": trials, "winner": winner,
            "seconds": time.monotonic() - began}


def calibrate(runner, engine: list[str]) -> tuple[list[dict], dict, dict[str, str]]:
    """The steps' measurements, the defaults and the chosen settings (flag -> value)."""
    steps = []
    began = time.monotonic()
    base = runner.measure("pool-workers", DEFAULT, engine, ROUNDS, MAX_NEW) | {"settings": {}}
    base_seconds = time.monotonic() - began
    found = defaults(base["log"])
    workers = found["--pool-workers"][0]
    if workers:
        step = climb_step(runner, engine, base, "--pool-workers", int(workers), 1, MIN_WORKERS, math.inf,
                          lambda value: str(round(value)))
    else:
        print("  --pool-workers skipped: the engine count is unknown")
        step = {"trials": {DEFAULT: base}, "winner": DEFAULT, "seconds": 0.0}
    step["name"] = f"--pool-workers, the share on automatic (engine count {workers})"
    step["seconds"] += base_seconds
    steps.append(step)

    share, line = settled_share(base["log"])
    if share is None:
        print("  --pcie-frac skipped: no share in the first start's log")
    else:
        step = climb_step(runner, engine, base, "--pcie-frac", share, PCIE_STEP, 0.0, 1.0,
                          lambda value: f"{value:.2f}")
        step["name"] = f"--pcie-frac in {PCIE_STEP} steps from {share:.2f}, fixed (default: automatic)"
        step["note"] = f"share in the first start: {line}"
        steps.append(step)

    step = climb_step(runner, engine, base, "--spec-min-p", SPEC_START, SPEC_STEP, SPEC_STEP, SPEC_MAX,
                      lambda v: f"{v:.1f}", measure_center=True)
    step["name"] = f"--spec-min-p from {SPEC_START} (default: {ENGINE_DEFAULTS['--spec-min-p']})"
    steps.append(step)
    picks = [step["trials"][step["winner"]] for step in steps if step["winner"] != DEFAULT]

    # the tier per decode window: how often it adapts, how many experts it swaps per window, how much of the usage
    # it keeps per window (the swaps and the decay apply per adaptation)
    every = float(ENGINE_DEFAULTS["--adapt-every"])
    rate = float(ENGINE_DEFAULTS["--adapt-swaps"]) / every
    memory = float(ENGINE_DEFAULTS["--adapt-decay"]) ** (1 / every)

    def tier(every: float, rate: float, memory: float) -> dict[str, str]:
        return {"--adapt-every": str(round(every)), "--adapt-swaps": str(round(rate * every)),
                "--adapt-decay": f"{memory ** every:.3f}"}

    step = climb_step(runner, engine, base, "--adapt-every", every, 1, 1, math.inf, lambda v: str(round(v)),
                      lambda v: tier(v, rate, memory))
    step["name"] = f"tier: --adapt-every from {round(every)}, {rate:g} swaps and {memory:.3f} memory per window"
    steps.append(step)
    current = step["trials"][step["winner"]]
    every = float(current["settings"].get("--adapt-every", every))
    step = climb_step(runner, engine, current, "--adapt-swaps", rate, TIER_RATE_STEP, TIER_RATE_STEP, math.inf,
                      lambda v: str(round(v * every)), lambda v: tier(every, v, memory))
    step["name"] = f"tier: swaps per window in {TIER_RATE_STEP} steps at --adapt-every {round(every)}"
    steps.append(step)
    current = step["trials"][step["winner"]]
    rate = float(current["settings"].get("--adapt-swaps", rate * every)) / every
    step = climb_step(runner, engine, current, "--adapt-decay", memory, TIER_MEMORY_STEP, TIER_MEMORY_MIN,
                      TIER_MEMORY_MAX, lambda v: f"{v ** every:.3f}", lambda v: tier(every, rate, v))
    step["name"] = f"tier: memory per window in {TIER_MEMORY_STEP} steps at --adapt-every {round(every)}"
    steps.append(step)
    current = step["trials"][step["winner"]]
    if current is not base:
        picks.append(current)

    combination = combine(runner, engine, base, picks)
    if combination is None:
        print("  combination and confirmation skipped: no step found a value faster than the default")
        return steps, found, {}
    steps.append(combination)
    chosen = combination["trials"][combination["winner"]]["settings"]

    began = time.monotonic()
    pair = {DEFAULT: {}, CHOSEN: chosen}
    trials = {label: {"rates": [], "starts": [], "moved": []} for label in pair}
    for label in [DEFAULT, CHOSEN] * CONFIRM_STARTS:
        trial = runner.measure("confirm", label, with_flags(engine, pair[label]), ROUNDS, MAX_NEW)
        for key in trials[label]:
            trials[label][key] += trial[key]
    kept = pick({label: trial["rates"] for label, trial in trials.items()})
    steps.append({"name": f"confirmation of {shlex.join(sum(chosen.items(), ()))} against no tuned flag",
                  "trials": trials, "winner": kept, "verdict": "kept",
                  "seconds": time.monotonic() - began})
    return steps, found, pair[kept]


def report(steps: list[dict], found: dict, settings: dict[str, str], starts: int, seconds: float):
    first = 1
    for step in steps:
        trials = step["trials"]
        print(f"\n{step['name']} ({step['seconds']:.0f} s)")
        if "note" in step:
            print(f"  {step['note']}")
        for label, trial in trials.items():
            samples = trial["rates"]
            rounds = " ".join(f"{s:.1f}" for s in samples) or "no sample"
            where = ", ".join(str(n) for n in trial["starts"])
            reused = " reused" if trial["starts"][0] < first else ""
            print(f"  {label:>10}: {rounds:<24} median {median(samples):6.1f}  (start {where}{reused})")
            for line in trial["moved"]:
                print(f"{'':14}PcieAuto moved the share during the measured rounds: {line}")
        base = median(trials[DEFAULT]["rates"])
        rate = median(trials[step["winner"]]["rates"])
        gain = f", {rate / base - 1.0:+.1%} over the default" if base > 0 and step["winner"] != DEFAULT else ""
        print(f"  {step.get('verdict', 'best')}: {step['winner']} ({rate:.1f} tok/s{gain})")
        first = max(n for trial in trials.values() for n in trial["starts"]) + 1
    print("\ndefaults:")
    for flag, (value, source) in found.items():
        print(f"  {flag} {value if value is not None else '?'} ({source})")
    print(f"\n{starts} starts, {seconds / 60:.1f} min in total")
    print(shlex.join(sum(settings.items(), ())))


def main():
    ap = argparse.ArgumentParser(usage="uv run tools/calibrate.py [--dry-run] -- RENZA-COMMAND ...",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the commands a run would start; start nothing")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    options = ap.parse_args(argv[:cut])
    command = argv[cut + 1:]
    if "--" not in command:
        ap.error("the renza command has no engine part (its own `-- ENGINE FLAGS`)")
    server, engine = command[:command.index("--")], command[command.index("--") + 1:]
    port = flag_value(server, "--port")
    if port is None:
        ap.error("the renza command has no --port")
    host = flag_value(server, "--host") or "127.0.0.1"
    host = "127.0.0.1" if host == "0.0.0.0" else host
    removed = [(f, v) for f in TUNED if (v := flag_value(engine, f)) is not None]
    if removed:
        print("removed from the command (measured from the engine's defaults): "
              + shlex.join(sum(removed, ())))
    engine = without_flags(engine, TUNED)

    if options.dry_run:
        print(f"dry run, assumed: every candidate is faster than the best so far (the most starts), an engine count of "
              f"{DRY_RUN_WORKERS} pool workers, the share settled at 0.55 (a native pack's start-up default)")
        runner = DryRun(server)
        calibrate(runner, engine)
        print(f"{runner.starts} starts under these assumptions")
        return
    with socket.socket() as probe:
        if probe.connect_ex((host, int(port))) == 0:
            sys.exit(f"port {port} is in use: stop the server on it first")
    runner = Server(server, f"http://{host}:{port}", flag_value(server, "--api-key"))
    began = time.monotonic()
    steps, found, settings = calibrate(runner, engine)
    report(steps, found, settings, runner.starts, time.monotonic() - began)


if __name__ == "__main__":
    main()
