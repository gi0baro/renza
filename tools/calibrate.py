# /// script
# requires-python = ">=3.14"
# ///
"""Tune renza's engine settings on this machine by measuring decode and prompt speed.

    uv run tools/calibrate.py [--phase decode|prefill|both] [--dry-run] -- RENZA-COMMAND ... -- ENGINE-FLAGS ...

The command is the full renza command line, its engine part after its own `--`. A setting is an engine flag or an
environment variable of the engine. The tool starts the command once per measurement (the candidate's flags replaced,
or added if absent, the rest as given; the candidate's variables set in the server's environment, the rest of the
environment inherited), waits for /v1/models, sends greedy streamed chat completions with thinking off, and stops the
server before the next start. URL, port and API key come from the command's --host, --port and --api-key.

The settings a chosen phase tunes are removed first, the flags from the command and the variables from the inherited
environment, so every setting is measured against the engine's own default (after an engine change, the old
calibration is not the baseline). --phase decode tunes decode speed, prefill the time to read prompts, both (the
default) runs decode and then prefill on the decode result.

Decode phase. Steps 1-4 each change one setting in isolation: every candidate is the command with no tuned setting
plus that setting, against that command as measured in the first start (reused). A climb measures both neighbours of
its starting value, then one step at a time in the faster direction while it improves: a step faster than the best so
far continues; one not faster but within 2% of the best (FLAT_BAND) gets one more step, so a flat first step does not
end the climb; a step more than 2% below the best, or the second step in a row that is not faster, ends it.
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
     other two per-window quantities; they start at the engine's (24 swaps and 0.7 ** (1/4) per window). Then, at
     the tier's result, STRATA_ADAPT_FETCH unset (off), 1 (the PCIe share also admits experts into the tier) and 2
     (instead of the tier's own copies), a start each; only if a mode is faster than unset, two climbs at that mode
     on the admission's thresholds: STRATA_ADAPT_FETCH_MIN from the engine's 1.0 in steps of 0.5 (at least 0.5),
     then STRATA_ADAPT_FETCH_MARGIN from 0.5 in steps of 0.25 (at least 0) at the MIN chosen. The step's last
     result (the tier's, or the tier's with the fetch settings) enters the combination together.
  5. Combination: the fastest value of steps 1-3 and the result of step 4, each when it beat the default, sorted by
     that gain, added one at a time to the defaults; each addition is kept if the set is faster with it than
     without, else dropped. The first addition is its own start (the defaults plus that value), reused.
  6. Leave-one-out confirmation: for each pick in the final combination, one start of that final set minus the
     pick, compared with the combination's measurement (reused). The tier's three flags are one pick and the
     fetch mode with its thresholds another; each other setting is one pick. A pick stays only if the set with it
     beats the set without it by more than 3% (MIN_GAIN). All comparisons use the same final combination, not the
     set after earlier drops; failed picks are removed together, with no further rounds. A pick whose comparison
     start fails is dropped (no evidence for it); a single pick has no start of its own (its set minus it is the
     base, which the last comparison measures). If every pick is dropped the result is the base, with no further
     start. Otherwise the confirmed set against
     the command with no tuned setting, a fresh start each, back to back: the set stays only if more than 3%
     faster, else the result is the engine's defaults. Start-ups, the adaptive tier and the OS move single
     measurements by a few percent. Steps 1-5 only choose what to measure next.
Decode tok/s of one request: its completion tokens minus one over the time from the first streamed text to the last.
The measured rounds are the turns of three conversations, as an agent's session grows: round N sends turn N of each
conversation with the earlier turns and the model's own answers as history, so no start answers the same text twice
(answering it again lets the expert tier learn those very texts) and the server reuses each conversation's prefix.
Every start goes through the same turns, so starts compare. The warm-ups use prompts of their own. A round is the
median over the conversations; a candidate's value is the median over its rounds.

Prefill phase. Its base is the command as given with the decode result (both), or the command and the environment
as given (prefill: their decode settings stay), always without --kv-disk-dir and --kv-disk-space-mb, so no start reads
a prompt's KV that an earlier start stored. Every start runs one warm-up prompt, then the same workload: cold prompts
of about 8K, 32K and 128K tokens, each a fresh conversation, then one conversation that opens at about 32K tokens and
grows to about 120K in turns of 2K, 3K and 4K new tokens (in turn), with the model's answers as history, so the server
reuses its prefix as an agent's session does. The text is renza's own sources (engine/strata/src and server/, files
in sorted path order, concatenated), each prompt its own slice of it, followed by a short question; answers are 16
tokens at most. Sizes are estimated at CHARS_PER_TOKEN (no tokenizer); the engine's lines give the real counts. A
start's value is its total prompt time: the sum of T over the workload's lines `strata serve: prompt N tokens = R
reused + X read in T ms`; lower is better (every comparison below is on that total; the report gives the cold prompts,
the growing conversation's first turn included, and the later turns apart, read tokens over ms). Each candidate
alone against the phase's base (the first start, reused):
  1. STRATA_SELECT_TWO_PASS unset (off) / 1.
  2. STRATA_IQ4_W4A8 unset (off) / 1 (the IQ4_XS and IQ4_NL products) / 2 (the IQ4_NL down products only).
  3. STRATA_PREFILL_CPU_SHARE unset (the engine's default: on where it arms it) / 0 (off); if 0 is not faster and the
     base's log shows the share on, a climb of STRATA_PREFILL_CPU_SHARE_MAX from the limit that log gives (3072 on
     this machine) in steps of 1024 (at least 1024), with the decode phase's climb rules.
  4. STRATA_NV_CHUNK unset (chunks up to 32768) / 16 (16384) / 0 (8192).
The two-pass scores and the W4A8 kernel give the same output bits: each is kept when it is not slower than what it is
compared with (its total prompt time at most the other's). The CPU share and the chunk ceiling change the rounding:
each is kept when faster, and the final confirmation asks them for more than MIN_GAIN. Combination as decode step 5
with these rules: each step's winner that beat the base (two-pass, W4A8: was not slower), highest gain first, added one
at a time; an addition stays when the set is faster with it (two-pass, W4A8: not slower). Each knob is one pick
(the CPU share and its _MAX together). Leave-one-out confirmation as in decode: one start of the final combination
minus each pick, against the combination's reused measurement. Two-pass and W4A8 stay when the set with the pick is
not slower; every other pick needs more than MIN_GAIN. Failed picks are removed together, with no further rounds.
Then a fresh start of the base and one of the confirmed set, back to back: any non-identical pick in the set asks
for more than MIN_GAIN over the base; only identical picks ask for not slower. Otherwise the result is the base.

It prints each step's candidates (decode: every round's tok/s and the median; prefill: the total prompt ms, cold and
later turns), what each step kept, the share PcieAuto settled on and any move of the share during a start's measured
rounds, the confirmations, where each default comes from, the starts and the wall time, and as the last line the
tuned settings in one line: the variables as NAME=value, `--`, then the engine flags, to use in place of any the
command and its environment had (nothing on either side: the engine's defaults). In run-renza.sh the variables are
exported before its `exec` and the flags go into the engine part of `args` (after its `--`). It writes no file.
--dry-run prints every command a run would start (with its variables), with its step, the prefill workload's sizes
and an estimate of the wall time, and starts nothing: it cannot know the results, so it assumes the most starts
(every candidate improves, except STRATA_PREFILL_CPU_SHARE=0, so that the climb of its limit is listed; leave-one-out
starts are slower, so all picks stay for the final comparison), a share
settled at a native pack's start-up default, and START_SECONDS, the request times and the prompt speeds below.
"""
import argparse
import json
import math
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

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
# the adaptive tier: every --adapt-every decode windows it swaps up to --adapt-swaps of the most-routed missing
# experts into VRAM, then multiplies the usage counts by --adapt-decay (engine.cpp)
ADAPT_FLAGS = ("--adapt-every", "--adapt-swaps", "--adapt-decay")
TIER_RATE_STEP = 8                 # swaps per window (32 swaps at every 4 windows)
TIER_MEMORY_STEP = 0.02            # usage kept per window
TIER_MEMORY_MIN, TIER_MEMORY_MAX = 0.5, 0.99   # 1.0 would never forget
# the PCIe share's admission into the tier (engine.cpp adapt_fetch): 1 beside the tier's copies from RAM, 2 instead
ADAPT_FETCH = "STRATA_ADAPT_FETCH"
FETCH_MODES = ("1", "2")
FETCH_MIN, FETCH_MARGIN = "STRATA_ADAPT_FETCH_MIN", "STRATA_ADAPT_FETCH_MARGIN"
FETCH_MIN_DEFAULT, FETCH_MIN_STEP, FETCH_MIN_LOW = 1.0, 0.5, 0.5
FETCH_MARGIN_DEFAULT, FETCH_MARGIN_STEP, FETCH_MARGIN_LOW = 0.5, 0.25, 0.0
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
# the prefill knobs; two-pass and W4A8 keep the output bits (qsa_two_pass.hpp's bound, MMQ's bits), the others not
TWO_PASS, W4A8 = "STRATA_SELECT_TWO_PASS", "STRATA_IQ4_W4A8"
CPU_SHARE, CPU_SHARE_MAX, NV_CHUNK = "STRATA_PREFILL_CPU_SHARE", "STRATA_PREFILL_CPU_SHARE_MAX", "STRATA_NV_CHUNK"
IDENTICAL = (TWO_PASS, W4A8)
SHARE_MAX_STEP, SHARE_MAX_LOW = 1024, 1024     # the engine takes no limit below 1024 (prefill.cpp cpu_share_max)
NV_CHUNKS = {"16": 16384, "0": 8192}           # STRATA_NV_CHUNK -> the chunk ceiling (nv_chunk.hpp); unset: 32768
# the prefill workload: the text, renza's own sources (relative to the repository), and its sizes in tokens
SOURCE_DIRS = ("engine/strata/src", "server")
CHARS_PER_TOKEN = 3.3              # the model's tokenizer on these sources: 3.28 over the whole workload
PREFILL_WARM_UP_TOKENS = 8000
COLD_TOKENS = (8000, 32000, 128000)
GROW_FROM, GROW_TO = 32000, 120000
TURN_TOKENS = (2000, 3000, 4000)   # new tokens per turn, in turn: on both sides of the CPU share's 3072 limit
PREFILL_MAX_NEW = 16
QUESTION = "Above are source files of a project, concatenated. In one short sentence: what does this code do?"
FOLLOW_UP = "Above are more files of the same project. In one short sentence: what do these add?"
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
# the engine's line per request (generate.cpp), the default CPU share's limit (prefill.cpp) and the chunk ceiling
PROMPT_TIMING = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
SHARE_ON = re.compile(r"the CPU share is ON by default for prompt chunks up to (\d+) tokens")
CHUNK_CEILING = re.compile(r"--prefill auto may take chunks up to (\d+)")
TUNED = ("--pcie-frac", "--spec-min-p", "--pool-workers", *ADAPT_FLAGS)
DECODE_VARIABLES = (ADAPT_FETCH, FETCH_MIN, FETCH_MARGIN)
PREFILL_VARIABLES = (TWO_PASS, W4A8, CPU_SHARE, CPU_SHARE_MAX, NV_CHUNK)
KV_DISK_FLAGS = ("--kv-disk-dir", "--kv-disk-space-mb")
# --dry-run's assumptions: any count from 6 up has three smaller candidates; 0.55 is a native pack's start-up share
DRY_RUN_WORKERS = 8
DRY_RUN_LOG = ["strata generate: PCIe probe: (dry run) -> pcie_frac 0.55 (default 0.55)",
               f"strata generate: {DRY_RUN_WORKERS} expert-pool workers + the host thread",
               "strata generate: NVIDIA: --prefill auto may take chunks up to 32768",
               "prefill: the CPU share is ON by default for prompt chunks up to 3072 tokens"]
DRY_RUN_GAIN = 1.1
DRY_RUN_SLOWER = {CPU_SHARE: "0"}
# the estimate: a start until /v1/models answers (0.2.13 here: 111-113 s from one start's end to the next's readiness,
# GPU session of 2026-11-10), a 128-token warm-up and a 512-token answer (fourth calibration run: 30 starts, ~70 s
# each), the prompt speeds of 8-128K cold prompts (3000-3900 tok/s) and of 1.5-1.8K turns at 20K (700-1300 tok/s)
START_SECONDS = 110.0
WARM_UP_SECONDS, ANSWER_SECONDS, SHORT_ANSWER_SECONDS = 1.7, 5.2, 0.5
COLD_TOKENS_PER_SECOND, TURN_TOKENS_PER_SECOND = 3300.0, 1000.0
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


def flags_of(settings: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in settings.items() if k.startswith("-")}


def variables_of(settings: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in settings.items() if not k.startswith("-")}


def shown(settings: dict[str, str]) -> str:
    """`settings` as a shell takes them: the variables as NAME=value, then the flags."""
    return shlex.join([f"{k}={v}" for k, v in variables_of(settings).items()]
                      + list(sum(flags_of(settings).items(), ())))


def changes(settings: dict[str, str], base: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in settings.items() if base.get(k) != v}


def speed(trial: dict) -> float:
    return trial["speed"]


def fastest(trials: dict[str, dict], identical: bool = False) -> str:
    """The label of the fastest candidate if it is faster than the default (`identical`: not slower), else the
    default."""
    others = [label for label in trials if label != DEFAULT]
    if not others:
        return DEFAULT
    top = max(others, key=lambda label: speed(trials[label]))
    if speed(trials[top]) > speed(trials[DEFAULT]) or identical and speed(trials[top]) >= speed(trials[DEFAULT]) > 0:
        return top
    return DEFAULT


def beats(trial: dict, than: dict) -> bool:
    return speed(trial) > speed(than) * (1.0 + MIN_GAIN)


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


def climb_step(runner, phase, base: dict, name: str, center: float, step: float, low: float, high: float, label,
               coupled=None, measure_center: bool = False) -> dict:
    """`base`, a measurement at `center`, against setting `name` at `center - step` and at `center + step`, then one
    `step` at a time in the faster direction while it improves (`Climb`), within `low`..`high`; the winner is the
    fastest (`fastest`). `label` formats a value as the setting takes it; `coupled`, if given, maps a value to other
    settings set with it; with `measure_center`, `center` itself is measured first, as a candidate."""
    began = time.monotonic()
    trials = {DEFAULT: base}

    def measure(value: float) -> dict:
        settings = base["settings"] | {name: label(value)} | (coupled(value) if coupled else {})
        trials[label(value)] = runner.measure(name.lstrip("-"), label(value), phase, settings)
        return trials[label(value)]

    def inside(value: float) -> bool:
        return low <= round(value, 3) <= high

    best = speed(base)
    if measure_center:
        best = max(best, speed(measure(center)))
    sides = {d: measure(center + d) for d in (step, -step) if inside(center + d)}
    if sides:
        direction = max(sides, key=lambda d: speed(sides[d]))
        climb = Climb(best)
        value, last = center + direction, sides[direction]
        while climb.goes_on(speed(last)) and inside(value + direction):
            value += direction
            last = measure(value)
    winner = fastest(trials)
    return {"name": name, "trials": trials, "winner": winner, "seconds": time.monotonic() - began}


def discrete_step(runner, phase, base: dict, name: str, values: dict[str, str], identical: bool = False) -> dict:
    """`base` against setting `name` at each of `values` (value -> its label), a start each."""
    began = time.monotonic()
    trials = {DEFAULT: base}
    for value, label in values.items():
        trials[label] = runner.measure(name, label, phase, base["settings"] | {name: value})
    return {"name": name, "trials": trials, "winner": fastest(trials, identical), "seconds": time.monotonic() - began}


def settled_share(log: list[str]) -> tuple[float | None, str | None]:
    """PcieAuto's last share in a start's log and its line, else the start-up share's."""
    for pattern, lines in ((PCIE_SETTLED, reversed(log)), (PCIE_START, log)):
        for line in lines:
            if m := pattern.search(line):
                return float(m[1]), line.strip()
    return None, None


def median(samples: list[float]) -> float:
    return statistics.median(samples) if samples else 0.0


def source_text(root: Path) -> tuple[str, int]:
    """The files of SOURCE_DIRS in sorted path order, each after a line with its path, and their count."""
    paths = sorted(p for d in SOURCE_DIRS for p in (root / d).rglob("*") if p.is_file())
    parts = [f"=== {p.relative_to(root)} ===\n{p.read_text(encoding='utf-8', errors='replace')}" for p in paths]
    return "\n".join(parts), len(paths)


def prefill_workload(root: Path) -> dict:
    """The prefill phase's prompts: each its own slice of the sources (about the size asked, ended at a line's end)
    and a question after it, so no prompt shares a prefix with another one; sizes in estimated tokens."""
    text, files = source_text(root)
    cursor = 0

    def prompt(tokens: int, question: str) -> str:
        nonlocal cursor
        end = text.find("\n", cursor + round(tokens * CHARS_PER_TOKEN))
        if end < 0:
            sys.exit(f"the sources ({len(text)} characters) are too short for the prefill workload")
        part, cursor = text[cursor:end], end + 1
        return f"{part}\n\n{question}"

    warm_up = prompt(PREFILL_WARM_UP_TOKENS, QUESTION)
    cold = [prompt(tokens, QUESTION) for tokens in COLD_TOKENS]
    growing = [prompt(GROW_FROM, QUESTION)]
    total = GROW_FROM
    while True:
        tokens = TURN_TOKENS[(len(growing) - 1) % len(TURN_TOKENS)]
        if total + tokens > GROW_TO:
            break
        growing.append(prompt(tokens, FOLLOW_UP))
        total += tokens
    return {"warm_up": warm_up, "cold": cold, "growing": growing, "files": files, "chars": len(text),
            "used": cursor}


def estimated_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


class Decode:
    """The decode phase's workload and its report."""
    name = "decode"
    base_label = DEFAULT

    def __init__(self, server_args: list[str], engine: list[str]):
        self.server_args, self.engine, self.base_settings = server_args, engine, {}

    def command(self, settings: dict[str, str]) -> list[str]:
        return self.server_args + ["--"] + with_flags(self.engine, flags_of(settings))

    def warm_ups(self, settings: dict[str, str], first: bool) -> int:
        warm_ups = FIRST_WARM_UPS if first else LATER_WARM_UPS
        if "--pcie-frac" not in with_flags(self.engine, flags_of(settings)):
            warm_ups = max(warm_ups, SETTLE_WARM_UPS)
        return warm_ups

    def run(self, server, model: str, log: list[str], settings: dict[str, str], first: bool) -> dict:
        for _ in range(self.warm_ups(settings, first)):
            for prompt in WARM_UP_PROMPTS:
                server.chat(model, [{"role": "user", "content": prompt}], WARM_UP_MAX_NEW)
        measured_from = len(log)
        return {"rates": server.conversations(model, ROUNDS, MAX_NEW), "measured_from": measured_from}

    def finish(self, fields: dict, log: list[str]) -> dict | None:
        """The measurement from the workload's `fields` and the start's whole log; None if it gave no sample."""
        if not fields["rates"]:
            return None
        moved = [line.strip() for line in log[fields["measured_from"]:] if PCIE_SETTLED.search(line)]
        return {"rates": fields["rates"], "speed": median(fields["rates"]), "moved": moved}

    def failed(self) -> dict:
        return {"rates": [], "speed": 0.0, "moved": []}

    def dry(self, value: float) -> dict:
        return {"rates": [value] * ROUNDS, "speed": value, "moved": []}

    def estimate(self, settings: dict[str, str], first: bool) -> float:
        warm_ups = self.warm_ups(settings, first) * len(WARM_UP_PROMPTS)
        return START_SECONDS + warm_ups * WARM_UP_SECONDS + ROUNDS * len(CONVERSATIONS) * ANSWER_SECONDS

    def summary(self, trial: dict) -> str:
        return f"{' '.join(f'{r:.1f}' for r in trial['rates'])} tok/s"

    def row(self, trial: dict) -> str:
        rounds = " ".join(f"{s:.1f}" for s in trial["rates"]) or "no sample"
        return f"{rounds:<24} median {speed(trial):6.1f}"

    def value(self, trial: dict) -> str:
        return f"{speed(trial):.1f} tok/s"


class Prefill:
    """The prefill phase's workload and its report; a start's speed is 1000 / its total prompt ms."""
    name = "prefill"
    base_label = "base"

    def __init__(self, server_args: list[str], engine: list[str], base_settings: dict[str, str], workload: dict):
        self.server_args, self.engine, self.base_settings = server_args, engine, base_settings
        self.workload = workload
        self.cold = len(workload["cold"]) + 1          # the growing conversation's first turn is cold too
        self.requests = len(workload["cold"]) + len(workload["growing"])

    def command(self, settings: dict[str, str]) -> list[str]:
        return self.server_args + ["--"] + with_flags(self.engine, flags_of(settings))

    def run(self, server, model: str, log: list[str], settings: dict[str, str], first: bool) -> dict:
        server.chat(model, [{"role": "user", "content": self.workload["warm_up"]}], PREFILL_MAX_NEW)
        for prompt in self.workload["cold"]:
            server.chat(model, [{"role": "user", "content": prompt}], PREFILL_MAX_NEW)
        history = []
        for prompt in self.workload["growing"]:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": server.chat(model, history, PREFILL_MAX_NEW)[1]})
        return {}

    def finish(self, fields: dict, log: list[str]) -> dict | None:
        # the engine logs a request's prompt line when the next request starts or the server stops, so the
        # warm-up's line is the log's first and the last request's comes with the stop
        lines = [m for line in log if (m := PROMPT_TIMING.search(line))]
        if len(lines) != self.requests + 1:
            print(f"      {len(lines)} prompt lines in the log for the warm-up and {self.requests} requests",
                  flush=True)
            return None
        read = [(int(m[3]), int(m[4])) for m in lines[1:]]
        cold, turns = read[:self.cold], read[self.cold:]
        ms = sum(t for _, t in read)
        return {"ms": ms, "speed": 1000.0 / ms, "cold": totals(cold), "turns": totals(turns), "moved": []}

    def failed(self) -> dict:
        return {"ms": None, "speed": 0.0, "cold": (0, 0), "turns": (0, 0), "moved": []}

    def tokens(self) -> tuple[int, int]:
        """The workload's estimated cold and later-turn tokens."""
        prompts = self.workload["cold"] + self.workload["growing"]
        cold = sum(estimated_tokens(p) for p in prompts[:self.cold])
        return cold, sum(estimated_tokens(p) for p in prompts[self.cold:])

    def dry(self, value: float) -> dict:
        return {"ms": 1000.0 / value, "speed": value, "cold": (0, 0), "turns": (0, 0), "moved": []}

    def estimate(self, settings: dict[str, str], first: bool) -> float:
        cold, turns = self.tokens()
        cold += estimated_tokens(self.workload["warm_up"])
        return (START_SECONDS + cold / COLD_TOKENS_PER_SECOND + turns / TURN_TOKENS_PER_SECOND
                + (self.requests + 1) * SHORT_ANSWER_SECONDS)

    def summary(self, trial: dict) -> str:
        return f"{trial['ms']:.0f} prompt ms (cold {rate(trial['cold'])}, later turns {rate(trial['turns'])})"

    def row(self, trial: dict) -> str:
        if trial["ms"] is None:
            return f"{'no sample':<24}"
        return f"{trial['ms']:8.0f} ms  cold {rate(trial['cold'])}, later turns {rate(trial['turns'])}"

    def value(self, trial: dict) -> str:
        return f"{trial['ms']:.0f} prompt ms" if trial["ms"] is not None else "no sample"


def totals(read: list[tuple[int, int]]) -> tuple[int, int]:
    return sum(x for x, _ in read), sum(t for _, t in read)


def rate(pair: tuple[int, int]) -> str:
    tokens, ms = pair
    return f"{tokens} tokens / {ms} ms = {tokens / ms * 1000:.0f} tok/s" if ms else "none"


class Server:
    """Starts renza per measurement and measures it over HTTP."""

    def __init__(self, url: str, api_key: str | None, env: dict[str, str]):
        self.url = url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.env = env
        self.starts = 0
        self.ready_starts = 0

    def measure(self, step: str, label: str, phase, settings: dict[str, str]) -> dict:
        """A fresh start with `settings` through `phase`'s workload (`phase.failed()` if it did not start or failed
        during the workload), with its start number, its log and `settings`."""
        began = time.monotonic()
        self.starts += 1
        log: list[str] = []
        proc = subprocess.Popen(phase.command(settings), env=self.env | variables_of(settings),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")

        def read():
            for line in proc.stdout:
                log.append(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        fields, problem = None, "the server did not start"
        try:
            model = self.wait_ready(proc)
            if model is not None:
                try:
                    fields = phase.run(self, model, log, settings, self.ready_starts == 0)
                except (OSError, ValueError) as e:
                    problem = f"the server failed during the workload ({e})"
                self.ready_starts += 1
        finally:
            proc.terminate()
            proc.wait()
            reader.join()
        trial = phase.finish(fields, log) if fields is not None else None
        if trial is not None:
            print(f"  [{self.starts}] {step} {label}: {phase.summary(trial)} "
                  f"({time.monotonic() - began:.0f} s with the start)", flush=True)
        else:
            tail = "\n      ".join(line.rstrip() for line in log[-LOG_TAIL:])
            print(f"  [{self.starts}] {step} {label}: {problem if fields is None else 'no sample'}, candidate "
                  f"dropped\n      {tail}", flush=True)
            trial = phase.failed()
        return trial | {"starts": [self.starts], "log": log, "settings": settings}

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
                rate_, answer = self.chat(model, history, max_new)
                history.append({"role": "assistant", "content": answer})
                if rate_ is not None:
                    round_rates.append(rate_)
            rates.append(median(round_rates))
        return rates

    def chat(self, model: str, messages: list[dict], max_new: int) -> tuple[float | None, str]:
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
    """Prints commands instead of starting them. Candidates improve except DRY_RUN_SLOWER; leave-one-out starts
    are slower so every pick stays (the most starts). Adds up the estimated wall time."""

    def __init__(self):
        self.starts = 0
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def measure(self, step: str, label: str, phase, settings: dict[str, str]) -> dict:
        self.starts += 1
        self.seconds[phase.name] = self.seconds.get(phase.name, 0.0) + phase.estimate(settings, self.starts == 1)
        self.counts[phase.name] = self.counts.get(phase.name, 0) + 1
        variables = [f"{k}={v}" for k, v in variables_of(settings).items()]
        print(f"{self.starts:3d} {phase.name} {step} {label}: {shlex.join(variables + phase.command(settings))}")
        value = DRY_RUN_GAIN ** self.starts
        if step == "leave-one-out" or any(settings.get(k) == v for k, v in DRY_RUN_SLOWER.items()):
            value = 1.0 / value
        return phase.dry(value) | {"starts": [self.starts], "log": DRY_RUN_LOG, "settings": settings}


def defaults(log: list[str]) -> dict[str, tuple[str | None, str]]:
    """Each decode setting's default and its source: the engine's own, or the log of a start with no tuned setting."""
    out = {flag: (value, "engine default") for flag, value in ENGINE_DEFAULTS.items()}
    for flag, pattern, source in (("--pool-workers", POOL_WORKERS, "start-up log"),
                                  ("--pcie-frac", PCIE_START, "start-up log, then PcieAuto (automatic)")):
        found = [m[1] for line in log if (m := pattern.search(line))]
        out[flag] = (found[0], source) if found else (None, "unknown: not in the start-up log")
    out[ADAPT_FETCH] = ("unset", "engine default: off")
    out[FETCH_MIN] = (f"{FETCH_MIN_DEFAULT:g}", "engine default")
    out[FETCH_MARGIN] = (f"{FETCH_MARGIN_DEFAULT:g}", "engine default")
    return out


def prefill_defaults(log: list[str]) -> dict[str, tuple[str | None, str]]:
    """Each prefill setting's default and its source, from the base start's log where it says."""
    share = [m[1] for line in log if (m := SHARE_ON.search(line))]
    chunk = [m[1] for line in log if (m := CHUNK_CEILING.search(line))]
    return {TWO_PASS: ("unset", "engine default: off"), W4A8: ("unset", "engine default: off"),
            CPU_SHARE: ("unset", "base log: on by default" if share else "base log: not armed (no line)"),
            CPU_SHARE_MAX: (share[0] if share else None, "base log" if share else "unknown: the share is off"),
            NV_CHUNK: ("unset", f"base log: chunks up to {chunk[0]}" if chunk else "base log: no NVIDIA line (8192)")}


def combine(runner, phase, base: dict, picks: list[dict]) -> dict | None:
    """`picks` (measurements faster than `base`, or not slower for an output-identical setting), highest gain
    first, added one at a time to `base`; an addition stays if the set is faster with it than without (an identical
    setting: not slower). None without picks."""
    began = time.monotonic()
    picks = sorted(picks, key=speed, reverse=True)
    if not picks:
        return None

    def label(trial: dict) -> str:
        return "+ " + shown(changes(trial["settings"], phase.base_settings))

    current = picks[0]
    trials = {DEFAULT: base, label(current): current}
    winner = label(current)
    for pick in picks[1:]:
        added = changes(pick["settings"], phase.base_settings)
        trial = runner.measure("combine", label(pick), phase, current["settings"] | added)
        trials[label(pick)] = trial
        identical = all(name in IDENTICAL for name in added)
        if speed(trial) > speed(current) or identical and speed(trial) >= speed(current) > 0:
            current, winner = trial, label(pick)
    return {"name": "combination, highest isolated gain first", "trials": trials, "winner": winner,
            "seconds": time.monotonic() - began}


def calibrate_decode(runner, phase: Decode) -> tuple[list[dict], dict, dict[str, str]]:
    """The steps' measurements, the defaults and the chosen settings (name -> value)."""
    steps = []
    began = time.monotonic()
    base = runner.measure("pool-workers", DEFAULT, phase, {})
    base_seconds = time.monotonic() - began
    found = defaults(base["log"])
    workers = found["--pool-workers"][0]
    if workers:
        step = climb_step(runner, phase, base, "--pool-workers", int(workers), 1, MIN_WORKERS, math.inf,
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
        step = climb_step(runner, phase, base, "--pcie-frac", share, PCIE_STEP, 0.0, 1.0,
                          lambda value: f"{value:.2f}")
        step["name"] = f"--pcie-frac in {PCIE_STEP} steps from {share:.2f}, fixed (default: automatic)"
        step["note"] = f"share in the first start: {line}"
        steps.append(step)

    step = climb_step(runner, phase, base, "--spec-min-p", SPEC_START, SPEC_STEP, SPEC_STEP, SPEC_MAX,
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

    step = climb_step(runner, phase, base, "--adapt-every", every, 1, 1, math.inf, lambda v: str(round(v)),
                      lambda v: tier(v, rate, memory))
    step["name"] = f"tier: --adapt-every from {round(every)}, {rate:g} swaps and {memory:.3f} memory per window"
    steps.append(step)
    current = step["trials"][step["winner"]]
    every = float(current["settings"].get("--adapt-every", every))
    step = climb_step(runner, phase, current, "--adapt-swaps", rate, TIER_RATE_STEP, TIER_RATE_STEP, math.inf,
                      lambda v: str(round(v * every)), lambda v: tier(every, v, memory))
    step["name"] = f"tier: swaps per window in {TIER_RATE_STEP} steps at --adapt-every {round(every)}"
    steps.append(step)
    current = step["trials"][step["winner"]]
    rate = float(current["settings"].get("--adapt-swaps", rate * every)) / every
    step = climb_step(runner, phase, current, "--adapt-decay", memory, TIER_MEMORY_STEP, TIER_MEMORY_MIN,
                      TIER_MEMORY_MAX, lambda v: f"{v ** every:.3f}", lambda v: tier(every, rate, v))
    step["name"] = f"tier: memory per window in {TIER_MEMORY_STEP} steps at --adapt-every {round(every)}"
    steps.append(step)
    current = step["trials"][step["winner"]]

    step = discrete_step(runner, phase, current, ADAPT_FETCH, {mode: mode for mode in FETCH_MODES})
    step["name"] = f"{ADAPT_FETCH} unset / {' / '.join(FETCH_MODES)} at the tier's result"
    steps.append(step)
    if step["winner"] != DEFAULT:
        current = step["trials"][step["winner"]]
        step = climb_step(runner, phase, current, FETCH_MIN, FETCH_MIN_DEFAULT, FETCH_MIN_STEP, FETCH_MIN_LOW,
                          math.inf, lambda v: f"{v:g}")
        step["name"] = f"{FETCH_MIN} in {FETCH_MIN_STEP} steps from {FETCH_MIN_DEFAULT:g} at that mode"
        steps.append(step)
        current = step["trials"][step["winner"]]
        step = climb_step(runner, phase, current, FETCH_MARGIN, FETCH_MARGIN_DEFAULT, FETCH_MARGIN_STEP,
                          FETCH_MARGIN_LOW, math.inf, lambda v: f"{v:g}")
        step["name"] = f"{FETCH_MARGIN} in {FETCH_MARGIN_STEP} steps from {FETCH_MARGIN_DEFAULT:g} at that MIN"
        steps.append(step)
        current = step["trials"][step["winner"]]
    if current is not base:
        picks.append(current)

    combination = combine(runner, phase, base, picks)
    if combination is None:
        print("  combination and confirmation skipped: no step found a value faster than the default")
        return steps, found, {}
    steps.append(combination)
    confirmation = confirm(runner, phase, combination)
    steps.extend(confirmation)
    final = confirmation[-1]
    return steps, found, final["trials"][final["winner"]]["settings"]


def calibrate_prefill(runner, phase: Prefill) -> tuple[list[dict], dict, dict[str, str]]:
    """The steps' measurements, the defaults and the chosen settings, the phase's base included."""
    steps = []
    began = time.monotonic()
    base = runner.measure("base", phase.base_label, phase, phase.base_settings)
    base_seconds = time.monotonic() - began
    found = prefill_defaults(base["log"])
    step = discrete_step(runner, phase, base, TWO_PASS, {"1": "1"}, identical=True)
    step["seconds"] += base_seconds
    step["name"] = f"{TWO_PASS} unset / 1 (same output: kept when not slower)"
    steps.append(step)
    step = discrete_step(runner, phase, base, W4A8, {"1": "1", "2": "2"}, identical=True)
    step["name"] = f"{W4A8} unset / 1 / 2 (same output: kept when not slower)"
    steps.append(step)
    step = discrete_step(runner, phase, base, CPU_SHARE, {"0": "0"})
    step["name"] = f"{CPU_SHARE} unset (on) / 0"
    steps.append(step)
    limit = found[CPU_SHARE_MAX][0]
    if step["winner"] != DEFAULT:
        print(f"  {CPU_SHARE_MAX} skipped: the share is faster off")
    elif limit is None:
        print(f"  {CPU_SHARE_MAX} skipped: the base's log does not show the share on")
    else:
        step = climb_step(runner, phase, base, CPU_SHARE_MAX, int(limit), SHARE_MAX_STEP, SHARE_MAX_LOW, math.inf,
                          lambda v: str(round(v)))
        step["name"] = f"{CPU_SHARE_MAX} in {SHARE_MAX_STEP} steps from {limit}, the share on"
        steps.append(step)
    step = discrete_step(runner, phase, base, NV_CHUNK, {k: f"{k} ({v})" for k, v in NV_CHUNKS.items()})
    step["name"] = f"{NV_CHUNK} unset (32768) / " + " / ".join(f"{k} ({v})" for k, v in NV_CHUNKS.items())
    steps.append(step)
    picks = [step["trials"][step["winner"]] for step in steps if step["winner"] != DEFAULT]

    combination = combine(runner, phase, base, picks)
    if combination is None:
        print("  combination and confirmation skipped: no step found a value faster than the base")
        return steps, found, phase.base_settings
    steps.append(combination)
    confirmation = confirm(runner, phase, combination)
    steps.extend(confirmation)
    final = confirmation[-1]
    return steps, found, final["trials"][final["winner"]]["settings"]


def confirm(runner, phase, combination: dict) -> list[dict]:
    """Compare the fixed combination with each pick omitted, reusing its measurement. Drop failures together,
    with no further rounds; then measure the base and the confirmed set afresh, back to back."""
    full = combination["trials"][combination["winner"]]
    confirmed = dict(full["settings"])
    chosen = changes(confirmed, phase.base_settings)
    groups = (ADAPT_FLAGS, DECODE_VARIABLES, (CPU_SHARE, CPU_SHARE_MAX))
    picks = [{k: v for k, v in chosen.items() if k in group} for group in groups]
    grouped = {name for group in groups for name in group}
    picks += [{k: v} for k, v in chosen.items() if k not in grouped]
    steps = []

    def passes(trial: dict, without: dict, pick: dict[str, str]) -> bool:
        if speed(without) <= 0:   # the comparison start failed: no evidence for the pick
            return False
        if all(name in IDENTICAL for name in pick):
            return speed(trial) >= speed(without)
        return beats(trial, without)

    for pick in filter(None, picks):
        began = time.monotonic()
        settings = phase.base_settings | {k: v for k, v in full["settings"].items() if k not in pick}
        if settings == phase.base_settings:   # a single pick: the final comparison below asks the same question
            continue
        without = runner.measure("leave-one-out", "without " + shown(pick), phase, settings)
        kept = passes(full, without, pick)
        if not kept:
            confirmed = phase.base_settings | {k: v for k, v in confirmed.items() if k not in pick}
        rule = "not slower" if all(name in IDENTICAL for name in pick) else f"> {MIN_GAIN:.0%} faster"
        steps.append({"name": f"leave-one-out: {shown(pick)} ({rule})", "base_label": "without pick",
                      "trials": {DEFAULT: without, CHOSEN: full}, "winner": CHOSEN if kept else DEFAULT,
                      "verdict": "pick kept" if kept else "pick dropped", "seconds": time.monotonic() - began})

    began = time.monotonic()
    if confirmed == phase.base_settings:   # every pick dropped: the base is the result, nothing left to compare
        steps.append({"name": f"confirmation: every pick dropped, the result is {phase.base_label}",
                      "trials": {DEFAULT: combination["trials"][DEFAULT]}, "winner": DEFAULT, "verdict": "kept",
                      "seconds": 0.0})
        return steps
    trials = {DEFAULT: runner.measure("confirm", phase.base_label, phase, phase.base_settings),
              CHOSEN: runner.measure("confirm", CHOSEN, phase, confirmed)}
    chosen = changes(confirmed, phase.base_settings)
    kept = passes(trials[CHOSEN], trials[DEFAULT], chosen)
    steps.append({"name": f"confirmation of {shown(chosen) or phase.base_label} against {phase.base_label}",
                  "trials": trials, "winner": CHOSEN if kept else DEFAULT, "verdict": "kept",
                  "seconds": time.monotonic() - began})
    return steps


def report(phase, steps: list[dict], found: dict):
    print(f"\n{phase.name} phase")
    first = min(n for trial in steps[0]["trials"].values() for n in trial["starts"])
    for step in steps:
        trials = step["trials"]
        base_label = step.get("base_label", phase.base_label)
        print(f"\n{step['name']} ({step['seconds']:.0f} s)")
        if "note" in step:
            print(f"  {step['note']}")
        for label, trial in trials.items():
            where = ", ".join(str(n) for n in trial["starts"])
            reused = " reused" if trial["starts"][0] < first else ""
            shown_label = base_label if label == DEFAULT else label
            print(f"  {shown_label:>10}: {phase.row(trial)}  (start {where}{reused})")
            for line in trial["moved"]:
                print(f"{'':14}PcieAuto moved the share during the measured rounds: {line}")
        base = speed(trials[DEFAULT])
        winner = trials[step["winner"]]
        gain = (f", {speed(winner) / base - 1.0:+.1%} speed over the {base_label}"
                if base > 0 and step["winner"] != DEFAULT else "")
        shown_winner = base_label if step["winner"] == DEFAULT else step["winner"]
        print(f"  {step.get('verdict', 'best')}: {shown_winner} ({phase.value(winner)}{gain})")
        first = max(n for trial in trials.values() for n in trial["starts"]) + 1
    print("\ndefaults:")
    for name, (value, source) in found.items():
        print(f"  {name} {value if value is not None else '?'} ({source})")


def result_line(settings: dict[str, str]) -> str:
    return shlex.join([f"{k}={v}" for k, v in variables_of(settings).items()] + ["--"]
                      + list(sum(flags_of(settings).items(), ())))


def print_workload(workload: dict):
    cold = ", ".join(f"~{estimated_tokens(p)}" for p in workload["cold"])
    growing = [estimated_tokens(p) for p in workload["growing"]]
    print(f"prefill workload ({CHARS_PER_TOKEN} characters per token estimated): warm-up "
          f"~{estimated_tokens(workload['warm_up'])} tokens; cold {cold}; growing ~{growing[0]} -> ~{sum(growing)} in "
          f"{len(growing) - 1} turns of ~{min(growing[1:])}-{max(growing[1:])}; text: {workload['files']} files of "
          f"{', '.join(SOURCE_DIRS)}, {workload['used']} of {workload['chars']} characters used")


def main():
    ap = argparse.ArgumentParser(usage="uv run tools/calibrate.py [--phase decode|prefill|both] [--dry-run] "
                                       "-- RENZA-COMMAND ...",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("decode", "prefill", "both"), default="both",
                    help="what to tune (default: both, decode then prefill on its result)")
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
    decode = options.phase in ("decode", "both")
    prefill = options.phase in ("prefill", "both")

    tuned_flags = TUNED if decode else ()
    tuned_variables = (DECODE_VARIABLES if decode else ()) + (PREFILL_VARIABLES if prefill else ())
    removed = [(f, v) for f in tuned_flags if (v := flag_value(engine, f)) is not None]
    if removed:
        print("removed from the command (measured from the engine's defaults): " + shlex.join(sum(removed, ())))
    engine = without_flags(engine, tuned_flags)
    env = {k: v for k, v in os.environ.items() if k not in tuned_variables}
    unset = [f"{k}={v}" for k, v in os.environ.items() if k in tuned_variables]
    if unset:
        print("removed from the environment (measured from the engine's defaults): " + shlex.join(unset))
    inherited = [f"{k}={v}" for k, v in env.items() if k in DECODE_VARIABLES + PREFILL_VARIABLES]
    if inherited:
        print("kept from the environment (not tuned in this run): " + shlex.join(inherited))
    workload = None
    if prefill:
        removed = [(f, v) for f in KV_DISK_FLAGS if (v := flag_value(server, f)) is not None]
        if removed:
            print("removed from the command in the prefill phase (no KV from an earlier start): "
                  + shlex.join(sum(removed, ())))
        workload = prefill_workload(Path(__file__).resolve().parent.parent)
        print_workload(workload)

    if options.dry_run:
        print(f"dry run, assumed: every candidate is faster than the best so far (the most starts) except "
              f"{shown(DRY_RUN_SLOWER)}; leave-one-out starts are slower (all picks kept), an engine count of "
              f"{DRY_RUN_WORKERS} pool workers, the share settled at "
              f"0.55 (a native pack's start-up default)")
        runner = DryRun()
        decoded = {}
        if decode:
            decoded = calibrate_decode(runner, Decode(server, engine))[2]
        if prefill:
            calibrate_prefill(runner, Prefill(without_flags(server, KV_DISK_FLAGS), engine, decoded, workload))
        for name, count in runner.counts.items():
            print(f"{name}: {count} starts, about {runner.seconds[name] / 60:.0f} min")
        print(f"{runner.starts} starts under these assumptions, about {sum(runner.seconds.values()) / 60:.0f} min "
              f"(a start {START_SECONDS:.0f} s; decode: {WARM_UP_SECONDS} s per warm-up, {ANSWER_SECONDS} s per "
              f"answer; prefill: {COLD_TOKENS_PER_SECOND:.0f} tok/s cold, {TURN_TOKENS_PER_SECOND:.0f} tok/s in "
              f"later turns, {SHORT_ANSWER_SECONDS} s per answer)")
        return
    with socket.socket() as probe:
        if probe.connect_ex((host, int(port))) == 0:
            sys.exit(f"port {port} is in use: stop the server on it first")
    runner = Server(f"http://{host}:{port}", flag_value(server, "--api-key"), env)
    began = time.monotonic()
    results = []
    settings: dict[str, str] = {}
    if decode:
        phase = Decode(server, engine)
        steps, found, settings = calibrate_decode(runner, phase)
        results.append((phase, steps, found))
    if prefill:
        phase = Prefill(without_flags(server, KV_DISK_FLAGS), engine, settings, workload)
        steps, found, settings = calibrate_prefill(runner, phase)
        results.append((phase, steps, found))
    for phase, steps, found in results:
        report(phase, steps, found)
    print(f"\n{runner.starts} starts, {(time.monotonic() - began) / 60:.1f} min in total")
    print(result_line(settings))


if __name__ == "__main__":
    main()
