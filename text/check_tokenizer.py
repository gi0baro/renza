#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


CORPUS = [
    "", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
    "def f(x):\n\treturn x  # comment\n", "你好，世界", "مرحبا",
    "😀🚀🇺🇸", "e\u0301\u0301 combining", "\x00\x01\x7f control",
    "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes",
    "\r\n\r\n", "\u2028\u2029", "x" * 5000,
]

ROOT = Path(__file__).resolve().parents[1]
LLAMA_VOCAB = ROOT / "upstream/strata-0.1.45/third_party/llama.cpp/models/ggml-vocab-qwen35.gguf"
VECTOR_SEPARATOR = "\n__ggml_vocab_test__"
# llama.cpp's vocabulary lacks the chat template's markers, and renza requires <think>: they
# are appended as Strata's tokenizer test appends them.  That test encodes the vectors without
# parse_special and no vector holds a special token, so they are compared in renza's raw mode.
LLAMA_ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
               "</tool_response>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
VISION_START = "<|vision_start|>"

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_STRING = 8
GGUF_ARRAY = 9

UTF8 = "utf-8"
RAW = None
CHAT = "--chat"
DETOK = "--detok"


def run(binary, gguf, mode, data):
    command = [str(binary), str(gguf)] + ([mode] if mode else [])
    result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    return result.stdout


def parse_ids(text):
    text = text.strip()
    return [int(token) for token in text.replace(b",", b" ").split()] if text else []


def write_vocab_gguf(path, tokens, merges):
    """A metadata-only GGUF v3 holding the two tables renza's tokenizer reads."""
    def string(value):
        data = value.encode(UTF8)
        return struct.pack("<Q", len(data)) + data

    def strings(key, values):
        return (string(key) + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(values))
                + b"".join(string(value) for value in values))

    path.write_bytes(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 2)
                     + strings("tokenizer.ggml.tokens", tokens)
                     + strings("tokenizer.ggml.merges", merges))


def read_vectors(vocab):
    inputs = vocab.with_suffix(".gguf.inp").read_text(encoding=UTF8).split(VECTOR_SEPARATOR + "\n")
    outputs = vocab.with_suffix(".gguf.out").read_text(encoding=UTF8).split("\n")
    if len(inputs) != len(outputs):
        return None
    cases = []
    for text, want in zip(inputs, outputs):
        if text.endswith(VECTOR_SEPARATOR):
            text = text[:-len(VECTOR_SEPARATOR)]
        cases.append((text, [int(token) for token in want.split()]))
    return cases


class Check:
    def __init__(self, binary):
        self.binary = binary
        self.checked = {}
        self.skipped = []
        self.bad = 0

    def skip(self, name, reason):
        self.skipped.append(name)
        print(f"SKIP {name}: {reason}")

    def fail(self, message):
        self.bad += 1
        print(f"FAIL {message}")

    def result(self, kind, name, failures, detail):
        self.checked[kind] = self.checked.get(kind, 0) + 1
        if failures:
            self.bad += 1
            print(f"FAIL {name}: {'; '.join(failures)}\n{detail()}")
        return not failures

    def against_oracle(self, kind, gguf, tokenizer, name, value, ids_file=None):
        raw = value.encode(UTF8)
        output = run(self.binary, gguf, CHAT, raw)
        actual = parse_ids(output)
        expected = tokenizer.encode(value, parse_special=True)
        failures = []
        if actual != expected:
            failures.append("ids differ from the Python tokenizer")
        if ids_file is not None and actual != parse_ids(ids_file.read_bytes()):
            failures.append(f"ids differ from {ids_file.name}")
        back = run(self.binary, gguf, DETOK, output)
        if back != raw:
            failures.append(f"C round trip differs: {back!r}")
        oracle_back = tokenizer.decode(expected).encode(UTF8)
        if oracle_back != raw:
            failures.append(f"oracle round trip differs: {oracle_back!r}")
        if self.result(kind, name, failures, lambda: f"input={value!r}\nC={actual}\nPython={expected}"):
            print(f"PASS {name}: {len(actual)} tokens" + (f", = {ids_file.name}" if ids_file else ""))

    def against_vectors(self, gguf, name, value, expected):
        raw = value.encode(UTF8)
        output = run(self.binary, gguf, RAW, raw)
        actual = parse_ids(output)
        failures = [] if actual == expected else ["ids differ from llama.cpp's vector"]
        back = run(self.binary, gguf, DETOK, output)
        if back != raw:
            failures.append(f"C round trip differs: {back!r}")
        if self.result("llama.cpp vectors", name, failures,
                       lambda: f"input={value!r}\nC={actual}\nllama.cpp={expected}"):
            print(f"PASS {name}: {len(actual)} tokens")


def check_baseline(check, gguf, tokenizer, inputs):
    if not inputs.is_dir():
        check.skip("baseline", f"{inputs} missing")
        return
    prompts = sorted(inputs.glob("*.prompt.txt"))
    if not prompts:
        check.fail(f"baseline: no *.prompt.txt in {inputs}")
        return
    for path in prompts:
        value = path.read_bytes().decode(UTF8)
        if VISION_START in value:
            check.skip(path.name, "vision tokens go in by id, not through the text tokenizer (unit 6)")
            continue
        ids_file = path.with_name(path.name.removesuffix(".prompt.txt") + ".ids")
        check.against_oracle("baseline", gguf, tokenizer, path.name, value,
                             ids_file if ids_file.is_file() else None)


def check_vectors(check, strata, vocab):
    missing = [str(path) for path in (vocab, vocab.with_suffix(".gguf.inp"),
                                      vocab.with_suffix(".gguf.out")) if not path.is_file()]
    if missing:
        check.skip("llama.cpp vectors", ", ".join(missing) + " missing")
        return
    cases = read_vectors(vocab)
    if cases is None:
        check.fail(f"llama.cpp vectors: inputs and outputs differ in number in {vocab}")
        return
    if not cases:
        check.fail(f"llama.cpp vectors: no cases in {vocab}")
        return
    sys.path.insert(0, str(strata))
    from tools.gguf_reader import GGUFFile
    metadata = GGUFFile(vocab).metadata
    tokens = list(metadata["tokenizer.ggml.tokens"])
    tokens += [token for token in LLAMA_ADDED if token not in tokens]
    with tempfile.TemporaryDirectory() as tmp:
        gguf = Path(tmp) / "vocab.gguf"
        write_vocab_gguf(gguf, tokens, list(metadata["tokenizer.ggml.merges"]))
        for i, (text, expected) in enumerate(cases):
            check.against_vectors(gguf, f"llama.cpp-{i}", text, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--strata", type=Path, default=ROOT / "upstream/strata-0.1.45")
    parser.add_argument("--inputs", type=Path, default=ROOT / "tests/baseline/results/inputs")
    parser.add_argument("--llama-vocab", type=Path, default=LLAMA_VOCAB)
    args = parser.parse_args()
    tokenizer_dir = Path(os.environ.get("STRATA_TOKENIZER", str(
        ROOT / "data/qwen/pack/tokenizer")))
    required = [args.gguf, args.strata / "tools/strata_tokenizer.py", tokenizer_dir / "vocab.json",
                tokenizer_dir / "merges.txt", tokenizer_dir / "token_type.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("SKIP: tokenizer oracle inputs missing: " + ", ".join(missing))
        return 77
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.strata))
    from tools.strata_tokenizer import Tokenizer

    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding=UTF8))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tokenizer = Tokenizer(tokens, (tokenizer_dir / "merges.txt").read_text(encoding=UTF8).split("\n"),
                          json.loads((tokenizer_dir / "token_type.json").read_text(encoding=UTF8)))
    check = Check(args.binary)
    for i, value in enumerate(CORPUS):
        check.against_oracle("corpus", args.gguf, tokenizer, f"corpus-{i}", value)

    golden_path = args.strata / "serve/chat_golden.json"
    if golden_path.is_file():
        goldens = json.loads(golden_path.read_text(encoding=UTF8))
        if not goldens:
            check.fail(f"goldens: no cases in {golden_path}")
        for item in goldens:
            check.against_oracle("goldens", args.gguf, tokenizer, "golden-" + item["name"], item["rendered"])
    else:
        check.skip("goldens", f"{golden_path} missing")

    check_baseline(check, args.gguf, tokenizer, args.inputs)
    check_vectors(check, args.strata, args.llama_vocab)

    kinds = ", ".join(f"{n} {kind}" for kind, n in check.checked.items())
    skipped = f" ({', '.join(check.skipped)})" if check.skipped else ""
    print(f"tokenizer: {sum(check.checked.values())} checked ({kinds}), "
          f"{len(check.skipped)} skipped{skipped}, {check.bad} failed")
    return int(check.bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
