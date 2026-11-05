# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.3"]
# ///
"""Build renza's MTP draft layer directly from the fetched BF16 tensors.

    uv run tools/mtp_rt.py --src DIR --draft-vocab draft_vocab.bin --out DIR/rt

Routed experts use Q2_0 with per-block MSE scale search; large dense projections
use Q8_0. The remaining matrices stay BF16, and F32 norm weights include Gemma's
1 + w. The output also includes the supplied draft vocabulary. Run
`tools/mtp_fetch.py verify` on --src first.

dense.txt has one line per dense tensor: name kind rows cols offset bytes, with kind q8_0, bf16 or f32 and the
offset into dense.bin.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

H, FF, NE = 2560, 640, 512
Q2_BLOCK, Q2_BYTES = 64, 18
Q8_BLOCK, Q8_BYTES = 32, 34
DENSE_ALIGNMENT = 256
BLOB = 3 * (H * FF * Q2_BYTES // Q2_BLOCK)
EXPERT_TENSORS = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def q2_0(w: np.ndarray) -> np.ndarray:
    """Q2_0 uses 64-value blocks: fp16 d and four 2-bit codes per byte, code = q + 1."""
    x = w.reshape(-1, Q2_BLOCK).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    # Candidate scales span amax/2 (the +2 level reaches the max) to amax (the ggml reference); 17 steps.
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)  # Quantize against the stored scale.
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], Q2_BYTES), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def q8_0(x: np.ndarray) -> bytes:
    """Q8_0 uses 32-value blocks with fp16 d = amax / 127 and rounding away from zero."""
    b = x.reshape(-1, Q8_BLOCK).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], Q8_BYTES), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """Interleave gate/up rows, then store gate/up and down codes before their scales."""
    gub = gu.reshape(2 * FF, H // Q2_BLOCK, Q2_BYTES)
    inter = np.empty_like(gub)
    inter[0::2] = gub[:FF]
    inter[1::2] = gub[FF:]
    dnb = dn.reshape(H, FF // Q2_BLOCK, Q2_BYTES)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, len(out)
    return out


def write_experts(src: Path, out: Path, tensors: dict) -> None:
    gu_meta, dn_meta = (tensors[name] for name in EXPERT_TENSORS)
    assert gu_meta["shape"] == [NE, 2 * FF, H], gu_meta["shape"]
    assert dn_meta["shape"] == [NE, H, FF], dn_meta["shape"]
    gu = np.memmap(src / gu_meta["file"], dtype="<u2", mode="r", shape=tuple(gu_meta["shape"]))
    dn = np.memmap(src / dn_meta["file"], dtype="<u2", mode="r", shape=tuple(dn_meta["shape"]))
    # The fused gate/up tensor would occupy 6.7 GB as F32; expand only one expert at a time.
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            f.write(blob_of(q2_0(bf16_to_f32(gu[e])), q2_0(bf16_to_f32(dn[e]))))


def write_dense(src: Path, out: Path, manifest: list[dict]) -> tuple[int, list[str]]:
    lines = []
    off = 0
    with open(out / "dense.bin", "wb") as f:
        # Dense tensor order is part of the runtime artifact format.
        for t in sorted(manifest, key=lambda t: t["name"]):
            name, shape = t["name"], t["shape"]
            if name in EXPERT_TENSORS:
                continue
            short = name[len("mtp."):]
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            data = np.fromfile(src / t["file"], dtype="<u2").reshape(shape)
            if len(shape) == 1:
                arr = bf16_to_f32(data) + 1.0
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:
                rows, cols = data.shape
                if short in Q8:
                    raw, kind = q8_0(bf16_to_f32(data)), "q8_0"
                else:
                    raw, kind = data.tobytes(), "bf16"
            pad = (-off) % DENSE_ALIGNMENT
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return off, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory written by tools/mtp_fetch.py fetch")
    ap.add_argument("--draft-vocab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    manifest = json.loads((src / "mtp-manifest.json").read_text(encoding="utf-8"))
    tensors = {t["name"]: t for t in manifest}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.draft_vocab, out / "draft_vocab.bin")
    write_experts(src, out, tensors)
    off, lines = write_dense(src, out, manifest)
    print(f"experts.bin {NE * BLOB} B, dense.bin {off} B, {len(lines)} tensors -> {out}")
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
