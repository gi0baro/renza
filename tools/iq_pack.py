# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.3"]
# ///
"""Build renza's pack for Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_S from the model's GGUF shards.

    uv run tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out <pack>

The engine reads the experts and every quantized dense tensor from the GGUF itself (--native); the pack holds:

  index.txt            the tensor table the engine loads; quantized tensors carry their shape only.
  dense.bin            the float tensors in the form the engine reads them (index kinds 4/5/2 = BF16/F16/F32).
  conversions.json     every tensor stored in another form than the engine reads: the routers, F32 in the GGUF
                       with values that are exactly BF16, written as BF16 (schema 1).
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes gate_off up_off down_off, then
                       the shard holding the layer when it is not shard 1, per role (`gate,up,down`, empty = shard
                       1) when the layer straddles the two shards (layer 14). Written last: a pack without it is
                       unfinished.

The per-layer-embedding table (per_layer_token_embd, 28.8 GB) is not in the pack: the engine reads it from the
GGUF. Input that is not this model (another tensor form, a missing shard, unknown types) is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
NOT_IN_PACK = {"per_layer_token_embd.weight"}
KIND = {"BF16": "4", "F16": "5", "F32": "2"}
ROLES = ("gate", "up", "down")
ALIGN = 64
SHARD_ONE = re.compile(r"-00001-of-(\d{5})\.gguf$")
ROUTER = "blk.%d.ffn_gate_inp.weight"
EXACT_BF16 = "f32->bf16: the values are BF16's (low 16 bits zero)"

# The float form the engine reads each pack tensor in (name without `blk.N.`): the residual, router, GDN, QSA and
# PLE kernels read these projections as BF16, the norms as F32 and the PLE conv as F16. Tensors not named here are
# written as stored (floats) or served from the GGUF (quantized).
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}


def form_of(name: str) -> str | None:
    return FORM.get(re.sub(r"^blk\.\d+\.", "", name))


class Model:
    """Both shards of the model: tensor name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path):
        m = SHARD_ONE.search(first.name)
        if m is None:
            raise SystemExit(f"{first.name}: expected shard 1 of the split model (<name>-00001-of-0000N.gguf)")
        total = int(m.group(1))
        self.paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                      for i in range(1, total + 1)]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise SystemExit("missing model shards: " + ", ".join(missing))
        self.files = [G.GGUFFile(p) for p in self.paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        check_split(self.files)
        self.where = {}
        for p, g in zip(self.paths, self.files):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise SystemExit(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise SystemExit(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)


def check_split(files) -> None:
    """The split keys, as the engine checks them (strata::GgufModel): shard 1 carries the metadata, and every
    shard declares split.count / split.no / split.tensors.count consistently."""
    n = len(files)
    if "general.architecture" not in files[0].metadata:
        raise SystemExit(f"{files[0].path.name} has no general.architecture; the first shard of a split model "
                         "carries the metadata")
    total = files[0].metadata.get("split.tensors.count")
    for i, g in enumerate(files):
        md = g.metadata
        if md.get("split.count") != n or md.get("split.no") != i or md.get("split.tensors.count") != total:
            raise SystemExit(f"{g.path.name} does not declare itself shard {i + 1} of {n} of this model "
                             "(split.count / split.no / split.tensors.count)")
    if sum(len(g.tensors) for g in files) != total:
        raise SystemExit(f"the {n} shards hold {sum(len(g.tensors) for g in files)} tensors, but "
                         f"split.tensors.count is {total}")


def tensor_bytes(mm, g, t) -> np.ndarray:
    start = g.data_start + t.offset
    return mm[start: start + t.expected_bytes()]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith("_exps.weight")


def convert(name: str, type_name: str, raw: np.ndarray):
    """The pack form of one float tensor: (kind, bytes, record for conversions.json or None)."""
    form = form_of(name)
    if form is None or form == type_name:
        return KIND[type_name], raw.tobytes(), None
    if (type_name, form) != ("F32", "BF16"):
        raise SystemExit(f"{name} is {type_name}, but the engine reads {form}; this tool converts F32 -> BF16 only")
    values = raw.view(np.float32)
    if not np.isfinite(values).all():
        raise SystemExit(f"{name} is F32 with non-finite values; refused rather than converted")
    u = values.view(np.uint32)
    if np.count_nonzero(u & 0xFFFF):
        raise SystemExit(f"{name} is F32 with values that are not BF16; the engine reads BF16 and this tool does "
                         "not round weights")
    rec = {"src_type": type_name, "dst_type": form, "method": EXACT_BF16, "exact": True, "max_abs_err": 0.0}
    return KIND[form], (u >> 16).astype("<u2").tobytes(), rec


def index_standalone(src: pathlib.Path, out: pathlib.Path, model: Model) -> None:
    """Every non-expert tensor: the floats into dense.bin in the form the engine reads them, the quantized ones
    served natively from the GGUF."""
    rows, records = [], []
    at = served = 0
    with open(out / "dense.bin", "wb") as fo:
        for name, (g, t, mm, p) in model.where.items():
            if is_expert(name) or name in NOT_IN_PACK:
                continue
            if len(t.shape) > 2:
                raise SystemExit("tensor %s has %d dimensions; the index holds two" % (name, len(t.shape)))
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            if t.type_name not in FLOAT:
                if form_of(name) is not None:
                    raise SystemExit(f"{name} is {t.type_name}, but the engine reads {form_of(name)} from the pack")
                served += 1
                rows.append([name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
                continue
            raw = tensor_bytes(mm, g, t)
            kind, data, rec = convert(name, t.type_name, raw)
            if rec is not None:
                records.append({"name": name, "shard": p.name, **rec,
                                "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
            rows.append([name, "0", kind, str(at), str(len(data)), "0", str(len(data)), str(ne0), str(ne1),
                         "0", "0", "1"] + ["0"] * 7)
            fo.write(data)
            pad = (-len(data)) % ALIGN
            fo.write(b"\0" * pad)
            at += len(data) + pad
    write_index(out, rows, src, served)
    (out / "conversions.json").write_text(json.dumps({
        "schema": 1, "tool": "tools/iq_pack.py", "compat_bf16": False,
        "source_shards": [{"name": q.name, "size": s} for q, s in zip(model.paths, model.sizes)],
        "tensors": records,
    }, indent=1) + "\n", encoding="utf-8")
    print("conversions.json: %d tensors converted to the engine's form (all exact)" % len(records))


def write_index(out: pathlib.Path, rows, src: pathlib.Path, served: int) -> None:
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt: %d tensors, %d served natively, arena %.2f GiB" % (len(rows), served, at / 2**30))


def expert_layout(model: Model, src: pathlib.Path) -> str:
    """native_experts.txt. Each role's offset is absolute in the shard that holds it: the shards' data sections do
    not start at the same byte."""
    T = {n: w[1] for n, w in model.where.items()}
    exps = [n for n in T if is_expert(n)]
    if not exps:
        raise SystemExit("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)")
    n_layers = 1 + max(int(n.split(".")[1]) for n in exps)
    n_expert = int(T[ROUTER % 0].shape[1])
    if any(int(T[ROUTER % l].shape[1]) != n_expert for l in range(n_layers)):
        raise SystemExit("the routers disagree on the expert count")
    lines, offset, split = [], 0, []
    for l in range(n_layers):
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
        if any(n not in T for n in names):
            raise SystemExit("layer %d: missing %s" % (l, ", ".join(n for n in names if n not in T)))
        ts = [T[n] for n in names]
        if any(len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            raise SystemExit("layer %d: an expert tensor is not [*, *, %d]" % (l, n_expert))
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            raise SystemExit("layer %d: gate and up differ in type" % l)
        blob = sum(per)
        ws = [model.where[n] for n in names]
        files = ["" if w[3] == src else w[3].name for w in ws]
        column = files[0] if len(set(files)) == 1 else ",".join(files)
        if len(set(files)) != 1:
            split.append(l)
        line = "%d %d %d %d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob,
                                            *[w[0].data_start + w[1].offset for w in ws])
        lines.append(line + ("" if not column else " " + column))
        offset += blob * n_expert
    head = ("# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
            "[shard | gate,up,down] (n_expert %d, total %d; absolute offsets in %s, or in the named shard "
            "beside it - per role where the column is gate,up,down)\n" % (n_expert, offset, src.name))
    print("native_experts.txt: %d layers, %d expert(s) each, per-role shard column for layer(s) %s"
          % (n_layers, n_expert, ", ".join(map(str, split)) or "none"))
    return head + "".join(line + "\n" for line in lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # .absolute(), not .resolve(): a symlinked shard keeps its -0000N-of-0000M name, by which the others are found
    src = pathlib.Path(a.gguf).absolute()
    out = pathlib.Path(a.out)

    model = Model(src)
    print("model shards: " + ", ".join(p.name for p in model.paths))
    experts = expert_layout(model, src)
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_experts.txt").unlink(missing_ok=True)
    index_standalone(src, out, model)
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write(experts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
