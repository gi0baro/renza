# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""GGUF v3 header reader (metadata and tensor table) for tools/iq_pack.py. Tensor data is not read."""
from __future__ import annotations

import dataclasses
import pathlib
import struct
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# The ggml types of Swift 1.5 IQ3_S: id -> (name, block elements, bytes per block). 42 = Q2_0, the GSQ-RCO
# routed-expert encoding, is not a ggml upstream type. A type not listed here has no byte size, which the pack
# tool refuses.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 18: ("IQ3_XXS", 256, 98), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18),
}

# metadata value type id -> (name, struct format, size); the format is the whole GGUF v3 set, needed to walk the
# header
GGUF_META: dict[int, tuple[str, str | None, int | None]] = {
    0: ("u8", "B", 1), 1: ("i8", "b", 1), 2: ("u16", "H", 2), 3: ("i16", "h", 2), 4: ("u32", "I", 4),
    5: ("i32", "i", 4), 6: ("f32", "f", 4), 7: ("bool", "?", 1), 8: ("string", None, None),
    9: ("array", None, None), 10: ("u64", "Q", 8), 11: ("i64", "q", 8), 12: ("f64", "d", 8),
}


@dataclasses.dataclass
class TensorInfo:
    name: str
    shape: list[int]
    type_id: int
    type_name: str
    offset: int

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def expected_bytes(self) -> int | None:
        if self.type_id not in GGML_TYPES:
            return None
        _, block_elems, block_bytes = GGML_TYPES[self.type_id]
        if self.elements % block_elems:
            return None
        return self.elements // block_elems * block_bytes


class GGUFFile:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: list[TensorInfo] = []
        self.alignment = 32
        with self.path.open("rb") as fh:
            self._parse(fh)

    def _parse(self, fh) -> None:
        magic = struct.unpack("<I", fh.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path.name}: not a GGUF file (magic {magic:#x})")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        if version != 3:
            raise ValueError(f"{self.path.name}: GGUF v{version}, this reader handles v3")
        for _ in range(n_kv):
            key = self._str(fh)
            self.metadata[key] = self._value(fh)
        for _ in range(n_tensors):
            name = self._str(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            shape = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            type_id, offset = struct.unpack("<IQ", fh.read(12))
            type_name = GGML_TYPES[type_id][0] if type_id in GGML_TYPES else f"type{type_id}"
            self.tensors.append(TensorInfo(name, shape, type_id, type_name, offset))
        align = self.metadata.get("general.alignment")
        if isinstance(align, int) and align:
            self.alignment = align
        pos = fh.tell()
        self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _str(self, fh) -> str:
        (n,) = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8", "replace")

    def _value(self, fh):
        (t,) = struct.unpack("<I", fh.read(4))
        name, fmt, size = GGUF_META[t]
        if name == "string":
            return self._str(fh)
        if name == "array":
            (et,) = struct.unpack("<I", fh.read(4))
            (count,) = struct.unpack("<Q", fh.read(8))
            ename, efmt, esize = GGUF_META[et]
            if ename == "string":
                return [self._str(fh) for _ in range(count)]
            return list(struct.unpack(f"<{count}{efmt}", fh.read(esize * count)))
        return struct.unpack(f"<{fmt}", fh.read(size))[0]
