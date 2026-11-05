#!/usr/bin/env bash
set -euo pipefail

third_party=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root="$third_party/llama.cpp"
reference_llama=/home/giovanni/ML/Strata/third_party/llama.cpp
llama_commit=3cf03257f219afbe7334045ff7c6a06ac68c627d

# A partial or unverified checkout must not be left buildable.
trap 'rm -rf -- "$root"' ERR
rm -rf -- "$root"
if [[ -d "$reference_llama" ]]; then
    # setup.py's source stamp seeds with the commit but does not hash llama.cpp.
    cp -a -- "$reference_llama" "$root"
    diff -rq -- "$reference_llama" "$root"
else
    python3 - "$third_party" "$llama_commit" <<'PY'
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

third_party = Path(sys.argv[1])
commit = sys.argv[2]
with tempfile.TemporaryDirectory(dir=third_party) as directory:
    temporary = Path(directory)
    archive = temporary / "llama.cpp.zip"
    urllib.request.urlretrieve(
        f"https://github.com/ggml-org/llama.cpp/archive/{commit}.zip", archive
    )
    with zipfile.ZipFile(archive) as source:
        source.extractall(
            temporary, [name for name in source.namelist() if "/tools/ui/" not in name]
        )
    extracted = [path for path in temporary.iterdir() if path.is_dir()]
    if len(extracted) != 1 or not (extracted[0] / "gguf-py").is_dir():
        sys.exit("unexpected llama.cpp archive layout")
    shutil.move(str(extracted[0]), third_party / "llama.cpp")
PY
fi
printf 'llama.cpp %s: %s\n' "$llama_commit" "$root"
