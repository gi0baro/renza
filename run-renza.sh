#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s PRESET\nPresets: qwen\n' "$0" >&2
    exit 2
}

(( $# == 1 )) || usage
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RENZA_MODELS=${RENZA_MODELS:-/data/ML/models-renza}
RENZA_DATA=${RENZA_DATA:-$repo/data}

case $1 in
    qwen)
        native=$RENZA_MODELS/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
        args=(
            -m "$native"
            --pack "$RENZA_DATA/qwen/pack"
            --mtp-model "$RENZA_DATA/qwen/mtp/rt"
            --vision "$RENZA_MODELS/mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"
            --
            --ple-gguf "$native"
            --expert-profile "$RENZA_DATA/qwen/expert-profile.bin"
            --expert-cache auto --prefill auto --spec 4 --kv int8
            --kv-resident 32768 --vram-reserve-mib 700 --spec-min-p 0.8
            --adapt-every 2 --adapt-swaps 96 --adapt-decay 0.837 --pool-workers 6
        )
        ;;
    *) usage ;;
esac

: "${RENZA_API_KEY:?RENZA_API_KEY must be set}"
binary=$repo/build/renza
if [[ ! -x $binary ]]; then
    printf 'Missing executable: %s\nBuild it with: cmake --build "%s/build" --target renza\n' \
        "$binary" "$repo" >&2
    exit 1
fi

exec "$binary" -c 262144 --host 0.0.0.0 --port 8844 --api-key "$RENZA_API_KEY" "${args[@]}"
