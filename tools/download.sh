#!/usr/bin/env bash
# Fetches the pinned files renza runs on: the two GGUF shards and the mmproj into --models, Strata's expert profile
# and draft vocabulary into --data. A file that matches its pin (size and SHA-256) is skipped; a mismatch stops
# the script with an error and nothing is deleted.
set -euo pipefail

usage() {
    printf 'Usage: %s --models DIR --data DIR\n' "$0" >&2
    exit 2
}

models=
data=
while (( $# )); do
    case $1 in
        --models)
            (( $# >= 2 )) && [[ -n $2 ]] || usage
            models=$2
            shift 2
            ;;
        --data)
            (( $# >= 2 )) && [[ -n $2 ]] || usage
            data=$2
            shift 2
            ;;
        *) usage ;;
    esac
done
[[ -n $models && -n $data ]] || usage

check_pin() {
    local file=$1 size=$2 hash=$3 actual_size actual_hash
    actual_size=$(stat -c %s -- "$file")
    actual_hash=$(sha256sum < "$file")
    actual_hash=${actual_hash%% *}
    if [[ $actual_size != "$size" || $actual_hash != "$hash" ]]; then
        printf 'mismatch: %s (expected %s bytes, SHA-256 %s; got %s bytes, SHA-256 %s)%s\n' \
            "$file" "$size" "$hash" "$actual_size" "$actual_hash" \
            "$([[ $file == *.part ]] && printf '; delete it to restart')" >&2
        return 1
    fi
}

fetch() {
    local file=$1 url=$2 size=$3 hash=$4
    if [[ -e $file ]]; then
        check_pin "$file" "$size" "$hash"
        printf 'skipped: %s\n' "$file"
        return
    fi
    mkdir -p -- "$(dirname -- "$file")"
    if ! curl --fail -L -C - --output "$file.part" "$url"; then
        printf 'download failed: %s\n' "$file" >&2
        return 1
    fi
    check_pin "$file.part" "$size" "$hash"
    mv -- "$file.part" "$file"
    printf 'downloaded: %s\n' "$file"
}

# The repo keeps every file at its root; locally the shards go in swift-IQ3_S/.
models_url=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d
data_url=https://raw.githubusercontent.com/Niko1221/Strata/v0.1.45/data
shard=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S
mmproj=mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf

fetch "$models/swift-IQ3_S/$shard-00001-of-00002.gguf" \
    "$models_url/$shard-00001-of-00002.gguf" 44900816736 \
    46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b
fetch "$models/swift-IQ3_S/$shard-00002-of-00002.gguf" \
    "$models_url/$shard-00002-of-00002.gguf" 38842920384 \
    70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608
fetch "$models/$mmproj" "$models_url/$mmproj" 907543520 \
    cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60
fetch "$data/expert-profile.bin" "$data_url/expert-profile.bin" 196632 \
    8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf
fetch "$data/draft_vocab.bin" "$data_url/draft_vocab.bin" 425196 \
    b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91
