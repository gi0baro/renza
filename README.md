# (infe)renza

My local inference harness: basically [Strata](https://github.com/Niko1221/Strata)'s engine,
inside [ds4](https://github.com/antirez/ds4)'s server.
One binary, no Python at run time, usual APIs.

Built and tuned for one Linux machine only: RTX 3090Ti, 128GB RAM, Zen3 CPU.
It is public so anyone with the same hardware can fork it and experiment; issues and pull requests are not accepted.

Built on ds4 and Strata, with llama.cpp's ggml. MIT, see [LICENSE](LICENSE).

## Bootstrap

```sh
third_party/fetch-llama.sh
cmake -S . -B build -G Ninja && cmake --build build --target renza

tools/download.sh --models /data/ML/models-renza --data data/qwen
uv run tools/iq_pack.py --gguf /data/ML/models-renza/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --out data/qwen/pack
uv run tools/mtp_fetch.py fetch --out data/qwen/mtp
uv run tools/mtp_fetch.py verify --out data/qwen/mtp
uv run tools/mtp_rt.py --src data/qwen/mtp --draft-vocab data/qwen/draft_vocab.bin --out data/qwen/mtp/rt
```

## Run

```sh
RENZA_API_KEY=... ./run-renza.sh qwen
```

After an engine change, re-tune with `uv run tools/calibrate.py -- <the renza command line from run-renza.sh>`
and paste the flags it prints into the preset.
