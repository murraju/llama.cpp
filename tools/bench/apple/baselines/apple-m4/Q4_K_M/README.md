# Apple M4 / Gemma 4 E4B / Q4_K_M — stock baseline

## Target

- **Chip:** Apple M4, 24 GB unified memory (dev laptop)
- **Model:** `/Users/rajum/PrivacyAI/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf`
- **Context cap:** 16k tokens per the durable laptop constraint. The matrix
  pins server `ctx_size=16384` — allocating more KV on 24 GB with Gemma 4's
  hybrid attention creates memory pressure. `longctx_16k` fills 16128 prompt +
  128 decode tokens, leaving a 128-token margin.

## Stock binary

Fresh build of this fork's `dev` branch at upstream `408225bb1`:

```bash
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server
```

Runtime environment pinned in `matrix.yaml`:

- `GGML_METAL_TENSOR_DISABLE=1`

## Baseline files

Populated by `run_matrix.py`. Each run produces a JSON at
`stock_<YYYY-MM-DD>_408225bb1_runN.json`, plus a `latest.json` symlink
pointing at the current reference run. A `noise-floor_408225bb1.txt` records
the run1-vs-run2 diff from `compare.py`.

## Captured baseline (2026-04-16, post-reboot)

Two back-to-back runs at `ctx_size=16384` on a freshly rebooted M4. Decode
tok/s (median across 3 iterations per profile):

| profile      | run1 (cool) | run2 (warm) | spec §3 ref |
| ------------ | ----------: | ----------: | ----------: |
| cold_short   |       29.43 |       24.07 |        22.23 |
| short_ctx    |       29.20 |       25.19 |        20.15 |
| rag_ctx      |       29.57 |       25.42 |        16.53 |
| longctx_8k   |       26.79 |       20.66 |        15.80 |
| longctx_16k  |       19.79 |       16.77 |        14.28 |

**Both runs beat the pre-reset `LLAMA_CPP_APPLE_SILICON_ENGINEERING_SPEC.md`
§3 reference on every profile.** The uplift over the pre-reset numbers comes
from (a) removing Sprint 1-7 hot-path telemetry atomics, (b) dropping
`ctx_size` from 32768 to 16384, (c) clean post-reboot memory state.

`latest.json` points at run1 as the cool reference. Greedy parity between
runs is exact on all 5 profiles — see `noise-floor_408225bb1.txt`.

## Noise floor on this chip

Run2 sits 14–23% below run1 despite identical configuration. The drop is
uniform across profiles in the same direction — the signature of sustained-
load thermal throttling on a 24 GB M4, not measurement noise. A tighter
±5% noise floor is not achievable on this chip under back-to-back runs.

**Implication for future workstream gates.** Effect sizes smaller than
~20% cannot be gated on this laptop alone. WS effects in that range should
be either (a) confirmed over 3+ runs with cooldowns between, using the
median, or (b) deferred to a cooler-chassis reference (Mac mini / M5 Pro
16") before default-ON promotion.

## Pre-reset three-way comparison

The original three-way comparison (stock vs. fork-off vs. fork-on) lives in
the Bluefin repo at `docs/evidence/llama-apple-m4-gemma4-e4b-q4km/`. It is
**not** duplicated here. This directory holds only post-reset stock baselines.
