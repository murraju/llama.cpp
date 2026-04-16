# Apple Silicon benchmark baselines

Versioned reference captures from `run_matrix.py`, organized per chip and
quantization so new builds can be compared against a known-good run.

## Layout

```
baselines/<chip>/<quant>/
  stock_<YYYY-MM-DD>_<upstream-short-sha>_runN.json
  latest.json                -> symlink to the current reference run
  noise-floor_<upstream-short-sha>.txt
```

- `<chip>` is a slug like `apple-m4`, `apple-m4-pro`, `apple-m4-max`.
- `<quant>` is the GGUF quant label (e.g. `Q4_K_M`, `Q5_K_M`).
- Each capture file name encodes the capture date and the upstream commit the
  fork was synced to at capture time. `runN` distinguishes repeated captures
  on the same (date, sha) pair — at least two are taken to establish a noise
  floor.
- `latest.json` is a symlink in each chip/quant directory pointing at the
  current reference run. Tools should prefer `latest.json`; the concrete
  file name exists for reproducibility.
- `noise-floor_<upstream-short-sha>.txt` is the output of
  `python3 tools/bench/apple/compare.py --baseline run1.json --candidate run2.json`.
  It quantifies the stock-vs-stock run-to-run variation so downstream diffs
  can be interpreted against real noise rather than an absolute threshold.

## Recapture policy

Refresh the baseline when any of the following is true:

- The fork rebases onto a newer upstream sha.
- A new chip family is introduced (new directory under `baselines/`).
- Decode tok/s for any profile drifts more than 5% between two consecutive
  weekly sanity runs on the same build.

## Phase 0 coverage

Only `apple-m4/Q4_K_M/` is populated in Phase 0. Additional chip directories
(M4 Pro, M4 Max, sidecar-class SKUs) land as that hardware comes online.
