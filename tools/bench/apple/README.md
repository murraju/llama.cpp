# Apple Silicon llama-server benchmark harness

Part of the Bluefin Gemma 4 x Apple Silicon engineering spec. Drives stock
(upstream) `llama-server` through a fixed profile matrix and compares result
files between builds.

## Files

- `matrix.yaml` — profile + run definitions (JSON-in-.yaml; parsed as JSON)
- `run_matrix.py` — launches `llama-server`, walks the matrix, writes a result JSON
- `compare.py` — diffs two result JSONs; reports metric deltas and greedy parity
- `tokenizer_compat.py` — hashes `tokenizer.ggml.tokens` to gate drafter pairs
- `prompts/` — seed prompts for each profile (cold_short, warm_short, warm_rag, warm_longctx)
- `baselines/` — versioned reference captures per chip and quantization

## Matrix notes

- `prompt_mode` is either `file` (verbatim seed) or `expand_to_tokens` (binary
  search to hit `target_prompt_tokens`).
- `process_isolation` is one of `shared`, `per_profile`, `per_iteration`.
  Cold-start profiles always spawn a fresh server regardless of isolation.
- `request_overrides` lets a profile override `cache_prompt`, sampling, etc.
  per request. Long-context profiles disable `cache_prompt` so the prefill is
  measured end-to-end.

## Typical usage

Run the matrix against a built `llama-server`:

```bash
python3 tools/bench/apple/run_matrix.py \
  --matrix tools/bench/apple/matrix.yaml \
  --server-bin build/bin/llama-server \
  --output /tmp/apple-bench/stock-run1.json
```

Diff two runs:

```bash
python3 tools/bench/apple/compare.py \
  --baseline /tmp/apple-bench/stock-run1.json \
  --candidate /tmp/apple-bench/stock-run2.json
```

## Greedy parity

`compare.py` prints a `parity` section after the metric-delta table. For every
`(run, profile)` pair present in both files it compares `response.content`
between the first iteration of each side. Temperature is pinned to 0 and
`top_k=1` in the matrix, so a stock vs. stock run should produce identical
output strings. Divergence prints the first-differing character index plus a
clipped window of each side. Parity divergence is informational only — this
tool is a noise-floor diagnostic, not a strict gate. Downstream workstreams
enforce correctness criteria.

## Notes

- `parity_check.py` was deliberately dropped — parity is folded into
  `compare.py` as the post-metric section above.
- The harness is stdlib-only; `matrix.yaml` uses JSON syntax on purpose.
