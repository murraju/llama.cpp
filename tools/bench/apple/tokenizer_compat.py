#!/usr/bin/env python3
"""
Tokenizer compatibility check for speculative decoding drafter pairs.

Speculative decoding requires the drafter and target models to share an
identical tokenizer. If their vocabularies diverge, accept/reject cannot
operate at the token level. This script hashes the GGUF tokenizer.ggml.tokens
array for two models and fails loudly if they differ.

Used as a gate before wiring drafter pairs (e.g. Gemma 4 E2B drafting Gemma 4
E4B).

Usage:
  python3 tools/bench/apple/tokenizer_compat.py <drafter.gguf> <target.gguf>

Exit 0: tokenizers match.
Exit 1: tokenizers differ (drafter pair is not safe).
Exit 2: invocation or file error.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
GGUF_PY = REPO_ROOT / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

try:
    from gguf import GGUFReader
except ImportError as exc:
    print(f"error: cannot import gguf module: {exc}", file=sys.stderr)
    print(f"hint: expected gguf-py at {GGUF_PY}", file=sys.stderr)
    sys.exit(2)


def tokenizer_fingerprint(gguf_path: Path) -> dict:
    reader = GGUFReader(str(gguf_path), "r")
    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    if tokens_field is None:
        raise RuntimeError(f"{gguf_path}: missing tokenizer.ggml.tokens")

    # Each entry is a byte string; canonicalize to bytes then hash.
    token_bytes = []
    for idx in tokens_field.data:
        raw = bytes(tokens_field.parts[idx])
        token_bytes.append(raw)

    h = hashlib.sha256()
    for tok in token_bytes:
        h.update(len(tok).to_bytes(4, "little"))
        h.update(tok)

    # Optional: also fingerprint scores + token_types for a stricter match.
    extras = {}
    for extra_key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
        field = reader.get_field(extra_key)
        if field is None:
            continue
        eh = hashlib.sha256()
        for idx in field.data:
            eh.update(bytes(field.parts[idx]))
        extras[extra_key] = eh.hexdigest()

    return {
        "path": str(gguf_path),
        "n_tokens": len(token_bytes),
        "tokens_sha256": h.hexdigest(),
        "extras": extras,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafter", type=Path, help="drafter GGUF path")
    parser.add_argument("target", type=Path, help="target GGUF path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also require scores and token types to match",
    )
    args = parser.parse_args()

    for p in (args.drafter, args.target):
        if not p.is_file():
            print(f"error: not a file: {p}", file=sys.stderr)
            return 2

    try:
        drafter = tokenizer_fingerprint(args.drafter)
        target = tokenizer_fingerprint(args.target)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def pretty(label: str, fp: dict) -> None:
        print(f"  {label}: {fp['path']}")
        print(f"    n_tokens:      {fp['n_tokens']}")
        print(f"    tokens_sha256: {fp['tokens_sha256']}")
        for k, v in fp["extras"].items():
            print(f"    {k:22s} {v}")

    pretty("drafter", drafter)
    pretty("target ", target)

    core_match = (
        drafter["n_tokens"] == target["n_tokens"]
        and drafter["tokens_sha256"] == target["tokens_sha256"]
    )
    extras_match = drafter["extras"] == target["extras"]

    if core_match and (extras_match or not args.strict):
        print("OK: tokenizers match; drafter pair is safe.")
        return 0

    print("FAIL: tokenizer mismatch — drafter pair would silently produce wrong tokens.")
    if not core_match:
        print("  - n_tokens or token-bytes differ")
    if args.strict and not extras_match:
        print("  - scores or token_type differ (strict mode)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
