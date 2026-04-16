#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def key_for(entry: dict[str, Any]) -> tuple[str, str]:
    return str(entry["run_name"]), str(entry["profile"])


def summary_for(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary")
    if isinstance(summary, dict):
        return summary

    response = entry["response"]
    timings = response.get("timings", {})

    return {
        "ttft_ms": float(timings.get("prompt_ms") or 0.0),
        "prompt_toks_per_s": float(timings.get("prompt_per_second") or 0.0),
        "decode_toks_per_s": float(timings.get("predicted_per_second") or 0.0),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}

    for entry in results:
        key = key_for(entry)
        summary = summary_for(entry)
        metrics = {
            "ttft_ms": float(summary.get("ttft_ms") or 0.0),
            "prompt_tps": float(summary.get("prompt_toks_per_s") or 0.0),
            "decode_tps": float(summary.get("decode_toks_per_s") or 0.0),
        }
        if key not in grouped:
            grouped[key] = {metric: [] for metric in metrics}
        for metric, value in metrics.items():
            grouped[key][metric].append(value)

    return {
        key: {metric: median(values) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def expected_key_map(doc: dict[str, Any]) -> dict[tuple[str, str], int]:
    expected_rows = doc.get("expected_results")
    if not isinstance(expected_rows, list):
        raise ValueError("result file is missing expected_results metadata")

    expected: dict[tuple[str, str], int] = {}
    for row in expected_rows:
        if not isinstance(row, dict):
            continue
        key = (str(row["run_name"]), str(row["profile"]))
        expected[key] = int(row.get("repeat", 1))
    return expected


def validate_document(doc: dict[str, Any], label: str) -> None:
    validation_errors = doc.get("validation_errors") or []
    if validation_errors:
        message = "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(f"{label} already contains matrix validation errors:\n{message}")

    expected = expected_key_map(doc)
    actual: dict[tuple[str, str], int] = {}
    for entry in doc.get("results", []):
        key = key_for(entry)
        actual[key] = actual.get(key, 0) + 1

    missing = [key for key, repeat in expected.items() if actual.get(key, 0) != repeat]
    if missing:
        formatted = ", ".join(f"{run}/{profile}" for run, profile in missing)
        raise ValueError(f"{label} is missing required result rows: {formatted}")


def pct_delta(baseline: float, candidate: float) -> str:
    if baseline == 0.0:
        return "n/a"
    return f"{((candidate - baseline) / baseline) * 100.0:+.2f}%"


def first_iteration_entry(results: list[dict[str, Any]], key: tuple[str, str]) -> dict[str, Any] | None:
    for entry in results:
        if key_for(entry) != key:
            continue
        if int(entry.get("iteration", -1)) == 0:
            return entry
    return None


def response_content(entry: dict[str, Any] | None) -> str | None:
    if entry is None:
        return None
    response = entry.get("response")
    if not isinstance(response, dict):
        return None
    content = response.get("content")
    if not isinstance(content, str):
        return None
    return content


def first_divergence(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def clip_window(text: str, center: int, radius: int = 20) -> str:
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    return text[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Apple Silicon benchmark result files.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    try:
        baseline = load(args.baseline)
        candidate = load(args.candidate)

        validate_document(baseline, "baseline")
        validate_document(candidate, "candidate")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    baseline_expected = expected_key_map(baseline)
    candidate_expected = expected_key_map(candidate)
    if baseline_expected != candidate_expected:
        baseline_only = sorted(set(baseline_expected) - set(candidate_expected))
        candidate_only = sorted(set(candidate_expected) - set(baseline_expected))
        details: list[str] = []
        if baseline_only:
            details.append("baseline-only: " + ", ".join(f"{run}/{profile}" for run, profile in baseline_only))
        if candidate_only:
            details.append("candidate-only: " + ", ".join(f"{run}/{profile}" for run, profile in candidate_only))
        print("benchmark definitions do not match:\n" + "\n".join(details), file=sys.stderr)
        return 2

    baseline_map = aggregate(baseline["results"])
    candidate_map = aggregate(candidate["results"])

    keys = sorted(set(baseline_map) & set(candidate_map))
    headers = ["run", "profile", "metric", "baseline", "candidate", "delta"]
    rows: list[list[str]] = []
    widths = [len(header) for header in headers]

    metrics = ["ttft_ms", "prompt_tps", "decode_tps"]

    for key in keys:
        base = baseline_map[key]
        cand = candidate_map[key]
        for metric in metrics:
            row = [
                key[0],
                key[1],
                metric,
                f"{base[metric]:.4f}",
                f"{cand[metric]:.4f}",
                pct_delta(base[metric], cand[metric]),
            ]
            rows.append(row)
            for idx, value in enumerate(row):
                widths[idx] = max(widths[idx], len(value))

    def fmt(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in rows:
        print(fmt(row))

    # Greedy parity section — compare response.content between first iterations.
    print()
    print("parity")
    for key in keys:
        base_entry = first_iteration_entry(baseline["results"], key)
        cand_entry = first_iteration_entry(candidate["results"], key)
        base_content = response_content(base_entry)
        cand_content = response_content(cand_entry)
        label = f"{key[0]}/{key[1]}"

        if base_content is None or cand_content is None:
            print(f"parity: {label}: MISSING_CONTENT")
            continue

        if base_content == cand_content:
            print(f"parity: {label}: ok ({len(base_content)} chars)")
            continue

        i = first_divergence(base_content, cand_content)
        print(f"parity: {label}: DIVERGED @ char {i}")
        print(f"    baseline:  ...{clip_window(base_content, i)}...")
        print(f"    candidate: ...{clip_window(cand_content, i)}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
