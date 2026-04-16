#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def timeout_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default

    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


DEFAULT_SERVER_STARTUP_TIMEOUT_S = timeout_env("LLAMA_BENCH_SERVER_STARTUP_TIMEOUT_S", 60.0)
DEFAULT_REQUEST_TIMEOUT_S = timeout_env("LLAMA_BENCH_REQUEST_TIMEOUT_S", 180.0)
DEFAULT_TOKENIZE_TIMEOUT_S = timeout_env("LLAMA_BENCH_TOKENIZE_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S)


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_prompt(base_dir: Path, relative_path: str) -> str:
    with (base_dir / relative_path).open("r", encoding="utf-8") as handle:
        return handle.read()


def pick_port(preferred: int) -> int:
    if preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def system_value(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return ""


def hardware_profile() -> dict[str, Any]:
    mem_bytes = system_value(["sysctl", "-n", "hw.memsize"])
    mem_gb = int(int(mem_bytes) / (1024 ** 3)) if mem_bytes.isdigit() else 0

    return {
        "chip": system_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "machine": system_value(["sysctl", "-n", "hw.model"]),
        "memory_bytes": int(mem_bytes) if mem_bytes.isdigit() else 0,
        "memory_gb": mem_gb,
        "memory_class": f"{mem_gb}GB" if mem_gb else "unknown",
        "os_version": system_value(["sw_vers", "-productVersion"]),
    }


def git_info(repo_root: Path) -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": run_git(["describe", "--always", "--dirty"]),
    }


def wait_for_server(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "server did not become ready"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(last_error)


def launch_server(run: dict[str, Any], server_bin: Path, log_path: Path) -> subprocess.Popen[str]:
    server_cfg = run["server"]
    artifacts = run["artifacts"]
    host = server_cfg.get("host", "127.0.0.1")
    port = pick_port(int(server_cfg.get("port", 0)))
    server_cfg["port"] = port

    args = [
        str(server_bin),
        "--host",
        host,
        "--port",
        str(port),
        "-m",
        artifacts["model_path"],
        "-c",
        str(server_cfg["ctx_size"]),
        "--no-webui",
    ]
    args.extend(server_cfg.get("server_args", []))

    env = os.environ.copy()
    for key, value in server_cfg.get("env", {}).items():
        env[str(key)] = str(value)

    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        wait_for_server(host, port, timeout_s=DEFAULT_SERVER_STARTUP_TIMEOUT_S)
    except Exception:
        process.terminate()
        process.wait(timeout=10.0)
        raise

    return process


def stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15.0)


def post_json(host: str, port: int, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def post_completion(host: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    return post_json(host, port, "/completion", payload, timeout_s=DEFAULT_REQUEST_TIMEOUT_S)


def tokenize_prompt(host: str, port: int, prompt: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
    response = post_json(
        host,
        port,
        "/tokenize",
        {
            "content": prompt,
            "add_special": add_special,
            "parse_special": parse_special,
        },
        timeout_s=DEFAULT_TOKENIZE_TIMEOUT_S,
    )

    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("tokenize response did not include a token list")

    return [int(token) for token in tokens]


def prompt_token_count(host: str, port: int, prompt: str) -> int:
    return len(tokenize_prompt(host, port, prompt))


def build_context_block(seed_prompt: str, index: int) -> str:
    return textwrap.dedent(
        f"""\
        [context-block {index:04d}]
        {seed_prompt.strip()}
        """
    ).strip() + "\n"


def build_context_tail(seed_lines: list[str], line_count: int) -> str:
    return "".join(f"{line}\n" for line in seed_lines[:line_count])


def build_context_prompt(seed_prompt: str, block_count: int, tail_line_count: int = 0) -> str:
    seed_lines = [line.rstrip() for line in seed_prompt.splitlines() if line.strip()]
    pieces = [build_context_block(seed_prompt, idx) for idx in range(1, block_count + 1)]
    if tail_line_count > 0:
        pieces.append(build_context_tail(seed_lines, tail_line_count))
    return "".join(pieces)


def expand_prompt_to_target_tokens(
    host: str,
    port: int,
    seed_prompt: str,
    target_prompt_tokens: int,
) -> tuple[str, int]:
    if target_prompt_tokens <= 0:
        raise ValueError("target_prompt_tokens must be positive")

    chunk_one = build_context_block(seed_prompt, 1)
    chunk_tokens = prompt_token_count(host, port, chunk_one)
    if chunk_tokens <= 0:
        raise RuntimeError("failed to tokenize long-context seed prompt")

    estimate = max(1, math.ceil(target_prompt_tokens / chunk_tokens))
    low = 0
    high = max(1, estimate)
    best_prompt = seed_prompt
    best_count = prompt_token_count(host, port, seed_prompt)

    high_prompt = build_context_prompt(seed_prompt, high)
    high_count = prompt_token_count(host, port, high_prompt)

    while high_count <= target_prompt_tokens:
        low = high
        best_prompt = high_prompt
        best_count = high_count
        high *= 2
        high_prompt = build_context_prompt(seed_prompt, high)
        high_count = prompt_token_count(host, port, high_prompt)

    while low + 1 < high:
        mid = (low + high) // 2
        mid_prompt = build_context_prompt(seed_prompt, mid)
        mid_count = prompt_token_count(host, port, mid_prompt)
        if mid_count <= target_prompt_tokens:
            low = mid
            best_prompt = mid_prompt
            best_count = mid_count
        else:
            high = mid

    seed_lines = [line.rstrip() for line in seed_prompt.splitlines() if line.strip()]
    if not seed_lines:
        return best_prompt, best_count

    tail_low = 0
    tail_high = len(seed_lines) + 1
    while tail_low + 1 < tail_high:
        mid = (tail_low + tail_high) // 2
        mid_prompt = build_context_prompt(seed_prompt, low, mid)
        mid_count = prompt_token_count(host, port, mid_prompt)
        if mid_count <= target_prompt_tokens:
            tail_low = mid
            best_prompt = mid_prompt
            best_count = mid_count
        else:
            tail_high = mid

    return best_prompt, best_count


def process_isolation_mode(run: dict[str, Any], profile: dict[str, Any]) -> str:
    return str(profile.get("process_isolation") or run.get("process_isolation") or "shared")


def profile_request(run: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    payload = dict(run["request"])
    payload.update(profile.get("request_overrides", {}))
    return payload


def profile_prompt(
    host: str,
    port: int,
    matrix_dir: Path,
    run: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[str, int]:
    prompt_file = str(profile["prompt_file"])
    prompt_seed = read_prompt(matrix_dir, prompt_file)
    prompt_mode = str(profile.get("prompt_mode", "file"))
    max_prompt_tokens = int(run["server"]["ctx_size"]) - int(profile["n_predict"])
    if max_prompt_tokens <= 0:
        raise RuntimeError("ctx_size must exceed n_predict for benchmarking")

    if prompt_mode == "file":
        prompt_tokens = prompt_token_count(host, port, prompt_seed)
        if prompt_tokens > max_prompt_tokens:
            raise RuntimeError(
                f"profile prompt exceeds ctx budget: {prompt_tokens} > {max_prompt_tokens}"
            )
        return prompt_seed, prompt_tokens

    if prompt_mode == "expand_to_tokens":
        target_prompt_tokens = int(profile["target_prompt_tokens"])
        if target_prompt_tokens > max_prompt_tokens:
            raise RuntimeError(
                f"target_prompt_tokens exceeds ctx budget: {target_prompt_tokens} > {max_prompt_tokens}"
            )
        prompt, prompt_tokens = expand_prompt_to_target_tokens(
            host,
            port,
            prompt_seed,
            target_prompt_tokens,
        )
        if prompt_tokens > max_prompt_tokens:
            raise RuntimeError(
                f"expanded prompt exceeds ctx budget: {prompt_tokens} > {max_prompt_tokens}"
            )
        return prompt, prompt_tokens

    raise ValueError(f"unsupported prompt_mode: {prompt_mode}")


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings", {})

    return {
        "ttft_ms": timings.get("prompt_ms"),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_toks_per_s": timings.get("prompt_per_second"),
        "decode_tokens": timings.get("predicted_n"),
        "decode_toks_per_s": timings.get("predicted_per_second"),
    }


def summarize_result(entry: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_response(entry["response"])
    summary["target_context_bucket"] = entry["target_context_bucket"]
    summary["actual_prompt_tokens"] = entry["actual_prompt_tokens"]
    summary["process_isolation_mode"] = entry["process_isolation_mode"]
    return summary


def expected_result_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = matrix["profiles"]

    for run in matrix["runs"]:
        profile_order = run.get("profile_order") or list(profiles.keys())
        for profile_name in profile_order:
            profile = profiles[profile_name]
            rows.append(
                {
                    "run_name": run["name"],
                    "profile": profile_name,
                    "repeat": int(profile.get("repeat", 1)),
                    "target_context_bucket": profile.get("target_context_bucket", "none"),
                    "process_isolation_mode": process_isolation_mode(run, profile),
                }
            )

    return rows


def validate_results(results: list[dict[str, Any]], expected_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_counts = {(row["run_name"], row["profile"]): int(row["repeat"]) for row in expected_rows}
    actual_counts: dict[tuple[str, str], int] = {}

    for entry in results:
        key = (str(entry["run_name"]), str(entry["profile"]))
        actual_counts[key] = actual_counts.get(key, 0) + 1

    for key, expected_repeat in expected_counts.items():
        actual_repeat = actual_counts.get(key, 0)
        if actual_repeat != expected_repeat:
            errors.append(
                f"{key[0]}/{key[1]}: expected {expected_repeat} result(s) but found {actual_repeat}"
            )

    return errors


def print_summary(results: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "profile",
        "ctx_bucket",
        "prompt_tok",
        "isolation",
        "start",
        "ttft_ms",
        "prompt_t/s",
        "decode_t/s",
    ]
    rows: list[list[str]] = []
    widths = [len(header) for header in headers]

    for entry in results:
        summary = entry["summary"]
        cold_start = bool(entry.get("cold_start", False))
        row = [
            entry["run_name"],
            entry["profile"],
            str(summary["target_context_bucket"]),
            str(summary["actual_prompt_tokens"]),
            str(summary["process_isolation_mode"]),
            "cold" if cold_start else "warm",
            f"{summary['ttft_ms'] or 0:8.2f}",
            f"{summary['prompt_toks_per_s'] or 0:8.2f}",
            f"{summary['decode_toks_per_s'] or 0:8.2f}",
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


def run_matrix(matrix_path: Path, server_bin: Path, output_path: Path) -> None:
    matrix = load_matrix(matrix_path)
    matrix_dir = matrix_path.parent
    repo_root = Path(__file__).resolve().parents[3]

    results: list[dict[str, Any]] = []
    profiles = matrix["profiles"]
    expected_rows = expected_result_rows(matrix)

    for run in matrix["runs"]:
        profile_order = run.get("profile_order") or list(profiles.keys())
        shared_process: subprocess.Popen[str] | None = None

        try:
            for profile_name in profile_order:
                profile = profiles[profile_name]
                repeat = int(profile.get("repeat", 1))
                cold_start = bool(profile.get("cold_start", False))
                isolation_mode = process_isolation_mode(run, profile)
                per_profile_process: subprocess.Popen[str] | None = None

                if isolation_mode != "shared":
                    stop_server(shared_process)
                    shared_process = None

                for iteration in range(repeat):
                    process: subprocess.Popen[str] | None = None
                    started_for_iteration = False
                    log_path = Path(tempfile.gettempdir()) / f"{run['name']}-{profile_name}-{iteration}.log"

                    if cold_start or isolation_mode == "per_iteration":
                        process = launch_server(run, server_bin, log_path)
                        started_for_iteration = True
                    elif isolation_mode == "per_profile":
                        if per_profile_process is None or per_profile_process.poll() is not None:
                            per_profile_process = launch_server(run, server_bin, log_path)
                        process = per_profile_process
                    else:
                        if shared_process is None or shared_process.poll() is not None:
                            shared_process = launch_server(run, server_bin, log_path)
                        process = shared_process

                    if process is None:
                        raise RuntimeError("failed to acquire a server process")

                    host = run["server"].get("host", "127.0.0.1")
                    port = int(run["server"]["port"])
                    prompt, actual_prompt_tokens = profile_prompt(host, port, matrix_dir, run, profile)

                    payload = profile_request(run, profile)
                    payload["prompt"] = prompt
                    payload["n_predict"] = int(profile["n_predict"])

                    response = post_completion(host, port, payload)

                    entry = {
                        "run_name": run["name"],
                        "profile": profile_name,
                        "iteration": iteration,
                        "cold_start": cold_start,
                        "route_type": run.get("route", {}).get("type", "this-device"),
                        "artifacts": run["artifacts"],
                        "target_context_bucket": profile.get("target_context_bucket", "none"),
                        "actual_prompt_tokens": actual_prompt_tokens,
                        "process_isolation_mode": isolation_mode,
                        "response": response,
                    }
                    entry["summary"] = summarize_result(entry)
                    results.append(entry)

                    if started_for_iteration:
                        stop_server(process)

                stop_server(per_profile_process)
        finally:
            stop_server(shared_process)

    validation_errors = validate_results(results, expected_rows)

    output = {
        "captured_at_epoch_s": int(time.time()),
        "matrix_path": str(matrix_path),
        "matrix": matrix,
        "expected_results": expected_rows,
        "server_bin": str(server_bin),
        "git": git_info(repo_root),
        "hardware": hardware_profile(),
        "validation_errors": validation_errors,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print_summary(results)

    if validation_errors:
        raise RuntimeError("matrix validation failed:\n- " + "\n- ".join(validation_errors))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Apple Silicon llama-server benchmark matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Notes:
            - `matrix.yaml` intentionally uses JSON syntax so the harness stays stdlib-only.
            - Each run writes a single JSON result file with raw server responses, derived summaries, and validation metadata.
            """
        ),
    )
    parser.add_argument("--matrix", required=True, type=Path, help="Path to tools/bench/apple/matrix.yaml")
    parser.add_argument("--server-bin", required=True, type=Path, help="Path to llama-server binary")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()

    try:
        run_matrix(args.matrix.resolve(), args.server_bin.resolve(), args.output.resolve())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
