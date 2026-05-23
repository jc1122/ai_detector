#!/usr/bin/env python3
"""Benchmark helper for `detector_daemon` JSONL scoring performance."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ModuleNotFoundError:
    psutil = None
    _HAS_PSUTIL = False


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return parsed


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if lowered in {"0", "false", "no", "off", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("invalid boolean value")


def _percentile(sorted_values: list[float], *, p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    p = max(0.0, min(1.0, p))
    rank = p * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]

    low_value = sorted_values[lower]
    high_value = sorted_values[upper]
    weight = rank - lower
    return low_value + (high_value - low_value) * weight


def _validate_scoring_response(response: dict[str, Any]) -> None:
    if response.get("command") == "shutdown":
        return
    if "error" in response:
        raise RuntimeError(f"daemon request failed: {response.get('error')}")
    for key in ("text_preview", "weights", "experts", "ensemble", "calibration", "device"):
        if key not in response:
            raise RuntimeError(f"missing response key '{key}'")

    experts = response["experts"]
    if not isinstance(experts, dict):
        raise RuntimeError("response.experts must be an object")
    for key in ("meld", "tmr", "raid"):
        if key not in experts:
            raise RuntimeError(f"response.experts missing key '{key}'")
        expert_payload = experts[key]
        if not isinstance(expert_payload, dict):
            raise RuntimeError(f"response.experts.{key} must be an object")
        for key in ("ai_score", "human_score", "ai_probability", "human_probability", "chunks", "loaded"):
            if key not in expert_payload:
                raise RuntimeError(f"response.experts[{key}] is missing key '{key}'")

    ensemble = response["ensemble"]
    if not isinstance(ensemble, dict):
        raise RuntimeError("response.ensemble must be an object")
    for key in ("ai_score", "human_score", "ai_probability", "human_probability", "threshold", "label"):
        if key not in ensemble:
            raise RuntimeError(f"response.ensemble missing key '{key}'")

    calibration = response["calibration"]
    if not isinstance(calibration, dict):
        raise RuntimeError("response.calibration must be an object")
    for key in ("status", "calibrated", "message"):
        if key not in calibration:
            raise RuntimeError(f"response.calibration missing key '{key}'")


def _humanize_text_by_chars(template: str, target_chars: int) -> str:
    text = ""
    if target_chars <= 0:
        return text

    pieces = []
    while len(" ".join(pieces)) < target_chars:
        pieces.append(template)
    return " ".join(pieces)[:target_chars]


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    experts: str
    threads: int
    batch_size: int
    max_chunks: int
    iterations: int
    warmup: int


@dataclass
class BenchmarkResult:
    scenario: BenchmarkScenario
    startup_ms: float
    latencies_ms: list[float]
    peak_rss_mb: float | None

    def median_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        sorted_latencies = sorted(self.latencies_ms)
        return _percentile(sorted_latencies, p=0.5)

    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        return _percentile(sorted(self.latencies_ms), p=0.95)

    def mean_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        return statistics.mean(self.latencies_ms)

    def min_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        return min(self.latencies_ms)

    def max_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        return max(self.latencies_ms)


class DaemonProcessRunner:
    def __init__(
        self,
        command: list[str],
        *,
        timeout_s: float,
        env: dict[str, str],
        workdir: Path | None = None,
    ) -> None:
        self.command = command
        self.timeout_s = timeout_s
        self.env = env
        self.workdir = workdir
        self.proc: subprocess.Popen[str] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stderr_tail: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "DaemonProcessRunner":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        self.stop()
        return None

    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, ""):
            if len(self._stderr_tail) < 200:
                self._stderr_tail.append(line.rstrip())

    def start(self) -> None:
        if self.proc is not None:
            return

        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.env,
            cwd=str(self.workdir) if self.workdir is not None else None,
        )
        if self.proc.stdin is None or self.proc.stdout is None or self.proc.stderr is None:
            raise RuntimeError("daemon process did not expose stdio pipes")

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_line(self, timeout: float) -> str:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("daemon process is not running")
        future: Future[str] = self._executor.submit(self.proc.stdout.readline)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            raise RuntimeError(f"timeout while reading daemon response after {timeout:.1f}s") from exc

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                try:
                    self._send_request_no_timing({"command": "shutdown"})
                except Exception:
                    pass
            if self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except ProcessLookupError:
                    return
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
            finally:
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        finally:
            self.proc = None

    def _send_request_no_timing(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("daemon process not running")
        body = json.dumps(payload, ensure_ascii=False)
        self.proc.stdin.write(body + "\n")
        self.proc.stdin.flush()
        while True:
            line = self._read_line(self.timeout_s)
            if not line:
                raise RuntimeError("daemon closed connection while waiting for response")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                response = json.loads(stripped)
            except json.JSONDecodeError:
                if len(self._stderr_tail) < 200:
                    self._stderr_tail.append(stripped)
                if "error" in stripped.lower():
                    raise RuntimeError(f"daemon emitted non-json error output: {stripped}")
                continue
            if not isinstance(response, dict):
                raise RuntimeError("daemon response is not a JSON object")
            return response

    def send_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        start = time.perf_counter()
        response = self._send_request_no_timing(payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return response, elapsed_ms

    @property
    def peak_rss_mb(self) -> float | None:
        if not _HAS_PSUTIL or self.proc is None or self.proc.pid is None:
            return None
        process = psutil.Process(self.proc.pid)  # type: ignore[attr-defined]
        mem = process.memory_info().rss
        return mem / (1024 * 1024)


def _build_request_payload(*, text: str, batch_size: int, max_chunks: int, quiet: bool) -> dict[str, Any]:
    return {
        "text": text,
        "batch_size": batch_size,
        "max_chunks": max_chunks,
        "quiet": quiet,
    }


def run_scenario(
    scenario: BenchmarkScenario,
    runner: DaemonProcessRunner,
    *,
    request_text: str,
    quiet_requests: bool,
) -> BenchmarkResult:
    latencies: list[float] = []
    startup_ms = float("nan")
    peak_rss: float | None = None

    try:
        runner.start()
        health_request = {"command": "health"}
        health_response, startup_latency = runner.send_request(health_request)
        startup_ms = startup_latency
        if health_response.get("status") != "ok":
            raise RuntimeError(f"health response missing status or unhealthy: {health_response}")

        for _ in range(scenario.warmup):
            response, _ = runner.send_request(
                _build_request_payload(
                    text=request_text,
                    batch_size=scenario.batch_size,
                    max_chunks=scenario.max_chunks,
                    quiet=quiet_requests,
                ),
            )
            if response.get("error"):
                raise RuntimeError(response["error"])
            _validate_scoring_response(response)

        for _ in range(scenario.iterations):
            response, latency_ms = runner.send_request(
                _build_request_payload(
                    text=request_text,
                    batch_size=scenario.batch_size,
                    max_chunks=scenario.max_chunks,
                    quiet=quiet_requests,
                ),
            )
            if response.get("error"):
                raise RuntimeError(response["error"])
            _validate_scoring_response(response)
            latencies.append(latency_ms)
            rss = runner.peak_rss_mb
            if rss is not None and (peak_rss is None or rss > peak_rss):
                peak_rss = rss

        if peak_rss is None:
            peak_rss = runner.peak_rss_mb
    except Exception:
        raise
    finally:
        if runner.proc is not None:
            runner.stop()

    return BenchmarkResult(
        scenario=scenario,
        startup_ms=startup_ms,
        latencies_ms=latencies,
        peak_rss_mb=peak_rss,
    )


def run_command_baseline(command: list[str], *, text: str, repeats: int) -> tuple[list[float], float | None]:
    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = subprocess.run(
            command,
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cold CLI baseline failed: {result.stderr or result.stdout}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        payload = json.loads(result.stdout)
        _validate_scoring_response(payload)

    if not latencies:
        return [], None
    return latencies, statistics.mean(latencies)


def _format_ms(value: float) -> str:
    return f"{value:.2f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ai-detector-daemon JSONL throughput and startup timing.",
    )
    parser.add_argument("--daemon-cmd", default="python3 detector_daemon.py", help="Daemon command to run.")
    parser.add_argument(
        "--workdir",
        default=".",
        help="Project root or daemon launch directory.",
    )
    parser.add_argument(
        "--text-size",
        type=int,
        default=3600,
        help="Target size of synthetic request text (characters).",
    )
    parser.add_argument(
        "--tmr-threads",
        type=_parse_int_list,
        default="2,4,8",
        help="Comma-separated thread counts for TMR-only scenarios.",
    )
    parser.add_argument(
        "--tmr-batch-sizes",
        type=_parse_int_list,
        default="1,4,8",
        help="Comma-separated batch sizes for TMR-only scenarios.",
    )
    parser.add_argument(
        "--tmr-max-chunks",
        type=_parse_int_list,
        default="1,4",
        help="Comma-separated max_chunks values for TMR-only scenarios.",
    )
    parser.add_argument(
        "--tmr-iterations",
        type=int,
        default=1,
        help="Measured request iterations per TMR scenario.",
    )
    parser.add_argument(
        "--tmr-warmup",
        type=int,
        default=1,
        help="Warmup requests per TMR scenario (not included in timings).",
    )
    parser.add_argument(
        "--all-experts-threads",
        type=_parse_int_list,
        default="2",
        help="Comma-separated thread counts for all-experts scenarios.",
    )
    parser.add_argument(
        "--all-experts-batch-sizes",
        type=_parse_int_list,
        default="4",
        help="Comma-separated batch sizes for all-experts scenarios.",
    )
    parser.add_argument(
        "--all-experts-max-chunks",
        type=_parse_int_list,
        default="1",
        help="Comma-separated max_chunks for all-experts scenarios.",
    )
    parser.add_argument(
        "--all-experts-iterations",
        type=int,
        default=1,
        help="Measured request iterations per all-experts scenario.",
    )
    parser.add_argument(
        "--all-experts-warmup",
        type=int,
        default=1,
        help="Warmup requests per all-experts scenario.",
    )
    parser.add_argument(
        "--baseline-cold-runs",
        type=int,
        default=1,
        help="Iterations for one-off cold CLI baseline run.",
    )
    parser.add_argument(
        "--daemon-request-timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument("--quiet", type=_parse_bool, default=True, help="Set request quiet mode.")
    parser.add_argument(
        "--run-baseline",
        type=_parse_bool,
        default=False,
        help="Run optional cold cli baseline for context.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional JSON path to save raw benchmark results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.text_size <= 0:
        raise SystemExit("--text-size must be > 0")
    if args.tmr_iterations <= 0 or args.all_experts_iterations < 0:
        raise SystemExit("iterations must be non-negative")

    base_text = _humanize_text_by_chars(
        "Model quality signals depend on text rhythm, token boundaries, and the interaction between chunk boundaries and scoring head overlap, so we include a stable synthetic paragraph.",
        target_chars=args.text_size,
    )
    request_text = base_text

    daemon_env = os.environ.copy()
    daemon_env.update(
        {
            "OMP_NUM_THREADS": str(max(args.tmr_threads)),
            "MKL_NUM_THREADS": str(max(args.tmr_threads)),
            "PYTHONUNBUFFERED": "1",
        }
    )

    daemon_cmd = shlex.split(args.daemon_cmd)
    baseline_python = daemon_cmd[0]
    base_args = [
        "--local-files-only",
        "--quiet",
        "--device",
        "cpu",
    ]

    tmr_config: list[BenchmarkScenario] = []
    for threads in args.tmr_threads:
        for batch_size in args.tmr_batch_sizes:
            for max_chunks in args.tmr_max_chunks:
                tmr_config.append(
                    BenchmarkScenario(
                        name=f"tmr threads={threads} batch={batch_size} max_chunks={max_chunks}",
                        experts="tmr",
                        threads=threads,
                        batch_size=batch_size,
                        max_chunks=max_chunks,
                        iterations=args.tmr_iterations,
                        warmup=args.tmr_warmup,
                    )
                )

    all_experts_config: list[BenchmarkScenario] = []
    for threads in args.all_experts_threads:
        for batch_size in args.all_experts_batch_sizes:
            for max_chunks in args.all_experts_max_chunks:
                all_experts_config.append(
                    BenchmarkScenario(
                        name=f"all experts threads={threads} batch={batch_size} max_chunks={max_chunks}",
                        experts="meld,tmr,raid",
                        threads=threads,
                        batch_size=batch_size,
                        max_chunks=max_chunks,
                        iterations=args.all_experts_iterations,
                        warmup=args.all_experts_warmup,
                    )
                )

    all_results: list[BenchmarkResult] = []
    measured_total = 0
    warmup_total = 0
    baseline_latencies: list[float] | None = None
    baseline_summary: dict[str, float] | None = None

    print(f"Benchmark text: {len(request_text)} chars")
    print(f"Request settings: text len {len(request_text)} chars, quiet={args.quiet}")

    print(f"\nRunning {len(tmr_config)} TMR-only scenarios ...")
    for scenario in tmr_config:
        cmd = daemon_cmd + base_args + ["--experts", scenario.experts, "--threads", str(scenario.threads)]
        print(f"  launching: {' '.join(cmd)}")
        runner_env = dict(daemon_env)
        runner_env.update(
            {
                "OMP_NUM_THREADS": str(scenario.threads),
                "MKL_NUM_THREADS": str(scenario.threads),
            }
        )
        runner = DaemonProcessRunner(
            cmd,
            timeout_s=args.daemon_request_timeout,
            env=runner_env,
            workdir=Path(args.workdir),
        )
        try:
            result = run_scenario(
                scenario,
                runner,
                request_text=request_text,
                quiet_requests=args.quiet,
            )
            all_results.append(result)
            measured_total += scenario.iterations
            warmup_total += scenario.warmup
            print(
                f"    startup_ms={_format_ms(result.startup_ms)} | "
                f"measured={len(result.latencies_ms)} | median={_format_ms(result.median_ms())} | "
                f"p95={_format_ms(result.p95_ms())} | mean={_format_ms(result.mean_ms())} | "
                f"min={_format_ms(result.min_ms())} | max={_format_ms(result.max_ms())} | "
                f"peak_rss_mb={_format_ms(result.peak_rss_mb) if result.peak_rss_mb is not None else 'n/a'}"
            )
        finally:
            runner.stop()

    print(f"\nRunning {len(all_experts_config)} all-experts scenarios ...")
    for scenario in all_experts_config:
        cmd = daemon_cmd + base_args + ["--experts", scenario.experts, "--threads", str(scenario.threads)]
        print(f"  launching: {' '.join(cmd)}")
        runner_env = dict(daemon_env)
        runner_env.update(
            {
                "OMP_NUM_THREADS": str(scenario.threads),
                "MKL_NUM_THREADS": str(scenario.threads),
            }
        )
        runner = DaemonProcessRunner(
            cmd,
            timeout_s=args.daemon_request_timeout,
            env=runner_env,
            workdir=Path(args.workdir),
        )
        try:
            result = run_scenario(
                scenario,
                runner,
                request_text=request_text,
                quiet_requests=args.quiet,
            )
            all_results.append(result)
            measured_total += scenario.iterations
            warmup_total += scenario.warmup
            print(
                f"    startup_ms={_format_ms(result.startup_ms)} | "
                f"measured={len(result.latencies_ms)} | median={_format_ms(result.median_ms())} | "
                f"p95={_format_ms(result.p95_ms())} | mean={_format_ms(result.mean_ms())} | "
                f"min={_format_ms(result.min_ms())} | max={_format_ms(result.max_ms())} | "
                f"peak_rss_mb={_format_ms(result.peak_rss_mb) if result.peak_rss_mb is not None else 'n/a'}"
            )
        finally:
            runner.stop()

    if args.run_baseline:
        baseline_command = [
            baseline_python,
            "run_ensemble.py",
            "--text",
            request_text,
            "--json",
            "--local-files-only",
            "--quiet",
        ]
        print("\nRunning cold CLI baseline ...")
        baseline_latencies, baseline_mean = run_command_baseline(
            baseline_command,
            text=request_text,
            repeats=args.baseline_cold_runs,
        )
        if not baseline_latencies:
            raise RuntimeError("no baseline timings captured")
        baseline_sorted = sorted(baseline_latencies)
        baseline_summary = {
            "runs": len(baseline_latencies),
            "median_ms": float(_percentile(baseline_sorted, p=0.5)),
            "mean_ms": float(baseline_mean or 0.0),
            "min_ms": float(min(baseline_sorted)),
            "max_ms": float(max(baseline_sorted)),
            "p95_ms": float(_percentile(baseline_sorted, p=0.95)),
        }
        print(
            f"  baseline_runs={int(baseline_summary['runs'])} "
            f"mean_ms={_format_ms(baseline_summary['mean_ms'])}"
        )

    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "command": {
                "daemon_cmd": args.daemon_cmd,
                "workdir": args.workdir,
                "text_size": args.text_size,
            },
            "settings": {
                "tmr_threads": list(args.tmr_threads),
                "tmr_batch_sizes": list(args.tmr_batch_sizes),
                "tmr_max_chunks": list(args.tmr_max_chunks),
                "tmr_iterations": args.tmr_iterations,
                "tmr_warmup": args.tmr_warmup,
                "all_experts_threads": list(args.all_experts_threads),
                "all_experts_batch_sizes": list(args.all_experts_batch_sizes),
                "all_experts_max_chunks": list(args.all_experts_max_chunks),
                "all_experts_iterations": args.all_experts_iterations,
                "all_experts_warmup": args.all_experts_warmup,
                "baseline_runs": args.baseline_cold_runs if args.run_baseline else 0,
            },
            "results": [
                {
                    "name": result.scenario.name,
                    "experts": result.scenario.experts,
                    "threads": result.scenario.threads,
                    "batch_size": result.scenario.batch_size,
                    "max_chunks": result.scenario.max_chunks,
                    "startup_ms": result.startup_ms,
                    "latencies_ms": result.latencies_ms,
                    "median_ms": result.median_ms(),
                    "p95_ms": result.p95_ms(),
                    "mean_ms": result.mean_ms(),
                    "min_ms": result.min_ms(),
                    "max_ms": result.max_ms(),
                    "peak_rss_mb": result.peak_rss_mb,
                }
                for result in all_results
            ],
            "warmup_total": warmup_total,
            "measured_total": measured_total,
            "iteration_total": measured_total + warmup_total,
            "baseline": {
                "runs": 0,
                "latencies_ms": [],
                "median_ms": None,
                "p95_ms": None,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
            },
        }
        if baseline_summary is not None and baseline_latencies is not None:
            payload["baseline"] = {
                "runs": int(baseline_summary["runs"]),
                "latencies_ms": baseline_latencies,
                "median_ms": baseline_summary["median_ms"],
                "p95_ms": baseline_summary["p95_ms"],
                "mean_ms": baseline_summary["mean_ms"],
                "min_ms": baseline_summary["min_ms"],
                "max_ms": baseline_summary["max_ms"],
                "command": baseline_command,
            }
        Path(args.results_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if measured_total < 10:
        raise SystemExit("benchmark completed with fewer than 10 measured iterations")

    print(
        f"\nCompleted with {len(all_results)} scenario results, "
        f"{measured_total} measured + {warmup_total} warmup requests."
    )

    sorted_results = sorted(all_results, key=lambda entry: entry.median_ms())
    best = sorted_results[0]
    print("Fastest median scenario:")
    print(f"  {best.scenario.name} | median={_format_ms(best.median_ms())} ms | p95={_format_ms(best.p95_ms())} ms")


if __name__ == "__main__":
    main()
