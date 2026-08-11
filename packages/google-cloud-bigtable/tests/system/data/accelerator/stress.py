# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone long-running / high-QPS stress driver for the accelerator.

This is deliberately *not* a pytest module (it is not collected: the filename
does not start with ``test_``). It is a soak/stress harness meant to be run by
hand or from CI against a real Bigtable instance, e.g. for the canonical
pre-release 6-hour run:

    python -m tests.system.data.accelerator.stress --hours 6 --qps 200

It drives the *real shipped path* end-to-end -- a real ``BigtableDataClient``
(async by default, or the generated sync client with ``--sync``) with
``use_accelerator`` on, which spawns the real ``AcceleratorDaemon`` over the real
bundled binary and routes ``read_row``/``mutate_row`` over the UDS. Nothing is
faked; ``--no-accelerator`` runs the identical load over the native path so the
two can be compared.

While it runs it holds a target QPS with a token bucket across N workers, each
owning a disjoint key prefix and its own ``ExpectedState``. Every mutation updates
the expected state and Bigtable; periodic canary reads assert Bigtable still
matches the expected state (correctness under sustained load). A monitor samples
RSS / open FDs and
checks daemon liveness on an interval, so resource leaks or a dead daemon are
caught. At the end it writes a JSON + text report and exits non-zero if any
breach threshold was crossed (error rate, canary drift, daemon death, or RSS
growth), so it is usable as a CI gate.

Config comes from the standard system-test env vars
(``GOOGLE_CLOUD_PROJECT``, ``BIGTABLE_TEST_INSTANCE``, ``BIGTABLE_TEST_TABLE``),
overridable by flags. The instance/table must already exist with the families
``test-family`` and ``test-family-2`` (the same the suite's conftest creates).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass

from google.cloud.bigtable.data import BigtableDataClient, BigtableDataClientAsync

from . import _harness

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StressConfig:
    project: str | None
    instance_id: str
    table_id: str
    duration_seconds: float
    qps: float
    workers: int
    use_accelerator: bool
    sync: bool
    sample_interval: float
    canary_every: int
    op_timeout: float
    # Breach thresholds (crossing any of these -> non-zero exit).
    error_threshold: float
    rss_growth_mb: float
    report_path: str | None
    progress_interval: float


def _build_config(argv: list[str]) -> StressConfig:
    p = argparse.ArgumentParser(
        prog="stress",
        description="Long-running / high-QPS accelerator stress driver.",
    )
    # Duration: --hours is the headline knob; --seconds overrides for smoke runs.
    p.add_argument("--hours", type=float, default=6.0, help="run duration in hours")
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="run duration in seconds (overrides --hours; for smoke tests)",
    )
    p.add_argument("--qps", type=float, default=100.0, help="target operations/sec")
    p.add_argument("--workers", type=int, default=8, help="concurrent workers")
    p.add_argument(
        "--no-accelerator",
        action="store_true",
        help="run the identical load over the native path (baseline)",
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help="use the generated sync client + OS threads instead of asyncio",
    )
    p.add_argument("--project", default=None, help="GCP project (default: env)")
    p.add_argument("--instance", default=None, help="instance id (default: env)")
    p.add_argument("--table", default=None, help="table id (default: env)")
    p.add_argument(
        "--sample-interval",
        type=float,
        default=30.0,
        help="seconds between RSS/FD/liveness samples",
    )
    p.add_argument(
        "--canary-every",
        type=int,
        default=25,
        help="verify a written row against the model every N ops per worker",
    )
    p.add_argument(
        "--op-timeout",
        type=float,
        default=30.0,
        help="per-operation timeout passed to read_row/mutate_row",
    )
    p.add_argument(
        "--error-threshold",
        type=float,
        default=0.01,
        help="max tolerated error fraction before the run is a failure",
    )
    p.add_argument(
        "--rss-growth-mb",
        type=float,
        default=256.0,
        help="max tolerated RSS growth (MB) before the run is a failure",
    )
    p.add_argument(
        "--progress-interval",
        type=float,
        default=60.0,
        help="seconds between progress lines on stderr",
    )
    p.add_argument(
        "--report",
        default=None,
        help="path to write the JSON report (also printed to stdout)",
    )
    args = p.parse_args(argv)

    duration = args.seconds if args.seconds is not None else args.hours * 3600.0
    if duration <= 0:
        p.error("duration must be positive")
    if args.qps <= 0:
        p.error("--qps must be positive")
    if args.workers <= 0:
        p.error("--workers must be positive")

    project = (
        args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    )
    instance_id = args.instance or os.getenv("BIGTABLE_TEST_INSTANCE")
    table_id = args.table or os.getenv("BIGTABLE_TEST_TABLE")
    if not instance_id:
        p.error("no instance id (pass --instance or set BIGTABLE_TEST_INSTANCE)")
    if not table_id:
        p.error("no table id (pass --table or set BIGTABLE_TEST_TABLE)")

    return StressConfig(
        project=project,
        instance_id=instance_id,
        table_id=table_id,
        duration_seconds=duration,
        qps=args.qps,
        workers=args.workers,
        use_accelerator=not args.no_accelerator,
        sync=args.sync,
        sample_interval=args.sample_interval,
        canary_every=args.canary_every,
        op_timeout=args.op_timeout,
        error_threshold=args.error_threshold,
        rss_growth_mb=args.rss_growth_mb,
        report_path=args.report,
        progress_interval=args.progress_interval,
    )


# ---------------------------------------------------------------------------
# Shared, thread-safe result accumulator
# ---------------------------------------------------------------------------


@dataclass
class ResourceSample:
    t: float
    rss_mb: float
    num_fds: int
    daemon_alive: bool


class StressStats:
    """Thread-safe accumulation of counters, latencies, and samples.

    A single lock guards everything; contention is negligible next to the RPC
    latencies being recorded, and it lets the async and threaded drivers share
    one implementation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mutate_latency = _harness.LatencyStats()
        self.read_latency = _harness.LatencyStats()
        self.total_ops = 0
        self.total_errors = 0
        self.error_histogram: dict[str, int] = {}
        self.canary_checks = 0
        self.canary_mismatches = 0
        self.mismatch_samples: list[str] = []
        self.samples: list[ResourceSample] = []
        self.daemon_deaths = 0

    def record_op(self, kind: str, seconds: float) -> None:
        with self._lock:
            self.total_ops += 1
            if kind == "read":
                self.read_latency.record(seconds)
            else:
                self.mutate_latency.record(seconds)

    def record_error(self, exc: BaseException) -> None:
        name = type(exc).__name__
        with self._lock:
            self.total_ops += 1
            self.total_errors += 1
            self.error_histogram[name] = self.error_histogram.get(name, 0) + 1

    def record_canary(self, mismatch_detail: str | None) -> None:
        with self._lock:
            self.canary_checks += 1
            if mismatch_detail is not None:
                self.canary_mismatches += 1
                # Keep only the first few, they are large and all indicate a bug.
                if len(self.mismatch_samples) < 5:
                    self.mismatch_samples.append(mismatch_detail)

    def record_sample(self, sample: ResourceSample) -> None:
        with self._lock:
            self.samples.append(sample)
            if not sample.daemon_alive:
                self.daemon_deaths += 1

    def error_rate(self) -> float:
        with self._lock:
            return self.total_errors / self.total_ops if self.total_ops else 0.0


# ---------------------------------------------------------------------------
# Shared helpers (no IO)
# ---------------------------------------------------------------------------


def _canary_mismatch(worker_id: int, key: bytes, row, expected_state) -> str | None:
    """Return a human-readable mismatch string, or None if the row matches."""
    got = _harness.normalize_row(row)
    expected = expected_state.expected_cells(key)
    if got == expected:
        return None
    return f"worker {worker_id} key {key!r}: got {got!r} expected {expected!r}"


def _worker_prefix(worker_id: int) -> bytes:
    # Disjoint, self-describing prefix per worker so writers never contend and a
    # mismatch points at a worker. A stable (run-agnostic) prefix keeps the
    # keyspace bounded across repeated runs; the soak reuses/overwrites rows.
    return f"stress-w{worker_id:03d}-".encode()


def _compute_breaches(cfg: StressConfig, stats: StressStats) -> list[str]:
    breaches: list[str] = []
    rate = stats.error_rate()
    if rate > cfg.error_threshold:
        breaches.append(
            f"error rate {rate:.4f} exceeds threshold {cfg.error_threshold:.4f}"
        )
    if stats.canary_mismatches > 0:
        breaches.append(
            f"{stats.canary_mismatches} canary correctness mismatch(es) "
            f"of {stats.canary_checks} checks"
        )
    if cfg.use_accelerator and stats.daemon_deaths > 0:
        breaches.append(
            f"daemon was observed dead on {stats.daemon_deaths} liveness sample(s)"
        )
    if stats.samples:
        rss_values = [s.rss_mb for s in stats.samples]
        growth = max(rss_values) - rss_values[0]
        if growth > cfg.rss_growth_mb:
            breaches.append(
                f"RSS grew {growth:.1f}MB (>{cfg.rss_growth_mb:.1f}MB): "
                f"start {rss_values[0]:.1f}MB peak {max(rss_values):.1f}MB"
            )
    return breaches


def _build_report(cfg: StressConfig, stats: StressStats, elapsed: float) -> dict:
    breaches = _compute_breaches(cfg, stats)
    rss_values = [s.rss_mb for s in stats.samples]
    fd_values = [s.num_fds for s in stats.samples]
    return {
        "config": {
            "mode": "sync" if cfg.sync else "async",
            "accelerator": cfg.use_accelerator,
            "duration_seconds": cfg.duration_seconds,
            "target_qps": cfg.qps,
            "workers": cfg.workers,
            "instance_id": cfg.instance_id,
            "table_id": cfg.table_id,
        },
        "elapsed_seconds": elapsed,
        "throughput": {
            "total_ops": stats.total_ops,
            "achieved_qps": stats.total_ops / elapsed if elapsed else 0.0,
            "total_errors": stats.total_errors,
            "error_rate": stats.error_rate(),
            "error_histogram": dict(stats.error_histogram),
        },
        "latency": {
            "mutate_row": stats.mutate_latency.summary_ms(),
            "read_row": stats.read_latency.summary_ms(),
        },
        "correctness": {
            "canary_checks": stats.canary_checks,
            "canary_mismatches": stats.canary_mismatches,
            "mismatch_samples": stats.mismatch_samples,
        },
        "resources": {
            "samples": len(stats.samples),
            "rss_mb_start": rss_values[0] if rss_values else None,
            "rss_mb_peak": max(rss_values) if rss_values else None,
            "rss_mb_end": rss_values[-1] if rss_values else None,
            "num_fds_start": fd_values[0] if fd_values else None,
            "num_fds_peak": max(fd_values) if fd_values else None,
            "num_fds_end": fd_values[-1] if fd_values else None,
            "daemon_deaths": stats.daemon_deaths,
        },
        "breaches": breaches,
        "passed": not breaches,
    }


def _print_report(report: dict) -> None:
    cfg = report["config"]
    tp = report["throughput"]
    res = report["resources"]
    lines = [
        "",
        "=" * 70,
        "ACCELERATOR STRESS REPORT",
        "=" * 70,
        (
            f"mode={cfg['mode']} accelerator={cfg['accelerator']} "
            f"workers={cfg['workers']} target_qps={cfg['target_qps']}"
        ),
        (
            f"elapsed={report['elapsed_seconds']:.1f}s "
            f"ops={tp['total_ops']} achieved_qps={tp['achieved_qps']:.1f}"
        ),
        (
            f"errors={tp['total_errors']} error_rate={tp['error_rate']:.4f} "
            f"histogram={tp['error_histogram']}"
        ),
        f"mutate_row latency={report['latency']['mutate_row']}",
        f"read_row   latency={report['latency']['read_row']}",
        (
            f"canary checks={report['correctness']['canary_checks']} "
            f"mismatches={report['correctness']['canary_mismatches']}"
        ),
        (
            f"rss_mb start/peak/end="
            f"{res['rss_mb_start']}/{res['rss_mb_peak']}/{res['rss_mb_end']} "
            f"fds start/peak/end="
            f"{res['num_fds_start']}/{res['num_fds_peak']}/{res['num_fds_end']} "
            f"daemon_deaths={res['daemon_deaths']}"
        ),
    ]
    if report["breaches"]:
        lines.append("BREACHES:")
        lines.extend(f"  - {b}" for b in report["breaches"])
        lines.append("RESULT: FAIL")
    else:
        lines.append("RESULT: PASS")
    lines.append("=" * 70)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------


async def _run_async(cfg: StressConfig, stats: StressStats) -> None:
    client = BigtableDataClientAsync(
        project=cfg.project, use_accelerator=cfg.use_accelerator
    )
    async with client, client.get_table(cfg.instance_id, cfg.table_id) as table:
        _assert_mode(table, cfg)
        deadline = time.monotonic() + cfg.duration_seconds
        per_worker_qps = cfg.qps / cfg.workers

        async def worker(worker_id: int) -> None:
            ops = _harness.RandomOps(worker_id, key_prefix=_worker_prefix(worker_id))
            expected_state = _harness.ExpectedState()
            bucket = _harness.TokenBucket(per_worker_qps)
            touched: list[bytes] = []
            i = 0
            while time.monotonic() < deadline:
                wait = bucket.time_until_next()
                if wait > 0:
                    await asyncio.sleep(wait)
                bucket.consume()
                i += 1
                key, mutation = ops.build_mutation(expected_state)
                t0 = time.monotonic()
                try:
                    await table.mutate_row(
                        key, mutation, operation_timeout=cfg.op_timeout
                    )
                    stats.record_op("mutate", time.monotonic() - t0)
                    touched.append(key)
                except Exception as exc:  # noqa: BLE001
                    stats.record_error(exc)
                    continue
                if touched and i % cfg.canary_every == 0:
                    ckey = touched[-1]
                    t0 = time.monotonic()
                    try:
                        row = await table.read_row(
                            ckey, operation_timeout=cfg.op_timeout
                        )
                        stats.record_op("read", time.monotonic() - t0)
                        stats.record_canary(
                            _canary_mismatch(worker_id, ckey, row, expected_state)
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats.record_error(exc)

        async def monitor() -> None:
            introspector = _make_introspector()
            while time.monotonic() < deadline:
                _sample_once(introspector, table, cfg, stats)
                await asyncio.sleep(cfg.sample_interval)

        async def progress() -> None:
            while time.monotonic() < deadline:
                await asyncio.sleep(cfg.progress_interval)
                _log_progress(cfg, stats, deadline)

        tasks = [asyncio.ensure_future(worker(w)) for w in range(cfg.workers)]
        tasks.append(asyncio.ensure_future(monitor()))
        tasks.append(asyncio.ensure_future(progress()))
        await asyncio.gather(*tasks)
        # A final sample so start/end deltas cover the whole run.
        _sample_once(_make_introspector(), table, cfg, stats)


# ---------------------------------------------------------------------------
# Sync driver
# ---------------------------------------------------------------------------


def _run_sync(cfg: StressConfig, stats: StressStats) -> None:
    client = BigtableDataClient(
        project=cfg.project, use_accelerator=cfg.use_accelerator
    )
    with client, client.get_table(cfg.instance_id, cfg.table_id) as table:
        _assert_mode(table, cfg)
        deadline = time.monotonic() + cfg.duration_seconds
        per_worker_qps = cfg.qps / cfg.workers
        stop = threading.Event()

        def worker(worker_id: int) -> None:
            ops = _harness.RandomOps(worker_id, key_prefix=_worker_prefix(worker_id))
            expected_state = _harness.ExpectedState()
            bucket = _harness.TokenBucket(per_worker_qps)
            touched: list[bytes] = []
            i = 0
            while not stop.is_set() and time.monotonic() < deadline:
                wait = bucket.time_until_next()
                if wait > 0:
                    time.sleep(wait)
                bucket.consume()
                i += 1
                key, mutation = ops.build_mutation(expected_state)
                t0 = time.monotonic()
                try:
                    table.mutate_row(key, mutation, operation_timeout=cfg.op_timeout)
                    stats.record_op("mutate", time.monotonic() - t0)
                    touched.append(key)
                except Exception as exc:  # noqa: BLE001
                    stats.record_error(exc)
                    continue
                if touched and i % cfg.canary_every == 0:
                    ckey = touched[-1]
                    t0 = time.monotonic()
                    try:
                        row = table.read_row(ckey, operation_timeout=cfg.op_timeout)
                        stats.record_op("read", time.monotonic() - t0)
                        stats.record_canary(
                            _canary_mismatch(worker_id, ckey, row, expected_state)
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats.record_error(exc)

        def monitor() -> None:
            introspector = _make_introspector()
            while not stop.is_set() and time.monotonic() < deadline:
                _sample_once(introspector, table, cfg, stats)
                stop.wait(cfg.sample_interval)

        def progress() -> None:
            while not stop.is_set() and time.monotonic() < deadline:
                stop.wait(cfg.progress_interval)
                _log_progress(cfg, stats, deadline)

        threads = [
            threading.Thread(target=worker, args=(w,), daemon=True)
            for w in range(cfg.workers)
        ]
        threads.append(threading.Thread(target=monitor, daemon=True))
        threads.append(threading.Thread(target=progress, daemon=True))
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        finally:
            stop.set()
        _sample_once(_make_introspector(), table, cfg, stats)


# ---------------------------------------------------------------------------
# Cross-mode helpers
# ---------------------------------------------------------------------------


def _assert_mode(table, cfg: StressConfig) -> None:
    """Fail fast if we are not on the code path we intend to stress."""
    active = table._accelerator_client is not None
    if cfg.use_accelerator and not active:
        raise RuntimeError(
            "accelerator was requested but the table fell back to native; "
            "refusing to run the stress test on the wrong code path. Check ADC "
            "identity verification and that the bundled daemon can start."
        )
    if not cfg.use_accelerator and active:
        raise RuntimeError("native run requested but accelerator is active")


def _make_introspector():
    try:
        return _harness.ProcessIntrospector()
    except Exception:  # noqa: BLE001 - psutil missing; degrade to no sampling
        return None


def _sample_once(introspector, table, cfg: StressConfig, stats: StressStats) -> None:
    if introspector is None:
        return
    import psutil

    snap = introspector.snapshot()
    try:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except psutil.Error:
        rss_mb = 0.0
    if cfg.use_accelerator:
        pid = _harness.daemon_pid(table)
        daemon_alive = pid is not None and _harness.pid_alive(pid)
    else:
        daemon_alive = True  # not applicable; never counts as a death
    stats.record_sample(
        ResourceSample(
            t=time.monotonic(),
            rss_mb=rss_mb,
            num_fds=snap.num_fds,
            daemon_alive=daemon_alive,
        )
    )


def _log_progress(cfg: StressConfig, stats: StressStats, deadline: float) -> None:
    remaining = max(0.0, deadline - time.monotonic())
    print(
        f"[stress] ops={stats.total_ops} errors={stats.total_errors} "
        f"canary_mismatch={stats.canary_mismatches} "
        f"remaining={remaining / 60:.1f}min",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = _build_config(sys.argv[1:] if argv is None else argv)

    # Guard the environment the same way the pytest suite does, but as hard
    # errors: a soak run silently skipping is worse than failing loudly.
    if os.environ.get("BIGTABLE_EMULATOR_HOST") and cfg.use_accelerator:
        print(
            "error: accelerator is not supported against the emulator; unset "
            "BIGTABLE_EMULATOR_HOST or pass --no-accelerator",
            file=sys.stderr,
        )
        return 2
    if cfg.use_accelerator and _harness.resolve_binary() is None:
        print(
            "error: no accelerator daemon binary available (set "
            f"{_harness.BIN_ENV_VAR} or install a wheel that bundles it)",
            file=sys.stderr,
        )
        return 2

    print(
        f"[stress] starting {'sync' if cfg.sync else 'async'} run: "
        f"accelerator={cfg.use_accelerator} qps={cfg.qps} workers={cfg.workers} "
        f"duration={cfg.duration_seconds:.0f}s",
        file=sys.stderr,
        flush=True,
    )

    stats = StressStats()
    started = time.monotonic()
    try:
        if cfg.sync:
            _run_sync(cfg, stats)
        else:
            asyncio.run(_run_async(cfg, stats))
    except KeyboardInterrupt:
        print("[stress] interrupted; reporting partial results", file=sys.stderr)
    elapsed = time.monotonic() - started

    report = _build_report(cfg, stats, elapsed)
    _print_report(report)
    payload = json.dumps(report, indent=2)
    if cfg.report_path:
        with open(cfg.report_path, "w") as f:
            f.write(payload)
        print(f"[stress] wrote JSON report to {cfg.report_path}", file=sys.stderr)
    else:
        print(payload)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
