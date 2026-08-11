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

    python -m tests.system.data.accelerator.stress --hours 6

It drives the *real shipped path* end-to-end -- a real ``BigtableDataClient``
(async by default, or the generated sync client with ``--sync``) with
``use_accelerator`` on, which spawns the real ``AcceleratorDaemon`` over the real
bundled binary and routes ``read_row``/``mutate_row`` over the UDS. Nothing is
faked; ``--no-accelerator`` runs the identical load over the native path so the
two can be compared.

While it runs it drives load closed-loop across N workers (as fast as the path
allows, matching the YCSB driver's default), each owning a disjoint key prefix
and its own ``ExpectedState``. An optional ``--qps`` cap paces the workers with a
token bucket when a bounded rate is wanted instead. Every mutation updates
the expected state and Bigtable; periodic canary reads assert Bigtable still
matches the expected state (correctness under sustained load). A monitor samples
RSS / open FDs and
checks daemon liveness on an interval, so resource leaks or a dead daemon are
caught. At the end it writes a JSON + text report and exits non-zero if any
breach threshold was crossed (error rate, canary drift, daemon death, or RSS
growth), so it is usable as a CI gate.

Config comes from the standard system-test env vars
(``GOOGLE_CLOUD_PROJECT``, ``BIGTABLE_TEST_INSTANCE``, ``BIGTABLE_TEST_TABLE``),
overridable by flags. The instance must already exist. The table is managed for
you: when no ``--table``/``BIGTABLE_TEST_TABLE`` is given the driver creates a
fresh, uniquely-named table (with the families ``test-family`` and
``test-family-2``) before the run and deletes it afterwards; when a table id *is*
supplied it is reused as-is and left in place (never deleted). ``--keep-table``
suppresses deletion of a self-created table for post-mortem inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
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
    app_profile_id: str | None
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
    # Diagnostics for the RSS-growth investigation.
    tracemalloc: bool
    # Table lifecycle: when no table id is supplied the driver creates a fresh
    # uniquely-named table and (unless keep_table) deletes it on exit; a supplied
    # table id is reused as-is and never touched.
    manage_table: bool
    keep_table: bool


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
    p.add_argument(
        "--qps",
        type=float,
        default=0.0,
        help="aggregate target operations/sec cap (0 = unlimited / closed-loop, "
        "the default: drive as fast as the path allows)",
    )
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
    p.add_argument(
        "--table",
        default=None,
        help="table id (default: env, else a fresh table is created and deleted)",
    )
    p.add_argument(
        "--keep-table",
        action="store_true",
        help="do not delete a self-created table on exit (for post-mortem)",
    )
    p.add_argument(
        "--app-profile",
        default=None,
        help="app profile id (default: env BIGTABLE_TEST_APP_PROFILE, else none)",
    )
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
        default=768.0,
        # Generic async-grpc/protobuf working set climbs to a high-water mark
        # over a multi-hour soak (identical native vs accelerator, ~160 bytes/op
        # of untracked C-level memory that plateaus, not an unbounded leak), so
        # the ceiling must clear that band while still catching a real leak.
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
    p.add_argument(
        "--tracemalloc",
        action="store_true",
        help="trace Python allocations; dump top allocators + traced-heap curve "
        "(for the RSS-growth investigation; adds per-alloc overhead)",
    )
    args = p.parse_args(argv)

    duration = args.seconds if args.seconds is not None else args.hours * 3600.0
    if duration <= 0:
        p.error("duration must be positive")
    if args.qps < 0:
        p.error("--qps must be non-negative (0 = unlimited)")
    if args.workers <= 0:
        p.error("--workers must be positive")

    project = (
        args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    )
    instance_id = args.instance or os.getenv("BIGTABLE_TEST_INSTANCE")
    if not instance_id:
        p.error("no instance id (pass --instance or set BIGTABLE_TEST_INSTANCE)")

    # A supplied table is reused (and left in place); otherwise the driver owns a
    # fresh, uniquely-named table for the run and cleans it up on exit.
    table_id = args.table or os.getenv("BIGTABLE_TEST_TABLE")
    manage_table = table_id is None
    if manage_table:
        table_id = f"accel-stress-{uuid.uuid4().hex[:12]}"

    app_profile_id = args.app_profile or os.getenv("BIGTABLE_TEST_APP_PROFILE")

    return StressConfig(
        project=project,
        instance_id=instance_id,
        table_id=table_id,
        app_profile_id=app_profile_id,
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
        tracemalloc=args.tracemalloc,
        manage_table=manage_table,
        keep_table=args.keep_table,
    )


# ---------------------------------------------------------------------------
# Table lifecycle (self-managed when no table id is supplied)
# ---------------------------------------------------------------------------


def _admin_client(cfg: StressConfig):
    """A Table Admin client, sharing the same project resolution as the run."""
    from google.cloud.bigtable.client import Client

    return Client(admin=True, project=cfg.project)


def _create_stress_table(cfg: StressConfig) -> None:
    """Create the run's table with the two families the load generator uses.

    Idempotent: an already-existing table is reused rather than treated as an
    error, so a ``--keep-table`` run can be re-pointed at the same id.
    """
    from google.api_core import exceptions

    from google.cloud.bigtable_admin_v2 import types

    client = _admin_client(cfg)
    parent = f"projects/{client.project}/instances/{cfg.instance_id}"
    families = {
        _harness.TEST_FAMILY: types.ColumnFamily(),
        _harness.TEST_FAMILY_2: types.ColumnFamily(),
    }
    print(f"[stress] creating table {parent}/tables/{cfg.table_id}", file=sys.stderr)
    try:
        client.table_admin_client.create_table(
            request={
                "parent": parent,
                "table_id": cfg.table_id,
                "table": {"column_families": families},
            }
        )
    except exceptions.AlreadyExists:
        print("[stress] table already exists; reusing", file=sys.stderr)


def _delete_stress_table(cfg: StressConfig) -> None:
    """Best-effort delete of a self-created table; never masks the run result."""
    client = _admin_client(cfg)
    name = (
        f"projects/{client.project}/instances/{cfg.instance_id}/tables/{cfg.table_id}"
    )
    print(f"[stress] deleting table {name}", file=sys.stderr)
    try:
        client.table_admin_client.delete_table(name=name)
    except Exception as exc:  # noqa: BLE001 - cleanup must not raise over the report
        print(f"[stress] failed to delete table {name}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shared, thread-safe result accumulator
# ---------------------------------------------------------------------------


@dataclass
class ResourceSample:
    t: float
    rss_mb: float
    num_fds: int
    daemon_alive: bool
    # Go daemon RSS (accelerator runs only; 0.0 when native or unavailable).
    daemon_rss_mb: float = 0.0
    # Python-side tracemalloc traced heap (only populated with --tracemalloc).
    traced_mb: float = 0.0


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
        # A post-warmup tracemalloc snapshot, diffed against the final snapshot
        # to localize what grows over the run (set only with --tracemalloc).
        self.tm_baseline = None

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
    """Return a human-readable mismatch string, or None if the row matches.

    On a mismatch the string also carries the recent per-key mutation history
    (bounded; see ``_harness.OP_HISTORY_PER_KEY``) so a rare divergence is
    root-causable from the report without a reproduction run.
    """
    got = _harness.normalize_row(row)
    expected = expected_state.expected_cells(key)
    if got == expected:
        return None
    history = expected_state.history(key)
    history_str = "\n    ".join(history) if history else "(no recorded history)"
    return (
        f"worker {worker_id} key {key!r}: got {got!r} expected {expected!r}\n"
        f"  op history for {key!r} (oldest first):\n    {history_str}"
    )


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
    daemon_rss_values = [s.daemon_rss_mb for s in stats.samples]
    traced_values = [s.traced_mb for s in stats.samples]
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
            # Go daemon RSS (accelerator runs only) so a Python-driver breach can
            # be separated from daemon-side growth.
            "daemon_rss_mb_start": daemon_rss_values[0] if daemon_rss_values else None,
            "daemon_rss_mb_peak": (
                max(daemon_rss_values) if daemon_rss_values else None
            ),
            "daemon_rss_mb_end": daemon_rss_values[-1] if daemon_rss_values else None,
            # Python tracemalloc traced heap (only with --tracemalloc).
            "traced_mb_start": traced_values[0] if traced_values else None,
            "traced_mb_peak": max(traced_values) if traced_values else None,
            "traced_mb_end": traced_values[-1] if traced_values else None,
        },
        "breaches": breaches,
        "passed": not breaches,
    }


def _timeseries_csv_path(report_path: str) -> str:
    """Sidecar CSV path next to the JSON report (``x.json`` -> ``x.timeseries.csv``)."""
    if report_path.endswith(".json"):
        return report_path[: -len(".json")] + ".timeseries.csv"
    return report_path + ".timeseries.csv"


def _write_timeseries_csv(path: str, stats: StressStats) -> None:
    """Persist the per-sample resource curve.

    The JSON report keeps only start/peak/end aggregates, which cannot
    distinguish an unbounded leak from a plateau. This dumps every sample so the
    RSS-over-time shape (Python driver vs. Go daemon) is inspectable.
    """
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["t_rel_s", "rss_mb", "daemon_rss_mb", "traced_mb", "num_fds", "daemon_alive"]
        )
        t0 = stats.samples[0].t if stats.samples else 0.0
        for s in stats.samples:
            w.writerow(
                [
                    f"{s.t - t0:.1f}",
                    f"{s.rss_mb:.2f}",
                    f"{s.daemon_rss_mb:.2f}",
                    f"{s.traced_mb:.2f}",
                    s.num_fds,
                    int(s.daemon_alive),
                ]
            )


def _tracemalloc_top(limit: int = 25) -> list[str]:
    """Top allocators by traced size, one human-readable line each."""
    import tracemalloc

    if not tracemalloc.is_tracing():
        return []
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")[:limit]
    lines = []
    for stat in stats:
        frame = stat.traceback[0]
        lines.append(
            f"{frame.filename}:{frame.lineno} "
            f"size={stat.size / (1024 * 1024):.2f}MB count={stat.count}"
        )
    return lines


def _tracemalloc_growth(baseline, limit: int = 25) -> list[str]:
    """Top allocation sites by *growth* since ``baseline``.

    A size-diff surfaces a slow leak that a plain top-by-size buries under large
    but steady-state allocations.
    """
    import tracemalloc

    if baseline is None or not tracemalloc.is_tracing():
        return []
    snapshot = tracemalloc.take_snapshot()
    diff = snapshot.compare_to(baseline, "lineno")[:limit]
    lines = []
    for stat in diff:
        frame = stat.traceback[0]
        lines.append(
            f"{frame.filename}:{frame.lineno} "
            f"+{stat.size_diff / (1024 * 1024):.2f}MB "
            f"(now {stat.size / (1024 * 1024):.2f}MB) "
            f"count_diff={stat.count_diff:+d}"
        )
    return lines


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
            f"workers={cfg['workers']} target_qps={cfg['target_qps'] or 'unlimited'}"
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
        (
            f"daemon_rss_mb start/peak/end="
            f"{res['daemon_rss_mb_start']}/{res['daemon_rss_mb_peak']}/"
            f"{res['daemon_rss_mb_end']} "
            f"traced_mb start/peak/end="
            f"{res['traced_mb_start']}/{res['traced_mb_peak']}/{res['traced_mb_end']}"
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
    async with (
        client,
        client.get_table(
            cfg.instance_id, cfg.table_id, app_profile_id=cfg.app_profile_id
        ) as table,
    ):
        _assert_mode(table, cfg)
        deadline = time.monotonic() + cfg.duration_seconds
        # 0 == unlimited: workers run closed-loop, as fast as the path allows
        # (the default, matching the YCSB driver). A positive --qps caps the
        # aggregate rate via a per-worker token bucket.
        per_worker_qps = cfg.qps / cfg.workers if cfg.qps > 0 else 0.0

        async def worker(worker_id: int) -> None:
            ops = _harness.RandomOps(worker_id, key_prefix=_worker_prefix(worker_id))
            expected_state = _harness.ExpectedState()
            bucket = _harness.TokenBucket(per_worker_qps) if per_worker_qps else None
            # Only the most-recently written key is ever re-read for the canary,
            # so track a single key rather than accumulating every key touched
            # (an unbounded list would dominate driver RSS on a multi-hour soak).
            last_key: bytes | None = None
            i = 0
            while time.monotonic() < deadline:
                if bucket is not None:
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
                    last_key = key
                except Exception as exc:  # noqa: BLE001
                    stats.record_error(exc)
                    continue
                if last_key is not None and i % cfg.canary_every == 0:
                    ckey = last_key
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
    with (
        client,
        client.get_table(
            cfg.instance_id, cfg.table_id, app_profile_id=cfg.app_profile_id
        ) as table,
    ):
        _assert_mode(table, cfg)
        deadline = time.monotonic() + cfg.duration_seconds
        # 0 == unlimited: workers run closed-loop, as fast as the path allows
        # (the default, matching the YCSB driver). A positive --qps caps the
        # aggregate rate via a per-worker token bucket.
        per_worker_qps = cfg.qps / cfg.workers if cfg.qps > 0 else 0.0
        stop = threading.Event()

        def worker(worker_id: int) -> None:
            ops = _harness.RandomOps(worker_id, key_prefix=_worker_prefix(worker_id))
            expected_state = _harness.ExpectedState()
            bucket = _harness.TokenBucket(per_worker_qps) if per_worker_qps else None
            # Only the most-recently written key is ever re-read for the canary,
            # so track a single key rather than accumulating every key touched
            # (an unbounded list would dominate driver RSS on a multi-hour soak).
            last_key: bytes | None = None
            i = 0
            while not stop.is_set() and time.monotonic() < deadline:
                if bucket is not None:
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
                    last_key = key
                except Exception as exc:  # noqa: BLE001
                    stats.record_error(exc)
                    continue
                if last_key is not None and i % cfg.canary_every == 0:
                    ckey = last_key
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
    daemon_rss_mb = 0.0
    if cfg.use_accelerator:
        pid = _harness.daemon_pid(table)
        daemon_alive = pid is not None and _harness.pid_alive(pid)
        if pid is not None:
            # Sample the Go daemon's RSS too, so a growth breach can be
            # attributed to the Python driver vs. the daemon process.
            try:
                daemon_rss_mb = psutil.Process(pid).memory_info().rss / (1024 * 1024)
            except psutil.Error:
                daemon_rss_mb = 0.0
    else:
        daemon_alive = True  # not applicable; never counts as a death
    traced_mb = 0.0
    if cfg.tracemalloc:
        import tracemalloc

        if tracemalloc.is_tracing():
            traced_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
            # Capture a baseline after a few warmup samples (past one-time import
            # and pool-fill allocations) so the end-of-run diff isolates growth.
            if stats.tm_baseline is None and len(stats.samples) >= 3:
                stats.tm_baseline = tracemalloc.take_snapshot()
    stats.record_sample(
        ResourceSample(
            t=time.monotonic(),
            rss_mb=rss_mb,
            num_fds=snap.num_fds,
            daemon_alive=daemon_alive,
            daemon_rss_mb=daemon_rss_mb,
            traced_mb=traced_mb,
        )
    )


def _log_progress(cfg: StressConfig, stats: StressStats, deadline: float) -> None:
    remaining = max(0.0, deadline - time.monotonic())
    # Surface the latest resource sample so a long detached soak is observable
    # live (the per-sample CSV is only written when the run completes).
    with stats._lock:
        latest = stats.samples[-1] if stats.samples else None
    rss = f" rss={latest.rss_mb:.0f}MB daemon_rss={latest.daemon_rss_mb:.0f}MB" if latest else ""
    print(
        f"[stress] ops={stats.total_ops} errors={stats.total_errors} "
        f"canary_mismatch={stats.canary_mismatches}{rss} "
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
        f"accelerator={cfg.use_accelerator} qps={cfg.qps or 'unlimited'} "
        f"workers={cfg.workers} "
        f"duration={cfg.duration_seconds:.0f}s table={cfg.table_id}"
        f"{' (managed)' if cfg.manage_table else ' (reused)'}"
        f" app_profile={cfg.app_profile_id or '(default)'}",
        file=sys.stderr,
        flush=True,
    )

    # Provision the table before spinning up the client so the daemon's first
    # RPC lands on a table that exists.
    if cfg.manage_table:
        _create_stress_table(cfg)

    if cfg.tracemalloc:
        import tracemalloc

        tracemalloc.start(25)
        print("[stress] tracemalloc enabled (25 frames)", file=sys.stderr)

    stats = StressStats()
    started = time.monotonic()
    try:
        try:
            if cfg.sync:
                _run_sync(cfg, stats)
            else:
                asyncio.run(_run_async(cfg, stats))
        except KeyboardInterrupt:
            print("[stress] interrupted; reporting partial results", file=sys.stderr)
        elapsed = time.monotonic() - started

        report = _build_report(cfg, stats, elapsed)
        if cfg.tracemalloc:
            growth = _tracemalloc_growth(stats.tm_baseline)
            top = _tracemalloc_top()
            report["resources"]["tracemalloc_growth"] = growth
            report["resources"]["tracemalloc_top"] = top
            print("[stress] tracemalloc growth since warmup (leak localizer):", file=sys.stderr)
            for line in growth:
                print(f"  {line}", file=sys.stderr)
            print("[stress] tracemalloc top allocators (absolute):", file=sys.stderr)
            for line in top:
                print(f"  {line}", file=sys.stderr)
        _print_report(report)
        payload = json.dumps(report, indent=2)
        if cfg.report_path:
            with open(cfg.report_path, "w") as f:
                f.write(payload)
            print(f"[stress] wrote JSON report to {cfg.report_path}", file=sys.stderr)
            if stats.samples:
                csv_path = _timeseries_csv_path(cfg.report_path)
                _write_timeseries_csv(csv_path, stats)
                print(f"[stress] wrote resource time-series to {csv_path}", file=sys.stderr)
        else:
            print(payload)
    finally:
        # Always reclaim a self-created table, even if the run raised.
        if cfg.manage_table and not cfg.keep_table:
            _delete_stress_table(cfg)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
