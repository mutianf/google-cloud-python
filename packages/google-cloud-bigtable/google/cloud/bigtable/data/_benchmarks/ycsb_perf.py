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
"""A YCSB-style performance driver for the Bigtable ``data`` client.

This is a standalone benchmark (not a pytest module -- nothing here is
collected), modelled on the `Yahoo! Cloud Serving Benchmark
<https://github.com/brianfrankcooper/YCSB>`_ and its Bigtable binding. It ships
inside the wheel and is exposed as the ``bigtable-ycsb`` console script, so it
runs straight from an installed wheel with no source checkout::

    # load 10k records, then run 20k operations of workload A (50/50 read/update)
    bigtable-ycsb --phase load --records 10000
    bigtable-ycsb --phase run --operations 20000 --workload a --threads 16

Equivalently (e.g. before the console script is on ``PATH``) it can be run as a
module::

    python -m google.cloud.bigtable.data._benchmarks.ycsb_perf --phase run

Like YCSB it has two phases:

* **load** — insert ``--records`` rows, each a single Bigtable row holding
  ``--fields`` columns (``field0``..``fieldN-1``) of ``--field-length`` random
  bytes in column family ``--family`` (``cf`` by default, matching the YCSB
  Bigtable binding's default table layout).
* **run** — drive ``--operations`` operations across ``--threads`` closed-loop
  worker threads. Each op is chosen by the workload's proportions
  (read / update / insert / read-modify-write / scan) and its key by the
  request distribution (uniform or zipfian). ``--target`` caps aggregate QPS
  (0 = unlimited / as-fast-as-possible, the YCSB default).

It reports YCSB-style output: overall runtime and throughput, plus per-operation
count and latency percentiles (min / avg / p50 / p95 / p99 / max), and exits
non-zero if any operation errored past ``--error-threshold``.

The workload mixes mirror the YCSB core workloads:

    A  50% read  / 50% update                (update heavy)
    B  95% read  /  5% update                (read mostly)
    C 100% read                              (read only)
    D  95% read  /  5% insert   latest-dist  (read latest)
    F  50% read  / 50% read-modify-write     (read-modify-write)

Config comes from flags, falling back to the standard system-test env vars
(``GOOGLE_CLOUD_PROJECT``, ``BIGTABLE_TEST_INSTANCE``, ``BIGTABLE_TEST_TABLE``,
``BIGTABLE_TEST_APP_PROFILE``). The table must already exist with the requested
column family.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field

from google.cloud.bigtable.data import (
    BigtableDataClient,
    BigtableDataClientAsync,
    ReadRowsQuery,
    SetCell,
)
from google.cloud.bigtable.data import RowRange
from google.cloud.bigtable.data.row_filters import CellsColumnLimitFilter

# Read only the latest cell per column, mirroring what YCSB measures (current
# field values). Without it, reads on a family with no version-GC return the
# full write history, so hot rows bloat and read latency climbs over a run.
_LATEST = CellsColumnLimitFilter(1)


# ---------------------------------------------------------------------------
# Workload definitions (YCSB core workloads a/b/c/d/f)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """A YCSB core workload: op proportions + request distribution."""

    name: str
    read: float
    update: float
    insert: float
    rmw: float  # read-modify-write
    scan: float
    distribution: str  # "uniform" | "zipfian" | "latest"


# Proportions match the canonical YCSB core workload property files.
WORKLOADS: dict[str, Workload] = {
    "a": Workload("a", 0.50, 0.50, 0.0, 0.0, 0.0, "zipfian"),
    "b": Workload("b", 0.95, 0.05, 0.0, 0.0, 0.0, "zipfian"),
    "c": Workload("c", 1.00, 0.00, 0.0, 0.0, 0.0, "zipfian"),
    "d": Workload("d", 0.95, 0.00, 0.05, 0.0, 0.0, "latest"),
    "f": Workload("f", 0.50, 0.00, 0.0, 0.50, 0.0, "zipfian"),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class YCSBConfig:
    project: str | None
    instance_id: str
    table_id: str
    app_profile_id: str | None
    phase: str  # "load" | "run"
    workload: Workload
    record_count: int
    operation_count: int
    max_seconds: float
    field_count: int
    field_length: int
    family: str
    threads: int
    use_async: bool
    use_accelerator: bool
    target_qps: float
    scan_length: int
    op_timeout: float
    error_threshold: float
    seed: int
    report_path: str | None


def _build_config(argv: list[str]) -> YCSBConfig:
    p = argparse.ArgumentParser(
        prog="ycsb_perf",
        description="YCSB-style benchmark for the Bigtable data (V3) client.",
    )
    p.add_argument(
        "--phase",
        choices=["load", "run"],
        default="run",
        help="load rows, or run the workload (default: run)",
    )
    p.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        default="a",
        help="YCSB core workload a/b/c/d/f (default: a)",
    )
    p.add_argument(
        "--records",
        type=int,
        default=10000,
        help="record count: rows loaded (load) and keyspace size (run)",
    )
    p.add_argument(
        "--operations",
        type=int,
        default=10000,
        help="operation count for the run phase",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="wall-clock cap for the run phase (0 = disabled; run --operations)",
    )
    p.add_argument("--fields", type=int, default=10, help="columns per row")
    p.add_argument(
        "--field-length", type=int, default=100, help="bytes per column value"
    )
    p.add_argument(
        "--family", default="cf", help="column family (default: cf, YCSB default)"
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="closed-loop workers: OS threads (sync) or coroutines (--async)",
    )
    p.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="use the async client (1 thread, N coroutines) -- avoids the "
        "sync client's GIL throughput ceiling; best for throughput/latency",
    )
    p.add_argument(
        "--no-accelerator",
        dest="use_accelerator",
        action="store_false",
        help="run over the native client instead of the accelerator daemon",
    )
    p.add_argument(
        "--target",
        type=float,
        default=0.0,
        help="aggregate target QPS cap (0 = unlimited, the default)",
    )
    p.add_argument(
        "--scan-length", type=int, default=100, help="max rows per scan op"
    )
    p.add_argument("--op-timeout", type=float, default=20.0, help="per-op timeout (s)")
    p.add_argument(
        "--error-threshold",
        type=float,
        default=0.01,
        help="fail if error rate exceeds this fraction (default 1%%)",
    )
    p.add_argument("--seed", type=int, default=1234, help="RNG seed")
    p.add_argument("--project", default=None, help="GCP project (default: env)")
    p.add_argument("--instance", default=None, help="instance id (default: env)")
    p.add_argument("--table", default=None, help="table id (default: env)")
    p.add_argument(
        "--app-profile",
        default=None,
        help="app profile id (default: env BIGTABLE_TEST_APP_PROFILE)",
    )
    p.add_argument("--report", default=None, help="write JSON report to this path")
    args = p.parse_args(argv)

    project = (
        args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    )
    instance_id = args.instance or os.getenv("BIGTABLE_TEST_INSTANCE")
    table_id = args.table or os.getenv("BIGTABLE_TEST_TABLE")
    app_profile_id = args.app_profile or os.getenv("BIGTABLE_TEST_APP_PROFILE")
    if not instance_id:
        p.error("no instance id (pass --instance or set BIGTABLE_TEST_INSTANCE)")
    if not table_id:
        p.error("no table id (pass --table or set BIGTABLE_TEST_TABLE)")

    return YCSBConfig(
        project=project,
        instance_id=instance_id,
        table_id=table_id,
        app_profile_id=app_profile_id,
        phase=args.phase,
        workload=WORKLOADS[args.workload],
        record_count=args.records,
        operation_count=args.operations,
        max_seconds=args.max_seconds,
        field_count=args.fields,
        field_length=args.field_length,
        family=args.family,
        threads=args.threads,
        use_async=args.use_async,
        use_accelerator=args.use_accelerator,
        target_qps=args.target,
        scan_length=args.scan_length,
        op_timeout=args.op_timeout,
        error_threshold=args.error_threshold,
        seed=args.seed,
        report_path=args.report,
    )


# ---------------------------------------------------------------------------
# Keys, values, and request distributions
# ---------------------------------------------------------------------------

_KEY_WIDTH = 20  # zero-padded, so lexical order == numeric order for scans


def _key(index: int) -> bytes:
    """YCSB-style deterministic key so load and run address the same space."""
    return f"user{index:0{_KEY_WIDTH}d}".encode()


class ValueGen:
    """Generates random column values of a fixed length, YCSB-style."""

    def __init__(self, field_count: int, field_length: int, rng: random.Random):
        self._field_count = field_count
        self._field_length = field_length
        self._rng = rng

    def fields(self) -> list[tuple[bytes, bytes]]:
        return [
            (f"field{i}".encode(), self._rng.randbytes(self._field_length))
            for i in range(self._field_count)
        ]

    def one_field(self) -> tuple[bytes, bytes]:
        i = self._rng.randrange(self._field_count)
        return (f"field{i}".encode(), self._rng.randbytes(self._field_length))


class InsertCounter:
    """Monotonic key counter shared by all workers.

    Inserted keys are handed out contiguously from ``record_count`` upward, so
    the ``latest`` distribution can see rows added *during* the run -- what YCSB
    workload D requires. Thread-safe for the sync driver; harmless (uncontended)
    under the single-threaded async driver.
    """

    def __init__(self, start: int):
        self._value = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            v = self._value
            self._value += 1
            return v

    def current(self) -> int:
        with self._lock:
            return self._value


class Chooser:
    """Picks a record index under the workload's request distribution."""

    def __init__(
        self,
        record_count: int,
        distribution: str,
        rng: random.Random,
        counter: InsertCounter | None = None,
    ):
        self._n = record_count
        self._dist = distribution
        self._rng = rng
        # For "latest": the live keyspace (loaded rows + rows inserted so far),
        # so reads track newly inserted keys instead of a frozen record_count.
        self._counter = counter
        # Zipfian: O(1) closed-form sampler from Gray et al., "Quickly
        # Generating Billion-Record Synthetic Databases" -- the same math YCSB's
        # ZipfianGenerator uses. Precompute the constants once (the zeta(n)
        # harmonic sum is the only O(n) cost, and it runs here, not per draw, so
        # a read no longer scans the keyspace). YCSB uses the constant exponent
        # theta = 0.99.
        self._theta = 0.99
        self._zipf_ready = distribution == "zipfian" and record_count >= 3
        if self._zipf_ready:
            self._zeta_n = self._zeta(record_count, self._theta)
            zeta_2 = self._zeta(2, self._theta)
            self._alpha = 1.0 / (1.0 - self._theta)
            self._eta = (1.0 - (2.0 / record_count) ** (1.0 - self._theta)) / (
                1.0 - zeta_2 / self._zeta_n
            )

    @staticmethod
    def _zeta(n: int, theta: float) -> float:
        return sum(1.0 / (i**theta) for i in range(1, n + 1))

    def next_index(self) -> int:
        if self._dist == "uniform":
            return self._rng.randrange(self._n)
        if self._dist == "latest":
            # Skew heavily toward the most-recently inserted keys, over the live
            # keyspace so keys inserted during the run are eligible.
            n = self._counter.current() if self._counter is not None else self._n
            return n - 1 - min(n - 1, int(self._rng.expovariate(1 / 50)))
        # zipfian: O(1) inverse sampling against the precomputed constants.
        if not self._zipf_ready:  # keyspace too small for the closed form
            return self._rng.randrange(self._n)
        u = self._rng.random()
        uz = u * self._zeta_n
        if uz < 1.0:
            return 0
        if uz < 1.0 + 0.5**self._theta:
            return 1
        return min(
            self._n - 1, int(self._n * (self._eta * u - self._eta + 1.0) ** self._alpha)
        )


# ---------------------------------------------------------------------------
# Latency measurement (bucketed histogram, YCSB "histogram" measurement type)
# ---------------------------------------------------------------------------


class Measurements:
    """Thread-safe per-operation counters + microsecond-bucketed latencies.

    A bucketed histogram (1us buckets up to 1s, then an overflow bucket) keeps
    memory bounded under millions of ops, unlike storing every raw sample.
    """

    _MAX_US = 1_000_000  # 1s; anything slower lands in the overflow bucket

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hist: dict[str, list[int]] = {}
        self._overflow: dict[str, int] = {}
        self._count: dict[str, int] = {}
        self._total_us: dict[str, int] = {}
        self._min_us: dict[str, int] = {}
        self._max_us: dict[str, int] = {}
        self.errors = 0

    def record(self, op: str, seconds: float) -> None:
        us = int(seconds * 1_000_000)
        with self._lock:
            if op not in self._hist:
                self._hist[op] = [0] * (self._MAX_US + 1)
                self._overflow[op] = 0
                self._count[op] = 0
                self._total_us[op] = 0
                self._min_us[op] = us
                self._max_us[op] = us
            if us > self._MAX_US:
                self._overflow[op] += 1
            else:
                self._hist[op][us] += 1
            self._count[op] += 1
            self._total_us[op] += us
            self._min_us[op] = min(self._min_us[op], us)
            self._max_us[op] = max(self._max_us[op], us)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def _percentile_us(self, op: str, pct: float) -> int:
        total = self._count[op]
        target = pct / 100.0 * total
        seen = 0
        for us, c in enumerate(self._hist[op]):
            seen += c
            if seen >= target:
                return us
        return self._MAX_US  # remainder is in overflow

    def summary(self) -> dict[str, dict]:
        with self._lock:
            out: dict[str, dict] = {}
            for op in sorted(self._count):
                n = self._count[op]
                out[op] = {
                    "count": n,
                    "min_us": self._min_us[op],
                    "avg_us": round(self._total_us[op] / n, 1) if n else 0,
                    "p50_us": self._percentile_us(op, 50),
                    "p95_us": self._percentile_us(op, 95),
                    "p99_us": self._percentile_us(op, 99),
                    "max_us": self._max_us[op],
                    "overflow": self._overflow[op],
                }
            return out


# ---------------------------------------------------------------------------
# Load phase
# ---------------------------------------------------------------------------


def _pick_op(wl: Workload, rng: random.Random) -> str:
    """Choose an operation name under the workload's proportions."""
    r = rng.random()
    for name, prob in (
        ("read", wl.read),
        ("update", wl.update),
        ("insert", wl.insert),
        ("rmw", wl.rmw),
        ("scan", wl.scan),
    ):
        if prob <= 0:
            continue
        if r < prob:
            return name
        r -= prob
    return "read"


def _run_load(cfg: YCSBConfig, table, meas: Measurements) -> None:
    """Insert ``record_count`` rows, sharded across worker threads."""
    per_thread = -(-cfg.record_count // cfg.threads)  # ceil-div
    stop = threading.Event()

    def worker(tid: int) -> None:
        rng = random.Random(cfg.seed + tid)
        vg = ValueGen(cfg.field_count, cfg.field_length, rng)
        start = tid * per_thread
        end = min(cfg.record_count, start + per_thread)
        for idx in range(start, end):
            if stop.is_set():
                return
            mutations = [
                SetCell(cfg.family, q, v) for q, v in vg.fields()
            ]
            t0 = time.monotonic()
            try:
                table.mutate_row(
                    _key(idx), mutations, operation_timeout=cfg.op_timeout
                )
                meas.record("insert", time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                meas.record_error()
                print(f"[load] error on {_key(idx)!r}: {exc!r}", file=sys.stderr)

    _spawn_join(worker, cfg.threads, stop)


# ---------------------------------------------------------------------------
# Run phase
# ---------------------------------------------------------------------------


def _run_workload(cfg: YCSBConfig, table, meas: Measurements) -> None:
    wl = cfg.workload
    per_thread = -(-cfg.operation_count // cfg.threads)
    # Aggregate target QPS -> per-thread inter-op interval (0 == unlimited).
    interval = cfg.threads / cfg.target_qps if cfg.target_qps > 0 else 0.0
    # Wall-clock cap: when set, workers loop until the deadline instead of a
    # fixed op count (per_thread is then treated as an effectively-infinite cap).
    deadline = time.monotonic() + cfg.max_seconds if cfg.max_seconds > 0 else None
    stop = threading.Event()
    # Shared across workers so inserted keys are contiguous and the "latest"
    # distribution sees rows added during the run (YCSB workload D).
    counter = InsertCounter(cfg.record_count)

    def worker(tid: int) -> None:
        rng = random.Random(cfg.seed + 100 + tid)
        vg = ValueGen(cfg.field_count, cfg.field_length, rng)
        chooser = Chooser(cfg.record_count, wl.distribution, rng, counter)
        next_at = time.monotonic()
        limit = float("inf") if deadline is not None else per_thread
        i = 0
        while i < limit:
            i += 1
            if stop.is_set():
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            if interval:
                now = time.monotonic()
                if next_at > now:
                    time.sleep(next_at - now)
                next_at += interval
            op = _pick_op(wl, rng)
            t0 = time.monotonic()
            try:
                if op == "read":
                    table.read_row(
                        _key(chooser.next_index()),
                        row_filter=_LATEST,
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "update":
                    q, v = vg.one_field()
                    table.mutate_row(
                        _key(chooser.next_index()),
                        [SetCell(cfg.family, q, v)],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "insert":
                    idx = counter.next()
                    table.mutate_row(
                        _key(idx),
                        [SetCell(cfg.family, q, v) for q, v in vg.fields()],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "rmw":
                    key = _key(chooser.next_index())
                    table.read_row(
                        key, row_filter=_LATEST, operation_timeout=cfg.op_timeout
                    )
                    q, v = vg.one_field()
                    table.mutate_row(
                        key,
                        [SetCell(cfg.family, q, v)],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "scan":
                    start_idx = chooser.next_index()
                    query = ReadRowsQuery(
                        row_ranges=RowRange(start_key=_key(start_idx)),
                        limit=cfg.scan_length,
                        row_filter=_LATEST,
                    )
                    for _row in table.read_rows_stream(
                        query, operation_timeout=cfg.op_timeout
                    ):
                        pass
                meas.record(op, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                meas.record_error()
                print(f"[run] {op} error: {exc!r}", file=sys.stderr)

    _spawn_join(worker, cfg.threads, stop)


def _spawn_join(worker, n: int, stop: threading.Event) -> None:
    threads = [
        threading.Thread(target=worker, args=(tid,), name=f"ycsb-{tid}")
        for tid in range(n)
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join()
        raise


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------
#
# The sync client is GIL-bound: N OS threads serialize on Python-side per-op
# work (proto (de)serialization, Row construction), so closed-loop throughput
# saturates (~800 qps on the dev VM) and measured latency then inflates with
# concurrency. The async client runs on ONE thread with N cooperative
# coroutines, so I/O overlaps without GIL contention -- the right way to
# measure both throughput and per-op latency at scale. Here ``--threads`` is
# reinterpreted as the number of concurrent worker coroutines.


async def _run_load_async(cfg: YCSBConfig, table, meas: Measurements) -> None:
    """Insert ``record_count`` rows across ``threads`` concurrent coroutines."""
    per_worker = -(-cfg.record_count // cfg.threads)  # ceil-div

    async def worker(wid: int) -> None:
        rng = random.Random(cfg.seed + wid)
        vg = ValueGen(cfg.field_count, cfg.field_length, rng)
        start = wid * per_worker
        end = min(cfg.record_count, start + per_worker)
        for idx in range(start, end):
            mutations = [SetCell(cfg.family, q, v) for q, v in vg.fields()]
            t0 = time.monotonic()
            try:
                await table.mutate_row(
                    _key(idx), mutations, operation_timeout=cfg.op_timeout
                )
                meas.record("insert", time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                meas.record_error()
                print(f"[load] error on {_key(idx)!r}: {exc!r}", file=sys.stderr)

    await asyncio.gather(*(worker(w) for w in range(cfg.threads)))


async def _run_workload_async(cfg: YCSBConfig, table, meas: Measurements) -> None:
    wl = cfg.workload
    per_worker = -(-cfg.operation_count // cfg.threads)
    interval = cfg.threads / cfg.target_qps if cfg.target_qps > 0 else 0.0
    deadline = time.monotonic() + cfg.max_seconds if cfg.max_seconds > 0 else None
    # Shared across coroutines so inserted keys are contiguous and the "latest"
    # distribution sees rows added during the run (YCSB workload D).
    counter = InsertCounter(cfg.record_count)

    async def worker(wid: int) -> None:
        rng = random.Random(cfg.seed + 100 + wid)
        vg = ValueGen(cfg.field_count, cfg.field_length, rng)
        chooser = Chooser(cfg.record_count, wl.distribution, rng, counter)
        next_at = time.monotonic()
        limit = float("inf") if deadline is not None else per_worker
        i = 0
        while i < limit:
            i += 1
            if deadline is not None and time.monotonic() >= deadline:
                return
            if interval:
                now = time.monotonic()
                if next_at > now:
                    await asyncio.sleep(next_at - now)
                next_at += interval
            op = _pick_op(wl, rng)
            t0 = time.monotonic()
            try:
                if op == "read":
                    await table.read_row(
                        _key(chooser.next_index()),
                        row_filter=_LATEST,
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "update":
                    q, v = vg.one_field()
                    await table.mutate_row(
                        _key(chooser.next_index()),
                        [SetCell(cfg.family, q, v)],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "insert":
                    idx = counter.next()
                    await table.mutate_row(
                        _key(idx),
                        [SetCell(cfg.family, q, v) for q, v in vg.fields()],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "rmw":
                    key = _key(chooser.next_index())
                    await table.read_row(
                        key, row_filter=_LATEST, operation_timeout=cfg.op_timeout
                    )
                    q, v = vg.one_field()
                    await table.mutate_row(
                        key,
                        [SetCell(cfg.family, q, v)],
                        operation_timeout=cfg.op_timeout,
                    )
                elif op == "scan":
                    start_idx = chooser.next_index()
                    query = ReadRowsQuery(
                        row_ranges=RowRange(start_key=_key(start_idx)),
                        limit=cfg.scan_length,
                        row_filter=_LATEST,
                    )
                    async for _row in await table.read_rows_stream(
                        query, operation_timeout=cfg.op_timeout
                    ):
                        pass
                meas.record(op, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                meas.record_error()
                print(f"[run] {op} error: {exc!r}", file=sys.stderr)

    await asyncio.gather(*(worker(w) for w in range(cfg.threads)))


async def _amain(cfg: YCSBConfig, meas: Measurements) -> float:
    """Async entrypoint: open the client, run the phase, return elapsed seconds."""
    client = BigtableDataClientAsync(
        project=cfg.project, use_accelerator=cfg.use_accelerator
    )
    async with client, client.get_table(
        cfg.instance_id, cfg.table_id, app_profile_id=cfg.app_profile_id
    ) as table:
        t0 = time.monotonic()
        if cfg.phase == "load":
            await _run_load_async(cfg, table, meas)
        else:
            await _run_workload_async(cfg, table, meas)
        return time.monotonic() - t0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(cfg: YCSBConfig, meas: Measurements, elapsed: float) -> dict:
    per_op = meas.summary()
    total_ops = sum(v["count"] for v in per_op.values())
    throughput = total_ops / elapsed if elapsed > 0 else 0.0
    report = {
        "phase": cfg.phase,
        "workload": cfg.workload.name,
        "instance": cfg.instance_id,
        "table": cfg.table_id,
        "app_profile": cfg.app_profile_id,
        "mode": "async" if cfg.use_async else "sync",
        "accelerator": cfg.use_accelerator,
        "threads": cfg.threads,
        "target_qps": cfg.target_qps,
        "record_count": cfg.record_count,
        "runtime_s": round(elapsed, 3),
        "total_ops": total_ops,
        "throughput_ops_s": round(throughput, 1),
        "errors": meas.errors,
        "operations": per_op,
    }

    print("\n" + "=" * 66)
    print(
        f"[OVERALL] phase={cfg.phase} workload={cfg.workload.name} "
        f"mode={'async' if cfg.use_async else 'sync'} "
        f"accel={cfg.use_accelerator} "
        f"{'coroutines' if cfg.use_async else 'threads'}={cfg.threads}"
    )
    print(f"[OVERALL] RunTime(s)      {elapsed:10.3f}")
    print(f"[OVERALL] Throughput(ops/s){throughput:9.1f}")
    print(f"[OVERALL] Operations      {total_ops:10d}")
    print(f"[OVERALL] Errors          {meas.errors:10d}")
    for op, s in per_op.items():
        print(
            f"[{op.upper():<7}] count={s['count']:<8d} "
            f"min={s['min_us']/1000:7.2f}ms avg={s['avg_us']/1000:7.2f}ms "
            f"p50={s['p50_us']/1000:7.2f}ms p95={s['p95_us']/1000:7.2f}ms "
            f"p99={s['p99_us']/1000:7.2f}ms max={s['max_us']/1000:7.2f}ms"
        )
    print("=" * 66)

    if cfg.report_path:
        with open(cfg.report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote JSON report to {cfg.report_path}")
    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg = _build_config(sys.argv[1:] if argv is None else argv)
    print(
        f"connecting: project={cfg.project} instance={cfg.instance_id} "
        f"table={cfg.table_id} app_profile={cfg.app_profile_id}"
    )
    meas = Measurements()
    unit = "coroutines" if cfg.use_async else "threads"
    if cfg.phase == "load":
        print(
            f"loading {cfg.record_count} records over {cfg.threads} {unit} "
            f"({'async' if cfg.use_async else 'sync'})..."
        )
    else:
        budget = (
            f"{cfg.max_seconds:g}s"
            if cfg.max_seconds > 0
            else f"{cfg.operation_count} ops"
        )
        print(
            f"running workload {cfg.workload.name} for {budget} over "
            f"{cfg.threads} {unit} "
            f"({'async' if cfg.use_async else 'sync'}, "
            f"target={cfg.target_qps or 'unlimited'} qps)..."
        )

    if cfg.use_async:
        elapsed = asyncio.run(_amain(cfg, meas))
    else:
        client = BigtableDataClient(
            project=cfg.project, use_accelerator=cfg.use_accelerator
        )
        with client, client.get_table(
            cfg.instance_id, cfg.table_id, app_profile_id=cfg.app_profile_id
        ) as table:
            t0 = time.monotonic()
            if cfg.phase == "load":
                _run_load(cfg, table, meas)
            else:
                _run_workload(cfg, table, meas)
            elapsed = time.monotonic() - t0

    report = _report(cfg, meas, elapsed)
    total = report["total_ops"] + meas.errors
    if total and meas.errors / total > cfg.error_threshold:
        print(
            f"FAIL: error rate {meas.errors}/{total} exceeds "
            f"{cfg.error_threshold:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
