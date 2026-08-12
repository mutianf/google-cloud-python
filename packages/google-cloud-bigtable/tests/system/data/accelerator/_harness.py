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
"""Shared, dependency-light helpers for the accelerator pre-release suite.

This module is intentionally free of async/await and of any Bigtable network IO
so it can be shared verbatim by the async tests, the CrossSync-generated sync
tests, and the standalone stress driver. The pieces are:

* ``ExpectedState`` — an in-memory model of expected cell state. Every mutation
  is applied to the expected state *and* to Bigtable; reads are asserted against
  the expected state and cross-checked against the native (non-accelerated) path.
* row/cell normalization + ``assert_rows_equivalent`` for differential checks.
* ``resolve_binary_or_skip`` / ``require_real_bigtable`` gating helpers.
* controlled-binary factories that drive the real daemon's failure/race paths.
* ``ProcessIntrospector`` + ``LeakSnapshot`` (psutil) for subprocess/FD/tempdir
  leak detection.
* ``TokenBucket`` and ``LatencyStats`` load-driver primitives (used by the
  concurrency tests and the stress driver).
* ``RandomOps`` — a seeded generator of mutations/reads for stress + concurrency.
"""

from __future__ import annotations

import glob
import os
import stat
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from google.cloud.bigtable.data import (
    AddToCell,
    DeleteAllFromFamily,
    DeleteAllFromRow,
    DeleteRangeFromColumn,
    ReadRowsQuery,
    SetCell,
)
from google.cloud.bigtable.data.row import Row

# ---------------------------------------------------------------------------
# Constants shared with the fixtures / tests.
# ---------------------------------------------------------------------------

# Test tables are created with millisecond timestamp granularity (the Bigtable
# default), so every explicit timestamp must be a whole number of milliseconds.
# The server rejects finer-grained timestamps with InvalidArgument.
MS = 1000

# Column families created by the shared system-test conftest.
TEST_FAMILY = "test-family"
TEST_FAMILY_2 = "test-family-2"
# An int64 "sum" aggregate family (see ``column_family_config`` in the system
# conftest). Only ``AddToCell`` may write to it; ``SetCell`` is server-rejected.
# Cells read back as 8-byte big-endian signed int64 of the accumulated sum.
TEST_AGGREGATE_FAMILY = "test-aggregate-family"
# Width of an aggregate cell's value on read (int64, big-endian, signed).
_AGGREGATE_VALUE_BYTES = 8

# Env var honored by the real daemon wrapper (google/.../_accelerator/_daemon.py)
# to override the binary location. We use it to point the *real* AcceleratorDaemon
# at a controlled binary for fault injection.
BIN_ENV_VAR = "BIGTABLE_ACCELERATOR_BIN"


# ---------------------------------------------------------------------------
# Expected-state model
# ---------------------------------------------------------------------------

# A single expected cell, in the shape Bigtable returns them.
Expected = tuple  # (family: str, qualifier: bytes, timestamp_micros: int, value: bytes)


class ExpectedState:
    """A minimal, deterministic model of Bigtable cell state.

    Only the semantics the accelerator routes through (``mutate_row`` +
    ``read_row``) are modelled: set-cell, add-to-cell (int64 ``sum`` aggregate),
    and the three delete flavors, with explicit millisecond-granular timestamps
    so the model is exact. Server-side timestamps are intentionally out of scope
    here (they are non-deterministic; tests that use them assert weaker
    invariants directly).

    Storage: ``{row_key: {(family, qualifier): {timestamp_micros: value}}}``. For
    ordinary families ``value`` is ``bytes``; for ``TEST_AGGREGATE_FAMILY`` it is
    the accumulated ``int`` sum, encoded to 8-byte big-endian at read time so it
    matches what Bigtable returns for an int64 aggregate cell.
    """

    def __init__(self) -> None:
        self._rows: dict[bytes, dict[tuple[str, bytes], dict[int, bytes]]] = {}

    # -- mutation builders: update the model and return the real Mutation ----

    def set_cell(
        self, row_key: bytes, family: str, qualifier: bytes, value: bytes, ts: int
    ) -> SetCell:
        if ts % MS != 0:
            raise ValueError(
                "timestamps must be millisecond-granular (multiple of 1000)"
            )
        col = self._rows.setdefault(row_key, {}).setdefault((family, qualifier), {})
        col[ts] = value
        return SetCell(family, qualifier, value, timestamp_micros=ts)

    def add_to_cell(
        self, row_key: bytes, qualifier: bytes, delta: int, ts: int
    ) -> AddToCell:
        """Accumulate ``delta`` into an int64 ``sum`` aggregate cell.

        Aggregate cells live only in ``TEST_AGGREGATE_FAMILY``. Repeated adds at
        the same (qualifier, timestamp) sum server-side — this mutation is *not*
        idempotent, which is exactly why the model tracks the running total
        rather than the last write. The stored value is an ``int``; it is encoded
        to big-endian bytes in ``expected_cells``.
        """
        if ts % MS != 0:
            raise ValueError(
                "timestamps must be millisecond-granular (multiple of 1000)"
            )
        col = self._rows.setdefault(row_key, {}).setdefault(
            (TEST_AGGREGATE_FAMILY, qualifier), {}
        )
        col[ts] = col.get(ts, 0) + delta
        return AddToCell(
            TEST_AGGREGATE_FAMILY, qualifier, delta, timestamp_micros=ts
        )

    def delete_range_from_column(
        self,
        row_key: bytes,
        family: str,
        qualifier: bytes,
        start: int | None = None,
        end: int | None = None,
    ) -> DeleteRangeFromColumn:
        """Delete cells with start <= ts < end (start=None→0, end=None→inf)."""
        col = self._rows.get(row_key, {}).get((family, qualifier))
        if col is not None:
            lo = 0 if start is None else start
            for ts in list(col):
                if ts >= lo and (end is None or ts < end):
                    del col[ts]
            if not col:
                self._rows[row_key].pop((family, qualifier), None)
        return DeleteRangeFromColumn(family, qualifier, start, end)

    def delete_from_family(self, row_key: bytes, family: str) -> DeleteAllFromFamily:
        row = self._rows.get(row_key)
        if row is not None:
            for key in [k for k in row if k[0] == family]:
                del row[key]
        return DeleteAllFromFamily(family)

    def delete_from_row(self, row_key: bytes) -> DeleteAllFromRow:
        self._rows.pop(row_key, None)
        return DeleteAllFromRow()

    # -- expectation -------------------------------------------------------

    def expected_cells(self, row_key: bytes) -> list[Expected]:
        """Return expected cells for a row in Bigtable read order.

        Order: family asc, qualifier asc, timestamp desc — matching the order
        the client yields cells (see ``Cell.__lt__``).
        """
        row = self._rows.get(row_key, {})
        out: list[Expected] = []
        for family, qualifier in sorted(row, key=lambda k: (k[0], k[1])):
            for ts in sorted(row[(family, qualifier)], reverse=True):
                value = row[(family, qualifier)][ts]
                if family == TEST_AGGREGATE_FAMILY:
                    # Bigtable returns an int64 aggregate cell as 8 big-endian,
                    # signed bytes; mirror that so the differential check matches.
                    value = int(value).to_bytes(
                        _AGGREGATE_VALUE_BYTES, "big", signed=True
                    )
                out.append((family, qualifier, ts, value))
        return out

    def row_is_empty(self, row_key: bytes) -> bool:
        return not self._rows.get(row_key)

    def keys(self) -> Iterable[bytes]:
        return list(self._rows)


# ---------------------------------------------------------------------------
# Row / cell normalization for differential comparison
# ---------------------------------------------------------------------------


def normalize_row(row: Row | None) -> list[Expected]:
    """Flatten a ``Row`` into comparable (family, qualifier, ts, value) tuples."""
    if row is None:
        return []
    return [
        (cell.family, cell.qualifier, cell.timestamp_micros, cell.value)
        for cell in row.cells
    ]


def _fmt(cells: Sequence[Expected]) -> str:
    return (
        "\n".join(f"  {fam}:{qual!r}@{ts} = {val!r}" for (fam, qual, ts, val) in cells)
        or "  <empty>"
    )


def assert_rows_equivalent(
    label_a: str, cells_a: Sequence[Expected], label_b: str, cells_b: Sequence[Expected]
) -> None:
    """Assert two normalized rows match, with a readable diff on failure."""
    if list(cells_a) != list(cells_b):
        raise AssertionError(
            f"row mismatch between {label_a} and {label_b}:\n"
            f"{label_a}:\n{_fmt(cells_a)}\n{label_b}:\n{_fmt(cells_b)}"
        )


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


def resolve_binary() -> str | None:
    """Return the accelerator binary path if one is available, else None.

    Honors ``BIGTABLE_ACCELERATOR_BIN`` first, then the bundled binary shipped in
    the package. Mirrors the daemon wrapper's own resolution so tests skip (not
    fail) on platforms where no binary is bundled.
    """
    override = os.environ.get(BIN_ENV_VAR)
    if override:
        return override if os.path.isfile(override) else None
    from google.cloud.bigtable.data._accelerator import _daemon

    return _daemon._default_binary_path()


def require_binary_or_skip() -> str:
    import pytest

    path = resolve_binary()
    if path is None:
        pytest.skip(
            "No accelerator daemon binary available "
            f"(set {BIN_ENV_VAR} or install a wheel that bundles it)."
        )
    return path


def require_real_bigtable_or_skip() -> None:
    """Skip when running without real Bigtable credentials/target.

    The accelerator refuses to run against the emulator, so these tests need a
    real instance. We treat the presence of the emulator env var as an explicit
    "no real backend" signal.
    """
    import pytest

    from google.cloud.environment_vars import BIGTABLE_EMULATOR

    if os.environ.get(BIGTABLE_EMULATOR):
        pytest.skip("accelerator is not supported against the emulator")
    if not (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")):
        pytest.skip("no GOOGLE_CLOUD_PROJECT set for live accelerator tests")


# ---------------------------------------------------------------------------
# Controlled-binary factories (fault + race injection for the real daemon)
# ---------------------------------------------------------------------------


def _write_script(
    directory: str, name: str, body: str, *, executable: bool = True
) -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(body)
    if executable:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def missing_binary_path(directory: str) -> str:
    """A path that does not exist (drives FileNotFoundError resolution)."""
    return os.path.join(directory, "does-not-exist-binary")


def nonexecutable_binary(directory: str) -> str:
    """A real file without the executable bit (drives an OSError on spawn)."""
    return _write_script(directory, "not-exec", "#!/bin/sh\nexit 0\n", executable=False)


def immediately_exiting_binary(directory: str, exit_code: int = 1) -> str:
    """A binary that exits during startup (drives the exit-during-startup path)."""
    return _write_script(
        directory,
        "exits-now",
        f"#!/bin/sh\nprintf 'boom\\n' 1>&2\nexit {exit_code}\n",
    )


def never_binds_binary(directory: str) -> str:
    """A binary that runs, stays alive, but never binds the UDS.

    Drives ``AcceleratorDaemon.start()`` down its startup-timeout path with the
    process still alive. Still drive it through ``call_with_timeout`` +
    ``force_kill_daemon`` so that a regression of the old ``_read_stderr_tail``
    hang surfaces as a bounded ``TimeoutError`` rather than wedging the suite.
    """
    return _write_script(directory, "never-binds", "#!/bin/sh\nexec sleep 3600\n")


def slow_bind_binary(directory: str, delay_seconds: float) -> str:
    """A binary that binds its UDS after ``delay_seconds``, then serves nothing.

    Used by the startup-race tests to drive ``AcceleratorDaemon._wait_until_ready``
    right up against its timeout. It parses ``--uds-path``, waits, binds an
    AF_UNIX socket, and stays alive until stdin closes.
    """
    body = f"""#!{sys.executable}
import os, socket, sys, time, signal

def uds_path(argv):
    for i, a in enumerate(argv):
        if a == "--uds-path":
            return argv[i + 1]
    raise SystemExit("no --uds-path")

path = uds_path(sys.argv)
# Consume the handshake secret line from stdin so the writer never blocks.
try:
    sys.stdin.readline()
except Exception:
    pass
time.sleep({delay_seconds!r})
try:
    os.unlink(path)
except OSError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(8)
# Exit when stdin reaches EOF (parent closed it), mirroring the real daemon.
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
try:
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
except Exception:
    pass
os._exit(0)
"""
    return _write_script(directory, "slow-bind", body)


# ---------------------------------------------------------------------------
# Process / FD / tempdir introspection (leak detection)
# ---------------------------------------------------------------------------

# Prefix the daemon wrapper uses for its per-daemon tempdir (holds the UDS).
ACCEL_TEMPDIR_GLOB = os.path.join("/tmp", "bt-accel-*")


def accel_tempdirs() -> set[str]:
    return set(glob.glob(ACCEL_TEMPDIR_GLOB))


@dataclass
class LeakSnapshot:
    child_pids: set[int]
    num_fds: int
    tempdirs: set[str]


class ProcessIntrospector:
    """Snapshots the current process's children, FDs, and accel tempdirs.

    Requires ``psutil``. Used to assert that constructing and closing accelerated
    tables leaves no orphaned daemon subprocesses, leaked file descriptors, or
    stray ``/tmp/bt-accel-*`` directories.
    """

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())

    def snapshot(self) -> LeakSnapshot:
        children = set()
        for child in self._proc.children(recursive=True):
            try:
                children.add(child.pid)
            except self._psutil.Error:
                pass
        try:
            num_fds = self._proc.num_fds()
        except (self._psutil.Error, AttributeError):
            num_fds = -1
        return LeakSnapshot(
            child_pids=children, num_fds=num_fds, tempdirs=accel_tempdirs()
        )

    def assert_no_leaks(
        self, before: LeakSnapshot, *, fd_slack: int = 8, label: str = ""
    ) -> None:
        after = self.snapshot()
        leaked_children = after.child_pids - before.child_pids
        leaked_dirs = after.tempdirs - before.tempdirs
        prefix = f"[{label}] " if label else ""
        assert not leaked_children, (
            f"{prefix}leaked {len(leaked_children)} daemon subprocess(es): "
            f"{sorted(leaked_children)}"
        )
        assert not leaked_dirs, f"{prefix}leaked accel tempdirs: {sorted(leaked_dirs)}"
        if before.num_fds >= 0 and after.num_fds >= 0:
            assert after.num_fds <= before.num_fds + fd_slack, (
                f"{prefix}fd count grew from {before.num_fds} to {after.num_fds} "
                f"(slack {fd_slack})"
            )


def call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run ``fn()`` on a daemon thread, raising ``TimeoutError`` if it hangs.

    Used to bound calls into ``AcceleratorDaemon.start()`` so a hang in the
    daemon wrapper (see ``never_binds_binary``) cannot wedge the test suite. The
    worker thread is left running (it is blocked in a syscall and cannot be
    force-joined); callers that time out should ``force_kill_daemon`` to unblock
    and reap it.
    """
    import threading

    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"call did not return within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def force_kill_daemon(daemon: Any) -> None:
    """Best-effort SIGKILL of a daemon subprocess, ignoring all errors.

    Reaches into the wrapper's private ``_proc`` because the public ``close()``
    can itself block on a wedged process. Safe to call on a half-started daemon.
    """
    proc = getattr(daemon, "_proc", None)
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


def daemon_pid(table: Any) -> int | None:
    """Return the real daemon's pid for an accelerated table, or None."""
    daemon = getattr(table, "_accelerator_daemon", None)
    if daemon is None:
        return None
    try:
        return daemon.pid
    except RuntimeError:
        return None


def pid_alive(pid: int) -> bool:
    import psutil

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


# ---------------------------------------------------------------------------
# Load-driver primitives
# ---------------------------------------------------------------------------


class TokenBucket:
    """A simple monotonic-clock rate limiter shared by concurrency + stress.

    ``time_until_next()`` returns the seconds a caller should sleep before its
    next operation to hold the target rate; ``consume()`` records that an
    operation happened. Kept pure (clock injected) so it is deterministically
    testable and usable from both async and threaded drivers.
    """

    def __init__(
        self, rate_per_sec: float, *, clock: Callable[[], float] = time.monotonic
    ):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._interval = 1.0 / rate_per_sec
        self._clock = clock
        self._next_at = clock()

    def time_until_next(self) -> float:
        return max(0.0, self._next_at - self._clock())

    def consume(self) -> None:
        now = self._clock()
        # Advance the schedule; never let it fall arbitrarily behind real time.
        self._next_at = max(self._next_at + self._interval, now)


@dataclass
class LatencyStats:
    """Accumulates latencies (seconds) and computes percentiles on demand."""

    samples: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.samples.append(seconds)

    def percentile(self, pct: float) -> float:
        if not self.samples:
            return float("nan")
        ordered = sorted(self.samples)
        k = max(
            0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        )
        return ordered[k]

    def summary_ms(self) -> dict[str, float]:
        if not self.samples:
            return {"count": 0}
        return {
            "count": len(self.samples),
            "p50_ms": self.percentile(50) * 1000,
            "p99_ms": self.percentile(99) * 1000,
            "max_ms": max(self.samples) * 1000,
            "mean_ms": (sum(self.samples) / len(self.samples)) * 1000,
        }


# ---------------------------------------------------------------------------
# Seeded random operation generator (stress + concurrency)
# ---------------------------------------------------------------------------


class RandomOps:
    """Deterministic (seeded) generator of keys, values, mutations, and reads.

    Each worker should use a disjoint ``key_prefix`` so concurrent writers never
    contend on the same row, keeping the per-worker expected state exact.
    """

    def __init__(
        self,
        seed: int,
        key_prefix: bytes = b"accel-",
        *,
        include_aggregate: bool = False,
    ):
        import random

        self._rand = random.Random(seed)
        self._prefix = key_prefix
        # A small, reused keyspace so reads hit written rows most of the time.
        self._keyspace = [key_prefix + f"{i:08d}".encode() for i in range(256)]
        # Aggregate ops need ``TEST_AGGREGATE_FAMILY`` on the table; keep them
        # opt-in so callers running against tables without that family (e.g. the
        # self-managed stress table) are unaffected.
        self._include_aggregate = include_aggregate
        # Randomize the column cardinality per instance (seeded) so different
        # runs spread cells across a variable number of qualifiers rather than a
        # fixed handful of columns. The number of *families* is capped by the
        # table schema: two ordinary families, plus the aggregate family when
        # enabled.
        self._families = [TEST_FAMILY, TEST_FAMILY_2]
        self._qualifiers = [
            f"q{i}".encode() for i in range(self._rand.randint(2, 16))
        ]
        self._agg_qualifiers = [
            f"agg{i}".encode() for i in range(self._rand.randint(1, 8))
        ]

    def key(self) -> bytes:
        return self._rand.choice(self._keyspace)

    def value(self, max_len: int = 64) -> bytes:
        n = self._rand.randint(0, max_len)
        return bytes(self._rand.getrandbits(8) for _ in range(n))

    def ms_timestamp(self) -> int:
        # Recent-ish, millisecond-granular timestamps.
        return self._rand.randint(1, 2_000_000) * MS

    def family(self) -> str:
        return self._rand.choice(self._families)

    def qualifier(self) -> bytes:
        return self._rand.choice(self._qualifiers)

    def agg_qualifier(self) -> bytes:
        return self._rand.choice(self._agg_qualifiers)

    def add_delta(self) -> int:
        # Bounded so even a long run of adds to one cell stays well inside int64.
        return self._rand.randint(-(2**20), 2**20)

    def _delete_target(self) -> tuple[str, bytes]:
        """A coherent (family, qualifier) to delete from — sometimes the
        aggregate family so aggregate cells are actually cleared, not just set."""
        if self._include_aggregate and self._rand.random() < 0.30:
            return TEST_AGGREGATE_FAMILY, self.agg_qualifier()
        return self.family(), self.qualifier()

    def _delete_family(self) -> str:
        families = self._families + (
            [TEST_AGGREGATE_FAMILY] if self._include_aggregate else []
        )
        return self._rand.choice(families)

    def build_set(self, expected_state: ExpectedState) -> tuple[bytes, SetCell]:
        key = self.key()
        mut = expected_state.set_cell(
            key, self.family(), self.qualifier(), self.value(), self.ms_timestamp()
        )
        return key, mut

    def build_mutation(self, expected_state: ExpectedState) -> tuple[bytes, object]:
        """Pick a random mutation (weighted toward writes), apply it to the
        expected state, and return ``(row_key, mutation)`` to send to the real
        table.

        Covers every accelerator-routed mutation flavor: set-cell, the three
        delete kinds (bounded and open-ended column ranges included), and — when
        ``include_aggregate`` is set — the non-idempotent int64 ``sum``
        add-to-cell aggregate.
        """
        roll = self._rand.random()
        key = self.key()
        if self._include_aggregate and roll < 0.18:
            return key, expected_state.add_to_cell(
                key, self.agg_qualifier(), self.add_delta(), self.ms_timestamp()
            )
        if roll < 0.70:
            return key, expected_state.set_cell(
                key, self.family(), self.qualifier(), self.value(), self.ms_timestamp()
            )
        if roll < 0.82:
            # Bounded or half-open column range; keep start <= end when both set.
            a = self._rand.choice([None, self.ms_timestamp()])
            b = self._rand.choice([None, self.ms_timestamp()])
            if a is not None and b is not None and a > b:
                a, b = b, a
            fam, qual = self._delete_target()
            return key, expected_state.delete_range_from_column(key, fam, qual, a, b)
        if roll < 0.92:
            return key, expected_state.delete_from_family(key, self._delete_family())
        return key, expected_state.delete_from_row(key)

    def read_query(self, key: bytes) -> ReadRowsQuery:
        return ReadRowsQuery(row_keys=key)
