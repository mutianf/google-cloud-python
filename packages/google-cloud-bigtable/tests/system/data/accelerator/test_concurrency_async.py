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
"""Concurrency / thread-safety tests for a shared accelerated table.

A single ``Table`` (and its single ``_accelerator_client`` over one UDS) is
driven from many concurrent workers — async tasks in the async build, real OS
threads in the generated sync build. Each worker owns a disjoint key prefix and
its own model, so any interleaving/corruption in the shared client surfaces as a
per-worker mismatch or a raised exception.
"""

import concurrent.futures
import functools
import multiprocessing
import os
import uuid

from google.cloud.bigtable.data._cross_sync import CrossSync

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_concurrency_autogen"


def _run_worker(table, worker_id, ops_per_worker):
    """Plain-sync per-worker workload: a private sequence of mutations/reads,
    self-verified against a per-worker model. Returns the keys it touched.

    Shared by the multi-process workers, which always drive the *sync* client
    from OS threads. Each worker owns a disjoint key prefix, so any
    interleaving/corruption surfaces as a per-worker mismatch or exception.
    """
    prefix = f"conc-{worker_id}-{uuid.uuid4().hex[:8]}-".encode()
    ops = _harness.RandomOps(worker_id, key_prefix=prefix)
    expected_state = _harness.ExpectedState()
    touched = set()
    for _ in range(ops_per_worker):
        key, mutation = ops.build_mutation(expected_state)
        table.mutate_row(key, mutation)
        touched.add(key)
    for key in touched:
        row = table.read_row(key)
        _harness.assert_rows_equivalent(
            f"worker-{worker_id}",
            _harness.normalize_row(row),
            "model",
            expected_state.expected_cells(key),
        )
    return sorted(touched)


def _process_workload(
    project,
    instance_id,
    table_id,
    base_worker_id,
    workers_per_process,
    ops_per_worker,
    result_queue,
):
    """Child-process entrypoint for the multi-process concurrency test.

    Each process builds its OWN accelerated client + table — hence its own daemon
    and UDS, since the daemon is spawned per ``Table`` — then drives it from
    several threads at once. This exercises inter-process concurrency (multiple
    independent daemons) on top of intra-process thread-safety (one shared
    client). Reports ``(ok, touched_keys, error)`` to the parent. Defined at
    module scope so it is importable by ``multiprocessing``.
    """
    from google.cloud.bigtable.data import BigtableDataClient

    try:
        with BigtableDataClient(project=project, use_accelerator=True) as client:
            with client.get_table(instance_id, table_id) as table:
                if table._accelerator_client is None:
                    result_queue.put((False, [], "accelerator fell back to native"))
                    return
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers_per_process
                ) as executor:
                    futures = [
                        executor.submit(
                            _run_worker, table, base_worker_id + w, ops_per_worker
                        )
                        for w in range(workers_per_process)
                    ]
                    touched = []
                    for future in futures:
                        touched.extend(future.result())
        result_queue.put((True, touched, None))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent process
        result_queue.put((False, [], repr(exc)))


@CrossSync.convert_class(sync_name="TestConcurrency")
class TestConcurrencyAsync(AcceleratorTestBase):
    """Many concurrent callers sharing one accelerated table."""

    NUM_WORKERS = 16
    OPS_PER_WORKER = 50
    # Multi-process fan-out: independent processes, each with its own daemon and
    # several concurrent threads.
    NUM_PROCESSES = 4
    WORKERS_PER_PROCESS = 8

    @CrossSync.convert
    async def _worker(self, table, worker_id):
        """Do a private sequence of mutations/reads and self-verify against a
        per-worker model. Returns the keys it touched (for cleanup)."""
        prefix = f"conc-{worker_id}-{uuid.uuid4().hex[:8]}-".encode()
        ops = _harness.RandomOps(worker_id, key_prefix=prefix)
        expected_state = _harness.ExpectedState()
        touched = set()
        for _ in range(self.OPS_PER_WORKER):
            key, mutation = ops.build_mutation(expected_state)
            await table.mutate_row(key, mutation)
            touched.add(key)
        for key in touched:
            row = await table.read_row(key)
            _harness.assert_rows_equivalent(
                f"worker-{worker_id}",
                _harness.normalize_row(row),
                "model",
                expected_state.expected_cells(key),
            )
        return sorted(touched)

    @CrossSync.pytest
    async def test_concurrent_workers_stay_consistent(self, accel_table, janitor):
        """Concurrent writers/readers on one shared client each see exactly their
        own writes, with no errors."""
        executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=self.NUM_WORKERS)
            if not CrossSync.is_async
            else None
        )
        try:
            partials = [
                functools.partial(self._worker, accel_table, i)
                for i in range(self.NUM_WORKERS)
            ]
            results = await CrossSync.gather_partials(
                partials, return_exceptions=True, sync_executor=executor
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        for i, result in enumerate(results):
            assert not isinstance(result, BaseException), (
                f"worker {i} failed: {result!r}"
            )
            for key in result:
                janitor.track(key)

    @CrossSync.pytest
    async def test_multiprocess_concurrent_clients_stay_consistent(
        self, instance_id, table_id, janitor
    ):
        """Many *processes*, each with its own accelerated client + daemon and
        several concurrent threads, must all stay correct with no errors.

        A single shared client is not enough coverage: the daemon is per-process,
        so real deployments run many independent daemons at once. Each process
        owns a disjoint block of worker ids (hence disjoint key prefixes), so any
        cross-process interference surfaces as a per-worker mismatch or a raised
        exception reported back over the queue.
        """
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or None
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue()
        procs = []
        for p in range(self.NUM_PROCESSES):
            proc = ctx.Process(
                target=_process_workload,
                args=(
                    project,
                    instance_id,
                    table_id,
                    p * self.WORKERS_PER_PROCESS,
                    self.WORKERS_PER_PROCESS,
                    self.OPS_PER_WORKER,
                    result_queue,
                ),
            )
            proc.start()
            procs.append(proc)
        results = []
        try:
            for _ in range(self.NUM_PROCESSES):
                results.append(result_queue.get(timeout=300))
        finally:
            for proc in procs:
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.kill()
                    proc.join()

        for ok, touched, error in results:
            assert ok, f"a worker process failed: {error}"
            for key in touched:
                janitor.track(key)
