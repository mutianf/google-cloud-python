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
"""Metrics-ownership parity tests.

For accelerated RPCs the daemon owns retries *and* metrics — the Python layer
deliberately attaches a handler-less metric (read_row) or skips metrics entirely
(mutate_row), so the client-side metrics handlers must observe nothing for those
calls. The native client, by contrast, records one completed operation per call.
This pins down that ownership boundary so accelerated traffic is never
double-counted by the Python exporter.
"""

import uuid

from google.cloud.bigtable.data._cross_sync import CrossSync
from google.cloud.bigtable.data._metrics.handlers._base import MetricsHandler
from google.cloud.bigtable.data.mutations import DeleteAllFromRow, SetCell

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_metrics_parity_autogen"


class _CountingMetricsHandler(MetricsHandler):
    """Records completed operations/attempts so tests can count them."""

    def __init__(self, **kwargs):
        self.completed_operations = []
        self.completed_attempts = []

    def on_operation_complete(self, op):
        self.completed_operations.append(op)

    def on_attempt_complete(self, attempt, _):
        self.completed_attempts.append(attempt)


@CrossSync.convert_class(sync_name="TestMetricsParity")
class TestMetricsParityAsync(AcceleratorTestBase):
    """Client-side metrics fire for native RPCs but not accelerated ones."""

    @CrossSync.convert
    async def _read_and_write(self, table):
        key = f"metrics-{uuid.uuid4().hex}".encode()
        await table.mutate_row(
            key,
            SetCell(
                _harness.TEST_FAMILY, b"q", b"v", timestamp_micros=1000 * _harness.MS
            ),
        )
        await table.read_row(key)
        await table.mutate_row(key, DeleteAllFromRow())

    @CrossSync.pytest
    async def test_accelerated_ops_bypass_python_metrics(self, instance_id, table_id):
        """Accelerated read_row/mutate_row emit no client-side metrics; the native
        path emits one completed operation per call."""
        accel_handler = _CountingMetricsHandler()
        native_handler = _CountingMetricsHandler()

        async with (
            self._make_client(use_accelerator=True) as accel_client,
            self._make_client(use_accelerator=False) as native_client,
        ):
            async with (
                accel_client.get_table(instance_id, table_id) as accel_table,
                native_client.get_table(instance_id, table_id) as native_table,
            ):
                self.assert_accelerator_active(accel_table)
                self.assert_native(native_table)
                accel_table._metrics.add_handler(accel_handler)
                native_table._metrics.add_handler(native_handler)

                await self._read_and_write(accel_table)
                await self._read_and_write(native_table)

        # Native path recorded a completed operation for each of its RPCs.
        assert len(native_handler.completed_operations) >= 2, (
            "native client should record client-side metrics for its RPCs; got "
            f"{len(native_handler.completed_operations)}"
        )
        # Accelerated path recorded nothing — the daemon owns those metrics.
        assert len(accel_handler.completed_operations) == 0, (
            "accelerated RPCs must not emit client-side metrics (daemon owns "
            f"them); got {len(accel_handler.completed_operations)}: "
            f"{accel_handler.completed_operations}"
        )
