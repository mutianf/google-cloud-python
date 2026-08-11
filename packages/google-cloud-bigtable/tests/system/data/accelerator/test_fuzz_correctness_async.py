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
"""Randomized differential correctness tests for the accelerator.

These generate random mutation/read sequences and assert *triple* agreement on
every touched row:

    accelerator read  ==  expected state  ==  native read

The accelerator read proves the daemon path writes/reads Bigtable correctly; the
expected state proves it matches the intended semantics; the native read proves
the accelerator and the pure-Python client observe identical backend state.

We drive this with a seeded generator rather than Hypothesis on purpose: every
example issues live RPCs, so shrinking + Hypothesis's 100-example default would
be prohibitively slow and would trip its slow-test health checks. Seeds are
fixed and printed in assertion output, so any failure is fully reproducible.
"""

import uuid

import pytest

from google.cloud.bigtable.data._cross_sync import CrossSync

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_fuzz_correctness_autogen"


@CrossSync.convert_class(sync_name="TestFuzzCorrectness")
class TestFuzzCorrectnessAsync(AcceleratorTestBase):
    """Randomized + targeted correctness of the real accelerator write/read path."""

    NUM_OPS = 40

    @CrossSync.convert
    async def _verify_key(self, accel_table, native_table, expected_state, key):
        """Assert accelerator read == expected == native read for one row."""
        accel_row = await accel_table.read_row(key)
        native_row = await native_table.read_row(key)
        expected = expected_state.expected_cells(key)
        accel_cells = _harness.normalize_row(accel_row)
        native_cells = _harness.normalize_row(native_row)
        _harness.assert_rows_equivalent(
            "accelerator", accel_cells, "expected", expected
        )
        _harness.assert_rows_equivalent(
            "accelerator", accel_cells, "native", native_cells
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @CrossSync.pytest
    async def test_random_mutations_match_model_and_native(
        self, accel_table, native_table, janitor, seed
    ):
        """Apply a random mix of set/delete mutations through the accelerator and
        continuously reconcile against the expected state and the native client."""
        # A unique key prefix per run keeps each expected state exact even if a
        # previous run's cleanup was incomplete (no cross-test row contamination).
        prefix = f"fuzz-{seed}-{uuid.uuid4().hex[:8]}-".encode()
        ops = _harness.RandomOps(seed, key_prefix=prefix)
        expected_state = _harness.ExpectedState()
        touched: set[bytes] = set()

        for i in range(self.NUM_OPS):
            key, mutation = ops.build_mutation(expected_state)
            await accel_table.mutate_row(key, mutation)
            janitor.track(key)
            touched.add(key)
            # Reconcile a random already-touched row part-way through, so ordering
            # bugs surface mid-sequence rather than only at the end.
            if touched and i % 5 == 4:
                vkey = ops._rand.choice(sorted(touched))
                await self._verify_key(accel_table, native_table, expected_state, vkey)

        # Final full reconciliation of every row we touched.
        for key in sorted(touched):
            await self._verify_key(accel_table, native_table, expected_state, key)

    @CrossSync.pytest
    async def test_multiple_versions_and_range_delete(
        self, accel_table, native_table, janitor
    ):
        """Deterministic coverage of multi-version cells + a bounded range delete,
        so this behavior is exercised regardless of the random draw."""
        from google.cloud.bigtable.data.mutations import DeleteRangeFromColumn

        key = janitor.track(f"versions-{uuid.uuid4().hex}".encode())
        family = _harness.TEST_FAMILY
        qualifier = b"q0"
        expected_state = _harness.ExpectedState()

        # Three explicit versions of the same cell.
        for ts_ms, value in [(1000, b"v1"), (2000, b"v2"), (3000, b"v3")]:
            ts = ts_ms * _harness.MS
            await accel_table.mutate_row(
                key, expected_state.set_cell(key, family, qualifier, value, ts)
            )
        await self._verify_key(accel_table, native_table, expected_state, key)

        # Delete the middle version only: [2_000_000, 3_000_000) removes v2@2000.
        expected_state.delete_range_from_column(
            key, family, qualifier, 2000 * _harness.MS, 3000 * _harness.MS
        )
        await accel_table.mutate_row(
            key,
            DeleteRangeFromColumn(
                family, qualifier, 2000 * _harness.MS, 3000 * _harness.MS
            ),
        )
        await self._verify_key(accel_table, native_table, expected_state, key)
