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
"""Enablement-contract tests for the ``use_accelerator`` flag.

The accelerator is on by default and controlled at client-construction time:

* ``use_accelerator=None`` (default): on, with graceful fallback.
* ``use_accelerator=True``: on; still falls back on start failure, except the
  emulator, which is a hard error.
* ``use_accelerator=False``: native only.

These verify the *observable* contract on the real path (which client the target
ends up using) plus the one hard error and the emulator auto-disable.
"""

import uuid

import pytest
from google.cloud.environment_vars import BIGTABLE_EMULATOR

from google.cloud.bigtable.data._cross_sync import CrossSync
from google.cloud.bigtable.data.mutations import SetCell

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = (
    "tests.system.data.accelerator.test_default_and_fallback_autogen"
)


@CrossSync.convert_class(sync_name="TestDefaultAndFallback")
class TestDefaultAndFallbackAsync(AcceleratorTestBase):
    """The observable ``use_accelerator`` contract on the real path."""

    @CrossSync.convert
    async def _write_and_read(self, table, janitor):
        key = janitor.track(f"contract-{uuid.uuid4().hex}".encode())
        ts = 1000 * _harness.MS
        await table.mutate_row(
            key, SetCell(_harness.TEST_FAMILY, b"q", b"v", timestamp_micros=ts)
        )
        row = await table.read_row(key)
        assert row is not None and row.cells[0].value == b"v"

    @CrossSync.pytest
    async def test_default_enables_accelerator(self, default_table, janitor):
        """Default construction (no flag) turns the accelerator on and works."""
        assert default_table._accelerator_client is not None
        await self._write_and_read(default_table, janitor)

    @CrossSync.pytest
    async def test_explicit_true_enables_accelerator(self, accel_table, janitor):
        """``use_accelerator=True`` runs on the accelerated path end-to-end."""
        assert accel_table._accelerator_client is not None
        await self._write_and_read(accel_table, janitor)

    @CrossSync.pytest
    async def test_explicit_false_uses_native(self, native_table, janitor):
        """``use_accelerator=False`` never attaches an accelerator client."""
        assert native_table._accelerator_client is None
        assert native_table._accelerator_daemon is None
        await self._write_and_read(native_table, janitor)

    @CrossSync.pytest
    async def test_emulator_with_explicit_true_raises(self, monkeypatch):
        """``use_accelerator=True`` + emulator is the one hard misconfiguration."""
        monkeypatch.setenv(BIGTABLE_EMULATOR, "localhost:8086")
        async with self._make_client(use_accelerator=True) as client:
            with pytest.raises(
                RuntimeError, match="use_accelerator=True is not supported"
            ):
                client.get_table("fake-instance", "fake-table")

    @CrossSync.pytest
    async def test_emulator_default_disables_with_warning(self, monkeypatch):
        """Default + emulator auto-disables the accelerator with a warning rather
        than breaking the caller."""
        monkeypatch.setenv(BIGTABLE_EMULATOR, "localhost:8086")
        async with self._make_client() as client:
            with pytest.warns(RuntimeWarning, match="Accelerator disabled"):
                table = client.get_table("fake-instance", "fake-table")
            assert table._accelerator_client is None
