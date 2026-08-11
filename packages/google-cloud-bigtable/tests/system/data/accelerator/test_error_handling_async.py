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
"""Error-handling tests for the real accelerator path.

Covers the three failure surfaces called out for the pre-release suite:

* **The daemon fails to start.** Driven by feeding the *real* ``AcceleratorDaemon``
  a deliberately broken binary (missing / non-executable / exits-immediately).
  The contract is graceful degradation: even ``use_accelerator=True`` warns and
  falls back to a fully functional native client (only the emulator is a hard
  error — see ``test_default_and_fallback``).
* **The daemon dies mid-flight.** A dead subprocess is unrecoverable, so the
  routing layer transparently falls back to the native client and trips the
  breaker (the daemon is never dialed again) rather than hanging or surfacing an
  error. This mirrors the Go client: an accelerator that cannot serve replies is
  retried natively.
* **Deadlines.** Only ``operation_timeout`` is forwarded to the accelerator as
  the RPC deadline, and an operation left at its default gets the same default
  deadline as the native client. These tests therefore pass ``operation_timeout``
  only.
* **Errors from the backend / Python layer.** These must behave identically on
  the accelerated and native paths (same exception type), so the accelerator's
  error translation is verified against the native client as the reference.

The ``UNIMPLEMENTED``-driven fallback and the consecutive-failure breaker cannot
be induced against a real backend (they require the daemon to fail session
creation on demand), so that policy is covered by ``test_accelerator_fallback``,
a unit test that drives the real fallback module with real gRPC status codes.
"""

import uuid

import pytest

from google.api_core import exceptions as core_exceptions

from google.cloud.bigtable.data._cross_sync import CrossSync
from google.cloud.bigtable.data.mutations import DeleteAllFromRow, SetCell

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_error_handling_autogen"


@CrossSync.convert_class(sync_name="TestErrorHandling")
class TestErrorHandlingAsync(AcceleratorTestBase):
    """Real-path error handling: start failures, mid-flight death, error parity."""

    @CrossSync.convert
    async def _assert_round_trip(self, table):
        """Write and read back a unique cell, returning the key (caller cleans up)."""
        key = f"errtest-{uuid.uuid4().hex}".encode()
        ts = 1000 * _harness.MS
        await table.mutate_row(
            key, SetCell(_harness.TEST_FAMILY, b"q", b"ok", timestamp_micros=ts)
        )
        row = await table.read_row(key)
        assert row is not None and row.cells[0].value == b"ok"
        return key

    # -- 1. daemon fails to start --------------------------------------------

    @pytest.mark.parametrize("fault", ["missing", "nonexecutable", "exits"])
    @CrossSync.pytest
    async def test_start_failure_falls_back_to_native(
        self, fault, tmp_bin_dir, instance_id, table_id, monkeypatch
    ):
        """A broken daemon binary must degrade to a working native client, with a
        warning, even for an explicit ``use_accelerator=True``."""
        if fault == "missing":
            path = _harness.missing_binary_path(tmp_bin_dir)
        elif fault == "nonexecutable":
            path = _harness.nonexecutable_binary(tmp_bin_dir)
        else:
            path = _harness.immediately_exiting_binary(tmp_bin_dir)
        monkeypatch.setenv(_harness.BIN_ENV_VAR, path)

        async with self._make_client(use_accelerator=True) as client:
            with pytest.warns(RuntimeWarning, match="Failed to start"):
                table = client.get_table(instance_id, table_id)
            async with table:
                # Explicit opt-in still degraded gracefully rather than raising.
                assert table._accelerator_client is None
                assert table._accelerator_daemon is None
                # ...and the native fallback actually works end-to-end.
                key = await self._assert_round_trip(table)
                await table.mutate_row(key, DeleteAllFromRow())

    # -- 2. daemon dies mid-flight -------------------------------------------

    @CrossSync.pytest
    async def test_daemon_killed_midflight_falls_back_to_native(
        self, instance_id, table_id, native_table
    ):
        """Killing the live daemon must not fail user calls: a dead subprocess is
        unrecoverable, so the next accelerated op falls back to native, trips the
        breaker, and every later op skips the daemon entirely."""
        async with self._make_client(use_accelerator=True) as client:
            async with client.get_table(instance_id, table_id) as table:
                self.assert_accelerator_active(table)
                key = await self._assert_round_trip(table)
                assert not table._accelerator_breaker.bypass()

                # Kill the real daemon out from under the client.
                pid = _harness.daemon_pid(table)
                assert pid is not None and _harness.pid_alive(pid)
                _harness.force_kill_daemon(table._accelerator_daemon)
                assert not table._accelerator_daemon.is_running

                # The next accelerated read transparently falls back to native and
                # returns the real row rather than raising. Bound the call so a
                # wedged connection can't hang the test.
                row = await table.read_row(key, operation_timeout=30)
                assert row is not None and row.cells[0].value == b"ok"

                # That first failure tripped the breaker: the accelerator is now
                # bypassed for good, though the client object is left intact.
                assert table._accelerator_breaker.bypass()
                assert table._accelerator_client is not None
                assert table._use_accelerator("read_row") is False
                assert table._use_accelerator("mutate_row") is False

                # A later mutate also succeeds — straight down the native path now.
                after_ts = 2000 * _harness.MS
                await table.mutate_row(
                    key,
                    SetCell(
                        _harness.TEST_FAMILY, b"q", b"after", timestamp_micros=after_ts
                    ),
                    operation_timeout=30,
                )
                roundtrip = await table.read_row(key)
                assert any(
                    c.value == b"after" and c.timestamp_micros == after_ts
                    for c in roundtrip.cells
                )

        # The backend is fine — the native client still round-trips the row.
        native_row = await native_table.read_row(key)
        assert native_row is not None
        await native_table.mutate_row(key, DeleteAllFromRow())

    # -- 3. deadlines ---------------------------------------------------------

    @CrossSync.pytest
    async def test_operation_timeout_enforced_as_deadline(
        self, accel_table, native_table
    ):
        """``operation_timeout`` is the accelerator RPC deadline: an impossibly
        short budget must raise ``DeadlineExceeded`` — the same type the native
        path raises — and must not fall back (a deadline is a real result)."""
        key = f"errtest-{uuid.uuid4().hex}".encode()
        # Only operation_timeout is meaningful on the accelerated path (the daemon
        # owns retry), so it is the only budget we pass.
        with pytest.raises(core_exceptions.DeadlineExceeded):
            await accel_table.read_row(key, operation_timeout=0.001)
        with pytest.raises(core_exceptions.DeadlineExceeded):
            await native_table.read_row(key, operation_timeout=0.001)
        # The deadline did not knock the accelerator offline.
        assert not accel_table._accelerator_breaker.bypass()
        assert accel_table._accelerator_client is not None

    @CrossSync.pytest
    async def test_default_deadline_matches_native(self, accel_table, native_table):
        """An operation left at its default must use the same default deadline on
        both paths, and that default must actually round-trip a real op."""
        # The resolved client-side defaults are identical (same client config).
        assert (
            accel_table.default_read_rows_operation_timeout
            == native_table.default_read_rows_operation_timeout
        )
        assert (
            accel_table.default_operation_timeout
            == native_table.default_operation_timeout
        )
        # And the accelerated path round-trips with no explicit timeout at all,
        # proving the default deadline is applied (never unbounded).
        key = await self._assert_round_trip(accel_table)
        await native_table.mutate_row(key, DeleteAllFromRow())

    # -- 4. backend + Python-layer error parity ------------------------------

    @CrossSync.pytest
    async def test_backend_error_type_matches_native(self, accel_table, native_table):
        """A backend-rejected mutation (unknown column family) must raise the same
        exception type through the accelerator as through the native client,
        proving the daemon's error translation matches."""
        key = f"errtest-{uuid.uuid4().hex}".encode()
        bad = SetCell(
            "no-such-column-family", b"q", b"v", timestamp_micros=1000 * _harness.MS
        )
        with pytest.raises(Exception) as accel_ei:
            await accel_table.mutate_row(key, bad)
        with pytest.raises(Exception) as native_ei:
            await native_table.mutate_row(key, bad)
        assert type(accel_ei.value) is type(native_ei.value), (
            "accelerator and native raised different exception types for the same "
            f"backend error: accelerator={type(accel_ei.value).__name__} "
            f"native={type(native_ei.value).__name__}"
        )

    @CrossSync.pytest
    async def test_python_layer_validation_matches_native(
        self, accel_table, native_table
    ):
        """Client-side validation errors happen before dispatch, so they must be
        identical on both paths."""
        # read_row(None) -> ValueError on both paths.
        with pytest.raises(ValueError):
            await accel_table.read_row(None)
        with pytest.raises(ValueError):
            await native_table.read_row(None)
        # mutate_row with no mutations -> ValueError on both paths.
        key = f"errtest-{uuid.uuid4().hex}".encode()
        with pytest.raises(ValueError):
            await accel_table.mutate_row(key, [])
        with pytest.raises(ValueError):
            await native_table.mutate_row(key, [])
