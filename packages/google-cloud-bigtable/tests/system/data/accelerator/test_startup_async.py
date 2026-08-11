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
"""Startup-race tests through the real client.

Drives the client's start path against a daemon that binds its UDS *late* (but
before the startup deadline). The client must wait for the bind rather than fail
early, and — because that stand-in daemon can't complete identity verification —
must then degrade gracefully to native. Low-level start-failure timing (and the
known never-binds hang) live in the sync-only ``test_daemon_startup`` module.
"""

import time

import pytest

from google.cloud.bigtable.data._cross_sync import CrossSync

from . import _harness

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_startup_autogen"


@CrossSync.convert_class(sync_name="TestStartup")
class TestStartupAsync(AcceleratorTestBase):
    """The client waits for a late UDS bind, then falls back cleanly."""

    @CrossSync.pytest
    async def test_slow_bind_waits_then_falls_back(
        self, tmp_bin_dir, instance_id, table_id, monkeypatch
    ):
        """A daemon that binds after a delay must not trip a premature start
        failure; the client waits for the bind and then degrades to native
        (the stand-in can't pass identity verification)."""
        delay = 1.5
        path = _harness.slow_bind_binary(tmp_bin_dir, delay)
        monkeypatch.setenv(_harness.BIN_ENV_VAR, path)

        started = time.monotonic()
        async with self._make_client(use_accelerator=True) as client:
            with pytest.warns(RuntimeWarning):
                table = client.get_table(instance_id, table_id)
            elapsed = time.monotonic() - started
            async with table:
                # It waited for the (late) bind rather than failing immediately.
                assert elapsed >= delay * 0.5, (
                    f"start returned in {elapsed:.2f}s, before the {delay}s bind; "
                    "it did not wait for the UDS to become connectable"
                )
                # And it degraded gracefully rather than raising.
                assert table._accelerator_client is None
