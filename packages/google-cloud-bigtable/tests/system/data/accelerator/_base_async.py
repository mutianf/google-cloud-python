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
"""Shared base class + fixtures for the accelerator pre-release suite.

This module is written async-first and converted to a sync twin
(``_base_autogen``) by CrossSync, so async test files and their generated sync
twins can share the exact same fixtures. Everything here drives the *real*
shipped path: ``CrossSync.DataClient(..., use_accelerator=...)`` producing a real
target that spawns the real ``AcceleratorDaemon`` over the real bundled binary.

The class reuses ``SystemTestRunner`` (temporary instance/table/family creation,
stale-instance cleanup) and adds accelerator-aware client/table fixtures plus a
``janitor`` for per-test row cleanup.
"""

import os

import pytest

from google.cloud.bigtable.data._cross_sync import CrossSync
from google.cloud.bigtable.data.mutations import DeleteAllFromRow, RowMutationEntry

from . import _harness
from .. import SystemTestRunner

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator._base_autogen"


@CrossSync.convert_class(sync_name="AcceleratorTestBase")
class AcceleratorTestBaseAsync(SystemTestRunner):
    """Base for accelerator system tests.

    Subclasses inherit the accelerator-aware fixtures below. Every table fixture
    asserts that the accelerator ended up in the expected state (active for
    ``accel_table``, native for ``native_table``) so a silent fallback surfaces
    as a test failure rather than passing on the wrong code path.
    """

    @pytest.fixture(scope="session", autouse=True)
    def _require_accel_env(self):
        """Cleanly skip the whole suite when the environment can't support it.

        Runs before the (expensive) instance/table fixtures because it is an
        autouse session fixture, so a missing binary or emulator-only setup
        skips instead of erroring out mid-provisioning.
        """
        _harness.require_real_bigtable_or_skip()
        _harness.require_binary_or_skip()

    def _make_client(self, use_accelerator=None):
        """Build a real data client with the given accelerator setting."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or None
        return CrossSync.DataClient(project=project, use_accelerator=use_accelerator)

    def assert_accelerator_active(self, table):
        assert table._accelerator_client is not None, (
            "expected the accelerator to be active (use_accelerator=True) but the "
            "target fell back to native. Check ADC principal / identity "
            "verification and that the bundled daemon binary can start."
        )

    def assert_native(self, table):
        assert table._accelerator_client is None, (
            "expected a native target (use_accelerator=False) but an accelerator "
            "client was attached."
        )

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def client(self):
        """Default-on client (``use_accelerator=None``).

        Also backs the ``project_id`` fixture from ``SystemTestRunner``.
        """
        async with self._make_client() as client:
            yield client

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def accel_client(self):
        async with self._make_client(use_accelerator=True) as client:
            yield client

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def native_client(self):
        async with self._make_client(use_accelerator=False) as client:
            yield client

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def accel_table(self, accel_client, instance_id, table_id):
        async with accel_client.get_table(instance_id, table_id) as table:
            self.assert_accelerator_active(table)
            yield table

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def native_table(self, native_client, instance_id, table_id):
        async with native_client.get_table(instance_id, table_id) as table:
            self.assert_native(table)
            yield table

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="session")
    async def default_table(self, client, instance_id, table_id):
        async with client.get_table(instance_id, table_id) as table:
            yield table

    @CrossSync.convert
    @CrossSync.pytest_fixture(scope="function")
    async def janitor(self, native_table):
        """Track written row keys and delete them after each test.

        Deletion goes through the native path so cleanup never depends on the
        component under test. Yields an object with ``.track(key)``.
        """

        class _Janitor:
            def __init__(self):
                self.keys = set()

            def track(self, key):
                self.keys.add(key)
                return key

        j = _Janitor()
        yield j
        if j.keys:
            entries = [RowMutationEntry(key, [DeleteAllFromRow()]) for key in j.keys]
            await native_table.bulk_mutate_rows(entries)
