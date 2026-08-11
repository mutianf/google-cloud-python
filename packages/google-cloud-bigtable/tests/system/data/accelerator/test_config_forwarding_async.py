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
"""Config-forwarding tests: the client must launch the daemon with the same
identity-affecting knobs it would use itself.

We check this grey-box against the real objects rather than functionally,
because most of these knobs (scopes, quota project, app profile) only change
behavior under specific IAM/routing setups that a generic test instance can't
guarantee. Instead we assert the exact CLI the *real* daemon was launched with
(``_cli_flags``) and the exact forward-config the *real* client computed
(``_accelerator_flags``).
"""

import os

from google.cloud.bigtable.data._cross_sync import CrossSync

from . import _harness  # noqa: F401  (ensures package import parity with siblings)

if CrossSync.is_async:
    from ._base_async import AcceleratorTestBaseAsync as AcceleratorTestBase
else:
    from ._base_autogen import AcceleratorTestBase

__CROSS_SYNC_OUTPUT__ = "tests.system.data.accelerator.test_config_forwarding_autogen"


@CrossSync.convert_class(sync_name="TestConfigForwarding")
class TestConfigForwardingAsync(AcceleratorTestBase):
    """The daemon is launched with the client's identity-affecting config."""

    @CrossSync.convert
    async def _accelerator_flags_for(self, **client_options):
        """Return the ``_accelerator_flags`` a client computes for the given
        ``client_options`` (no daemon/RPC needed — this is pure construction)."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or None
        async with CrossSync.DataClient(
            project=project, use_accelerator=True, client_options=client_options
        ) as client:
            return list(client._accelerator_flags)

    @staticmethod
    def _flag_value(flags, name):
        assert name in flags, f"expected {name} in daemon/client flags: {flags}"
        return flags[flags.index(name) + 1]

    @CrossSync.pytest
    async def test_effective_scopes_forwarded_by_default(self):
        """The client always forwards its effective auth scopes so the daemon's
        token audience matches."""
        flags = await self._accelerator_flags_for()
        assert "--scopes" in flags
        # Also forwards a caller user-agent so the daemon can build the UA prefix.
        assert "--caller-user-agent" in flags

    @CrossSync.pytest
    async def test_quota_project_forwarded_to_flags(self):
        flags = await self._accelerator_flags_for(quota_project_id="my-quota-project")
        assert self._flag_value(flags, "--quota-project") == "my-quota-project"

    @CrossSync.pytest
    async def test_app_profile_forwarded_to_daemon(self, instance_id, table_id):
        """The per-table app profile is passed to the daemon it launches."""
        async with self._make_client(use_accelerator=True) as client:
            async with client.get_table(
                instance_id, table_id, app_profile_id="test-profile"
            ) as table:
                self.assert_accelerator_active(table)
                flags = table._accelerator_daemon._cli_flags
                assert self._flag_value(flags, "--app-profile") == "test-profile"
                # The base project/instance are always forwarded too.
                assert self._flag_value(flags, "--instance") == instance_id
