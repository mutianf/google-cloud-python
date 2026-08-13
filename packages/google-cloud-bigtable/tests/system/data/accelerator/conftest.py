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
"""Pytest configuration shared by the accelerator system-test package.

The ``event_loop`` fixture and the async/sync markers are inherited from
``tests/system/conftest.py``. Here we only register the markers this package
uses and provide a scratch directory for the controlled-binary fault-injection
tests.
"""

import functools
import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _default_app_profile():
    """Default every ``get_table`` in this package to ``BIGTABLE_TEST_APP_PROFILE``.

    Several accelerator tests build clients/tables inline via
    ``get_table(instance_id, table_id)`` without an app profile. Against this dev
    instance the accelerator requires single-cluster routing (``jetstream100``);
    the default app profile trips the daemon's session pool and silently falls
    back to native. Rather than thread the profile through ~10 call sites (and
    their CrossSync-generated twins), inject it here when the caller did not
    specify one explicitly (so tests that pass their own, e.g. config-forwarding,
    are untouched).
    """
    profile = os.getenv("BIGTABLE_TEST_APP_PROFILE")
    if not profile:
        yield
        return

    from google.cloud.bigtable.data._async.client import BigtableDataClientAsync
    from google.cloud.bigtable.data._sync_autogen.client import BigtableDataClient

    patched = []
    for cls in (BigtableDataClientAsync, BigtableDataClient):
        original = cls.get_table

        def make_wrapper(original):
            @functools.wraps(original)
            def get_table(self, instance_id, table_id, *args, **kwargs):
                if not args and "app_profile_id" not in kwargs:
                    kwargs["app_profile_id"] = profile
                return original(self, instance_id, table_id, *args, **kwargs)

            return get_table

        cls.get_table = make_wrapper(original)
        patched.append((cls, original))

    yield

    for cls, original in patched:
        cls.get_table = original


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "accelerator: pre-release integration tests for the Bigtable accelerator",
    )
    config.addinivalue_line(
        "markers",
        "slow: longer-running accelerator tests (leak loops, startup races)",
    )


@pytest.fixture
def tmp_bin_dir():
    """A fresh temp directory for building controlled daemon binaries.

    Used by the error-handling and startup tests to hand deliberately broken or
    slow binaries to the *real* ``AcceleratorDaemon`` as inputs.
    """
    with tempfile.TemporaryDirectory(prefix="accel-test-bin-") as d:
        yield d
