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

import tempfile

import pytest


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
