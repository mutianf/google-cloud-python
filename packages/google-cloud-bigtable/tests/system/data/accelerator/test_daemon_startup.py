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
"""Low-level ``AcceleratorDaemon`` start-failure tests.

``AcceleratorDaemon`` is deliberately sync-only, so these are plain sync tests
(no CrossSync twin) and need neither a real backend nor the bundled binary — they
drive the real wrapper against deliberately broken stand-in binaries. They pin
down the wrapper's own failure contract, which the client relies on to decide
between routing and falling back.
"""

import time

import pytest

from google.cloud.bigtable.data._accelerator._daemon import AcceleratorDaemon

from . import _harness

_FLAGS = ["--project", "p", "--instance", "i", "--app-profile", "ap"]


def test_missing_binary_raises_at_construction(tmp_bin_dir, monkeypatch):
    """A resolved binary path that doesn't exist fails fast, at construction,
    before any spawn. This is the env-var resolution contract the client relies
    on (an explicit ``binary_path`` is instead validated lazily at ``start()``)."""
    path = _harness.missing_binary_path(tmp_bin_dir)
    monkeypatch.setenv(_harness.BIN_ENV_VAR, path)
    with pytest.raises(FileNotFoundError):
        AcceleratorDaemon(_FLAGS)


def test_nonexecutable_binary_raises_on_start(tmp_bin_dir):
    """A present-but-non-executable binary resolves, then fails to spawn."""
    path = _harness.nonexecutable_binary(tmp_bin_dir)
    daemon = AcceleratorDaemon(_FLAGS, binary_path=path, startup_timeout=2.0)
    with pytest.raises(RuntimeError, match="spawn"):
        _harness.call_with_timeout(daemon.start, timeout=5.0)
    _harness.force_kill_daemon(daemon)


def test_exit_during_startup_raises_promptly(tmp_bin_dir):
    """A daemon that exits during startup raises quickly (well within the
    startup timeout), reporting the exit."""
    path = _harness.immediately_exiting_binary(tmp_bin_dir, exit_code=3)
    daemon = AcceleratorDaemon(_FLAGS, binary_path=path, startup_timeout=5.0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited during startup"):
        _harness.call_with_timeout(daemon.start, timeout=5.0)
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, f"exit-during-startup took {elapsed:.2f}s; not prompt"
    _harness.force_kill_daemon(daemon)


def test_never_binds_honors_startup_timeout(tmp_bin_dir):
    """A daemon that stays alive but never binds the UDS must honor
    ``startup_timeout``: it raises a timeout ``RuntimeError`` shortly after the
    deadline rather than hanging in ``_read_stderr_tail`` (which reads stderr
    non-blockingly). ``call_with_timeout`` bounds the call so a regression of
    that hang surfaces as a ``TimeoutError`` instead of wedging the suite."""
    path = _harness.never_binds_binary(tmp_bin_dir)
    daemon = AcceleratorDaemon(_FLAGS, binary_path=path, startup_timeout=1.0)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="did not become ready"):
            _harness.call_with_timeout(daemon.start, timeout=4.0)
        elapsed = time.monotonic() - started
        # Raised off the 1s deadline, not by the 4s watchdog (the old hang).
        assert elapsed < 3.0, f"never-binds took {elapsed:.2f}s; not honoring timeout"
    finally:
        _harness.force_kill_daemon(daemon)
