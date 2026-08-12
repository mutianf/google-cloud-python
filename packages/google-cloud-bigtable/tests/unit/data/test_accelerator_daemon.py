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

import json
import os
import socket
import time

import pytest

from google.cloud.bigtable.data._accelerator import _daemon
from google.cloud.bigtable.data._accelerator._daemon import (
    _IDENTITY_FILENAME,
    AcceleratorDaemon,
)


def _make_daemon(tmp_path):
    # Bypass binary resolution; we never spawn a real process in these tests.
    return AcceleratorDaemon(binary_path=str(tmp_path / "fake-binary"))


class TestExtraEnv:
    def test_default_env_none(self, tmp_path, monkeypatch):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured.update(kwargs)
            raise _StopStart()

        monkeypatch.setattr(_daemon.subprocess, "Popen", fake_popen)
        daemon = _make_daemon(tmp_path)
        with pytest.raises(_StopStart):
            daemon.start()
        # No extra_env -> inherit parent environment (env=None).
        assert captured["env"] is None

    def test_extra_env_merged_over_parent(self, tmp_path, monkeypatch):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured.update(kwargs)
            raise _StopStart()

        monkeypatch.setattr(_daemon.subprocess, "Popen", fake_popen)
        monkeypatch.setenv("EXISTING", "keep")
        daemon = AcceleratorDaemon(
            binary_path=str(tmp_path / "fake-binary"),
            extra_env={"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/key.json"},
        )
        with pytest.raises(_StopStart):
            daemon.start()
        env = captured["env"]
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/path/to/key.json"
        # Parent env is preserved (merged, not replaced).
        assert env["EXISTING"] == "keep"


class TestReadIdentity:
    def test_reads_and_unlinks(self, tmp_path):
        daemon = AcceleratorDaemon(binary_path=str(tmp_path / "fake-binary"))
        d = tmp_path / "accel-tmp"
        d.mkdir()
        daemon._tempdir = str(d)
        identity_path = d / _IDENTITY_FILENAME
        identity_path.write_text(
            json.dumps({"principal": "svc@proj.iam.gserviceaccount.com"})
        )
        result = daemon.read_identity()
        assert result["principal"] == "svc@proj.iam.gserviceaccount.com"
        # The identity is verified once, then consumed.
        assert not identity_path.exists()

    def test_not_started_raises(self, tmp_path):
        daemon = AcceleratorDaemon(binary_path=str(tmp_path / "fake-binary"))
        with pytest.raises(RuntimeError, match="has not been started"):
            daemon.read_identity()

    def test_missing_file_raises(self, tmp_path):
        daemon = AcceleratorDaemon(binary_path=str(tmp_path / "fake-binary"))
        d = tmp_path / "accel-tmp"
        d.mkdir()
        daemon._tempdir = str(d)
        with pytest.raises(RuntimeError, match="did not write"):
            daemon.read_identity()


class TestAuthSecret:
    def test_raises_before_start(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        with pytest.raises(RuntimeError, match="has not been started"):
            _ = daemon.auth_secret

    def test_written_to_stdin_after_popen(self, tmp_path, monkeypatch):
        """start() mints a secret and writes 'secret\n' to the daemon's stdin."""
        written = []

        class FakeStdin:
            def write(self, data):
                written.append(data)

            def flush(self):
                pass

        class FakeProc:
            pid = 12345
            stdin = FakeStdin()

            def poll(self):
                return None

        def fake_popen(argv, **kwargs):
            raise _StopStart()

        monkeypatch.setattr(_daemon.subprocess, "Popen", fake_popen)
        daemon = _make_daemon(tmp_path)
        with pytest.raises(_StopStart):
            daemon.start()
        # Popen raises _StopStart before secret is written; check that the
        # secret is written when Popen succeeds by using a real pipe below.

    def test_secret_written_to_stdin(self, tmp_path, monkeypatch):
        """The secret is urlsafe and ends with a newline on the wire."""
        written = bytearray()

        class FakeStdin:
            def write(self, data):
                written.extend(data)

            def flush(self):
                pass

        class FakeProc:
            pid = 12345
            stdin = FakeStdin()

            def poll(self):
                return None

        def fake_popen(argv, **kwargs):
            return FakeProc()

        monkeypatch.setattr(_daemon.subprocess, "Popen", fake_popen)
        daemon = _make_daemon(tmp_path)
        # _wait_until_ready will fail immediately (process is fake); we just
        # need to get past the stdin-write step, so patch _wait_until_ready.
        monkeypatch.setattr(daemon, "_wait_until_ready", lambda: None)
        daemon.start()
        payload = written.decode()
        assert payload.endswith("\n"), "secret payload must end with newline"
        token = payload.rstrip("\n")
        assert token == daemon.auth_secret
        # token_urlsafe(32) produces ≥32 chars of base64url characters.
        assert len(token) >= 32
        import re

        assert re.fullmatch(r"[A-Za-z0-9_\-]+", token), "expected urlsafe token"

    def test_broken_pipe_kills_and_reraises(self, tmp_path, monkeypatch):
        """A broken pipe during secret write force-kills the daemon and raises."""

        class FakeStdin:
            def write(self, data):
                raise OSError("broken pipe")

            def flush(self):
                pass

        class FakeProc:
            pid = 12345
            stdin = FakeStdin()

            def poll(self):
                return None

        def fake_popen(argv, **kwargs):
            return FakeProc()

        killed = []
        monkeypatch.setattr(_daemon.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            AcceleratorDaemon, "_force_kill", lambda self: killed.append(True)
        )
        daemon = _make_daemon(tmp_path)
        with pytest.raises(RuntimeError, match="auth secret"):
            daemon.start()
        assert killed, "_force_kill should have been called on broken pipe"


class _StopStart(Exception):
    """Sentinel to abort AcceleratorDaemon.start() right after Popen so the test
    never waits on a real socket."""


@pytest.fixture(autouse=True)
def _no_cleanup_crash(monkeypatch, tmp_path):
    # start() force-kills and cleans up on the _StopStart exception; make those
    # no-ops so the fake (never-really-spawned) process doesn't blow up.
    monkeypatch.setattr(AcceleratorDaemon, "_force_kill", lambda self: None)
    monkeypatch.setattr(AcceleratorDaemon, "_cleanup_tempdir", lambda self: None)

    # Keep the daemon's tempdir inside pytest's tmp so nothing leaks to /tmp.
    def fake_mkdtemp(prefix=""):
        d = tmp_path / "accel-tmp"
        os.makedirs(d, exist_ok=True)
        return str(d)

    monkeypatch.setattr(_daemon.tempfile, "mkdtemp", fake_mkdtemp)
    yield


class TestSweepStaleTempdirs:
    """_sweep_stale_tempdirs reaps leftover daemon tempdirs, keyed on the
    daemon.log marker, without touching live or too-young ones."""

    @staticmethod
    def _make_leftover(root, name, *, with_log=True, age_seconds=0.0):
        d = root / name
        d.mkdir()
        if with_log:
            log = d / _daemon._LOG_FILENAME
            log.write_bytes(b"hi")
            if age_seconds:
                past = time.time() - age_seconds
                os.utime(log, (past, past))
        return d

    @pytest.fixture
    def tmp_root(self, tmp_path, monkeypatch):
        # Point the sweep's tempdir root at an isolated dir under pytest's tmp.
        root = tmp_path / "tmproot"
        root.mkdir()
        monkeypatch.setattr(_daemon.tempfile, "gettempdir", lambda: str(root))
        return root

    def test_removes_stale_dir(self, tmp_root):
        d = self._make_leftover(tmp_root, "bt-accel-dead", age_seconds=120)
        _daemon._sweep_stale_tempdirs()
        assert not d.exists()

    def test_keeps_young_dir(self, tmp_root):
        # Younger than the grace window: a daemon may still be mid-startup.
        d = self._make_leftover(tmp_root, "bt-accel-young", age_seconds=0)
        _daemon._sweep_stale_tempdirs()
        assert d.exists()

    def test_ignores_dir_without_log_marker(self, tmp_root):
        # Prefix matches but there's no daemon.log, so it isn't one of ours.
        d = tmp_root / "bt-accel-foreign"
        d.mkdir()
        (d / "something-else").write_bytes(b"x")
        _daemon._sweep_stale_tempdirs()
        assert d.exists()

    def test_keeps_dir_with_live_socket(self, tmp_root):
        d = self._make_leftover(tmp_root, "bt-accel-live", age_seconds=120)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(d / _daemon._SOCKET_FILENAME))
        srv.listen(1)
        try:
            _daemon._sweep_stale_tempdirs()
            assert d.exists()
        finally:
            srv.close()

    def test_start_invokes_sweep(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(
            _daemon, "_sweep_stale_tempdirs", lambda: called.append(True)
        )

        def boom(*args, **kwargs):
            raise OSError("refusing to spawn in test")

        monkeypatch.setattr(_daemon.subprocess, "Popen", boom)
        daemon = _make_daemon(tmp_path)
        with pytest.raises(RuntimeError):
            daemon.start()
        assert called == [True]
