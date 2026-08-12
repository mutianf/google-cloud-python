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
"""Unit tests for the real accelerator fallback policy (``_fallback.py``).

These drive the *real* shipped fallback module — ``AcceleratorBreaker``,
``handle_accelerator_error``, and ``_AcceleratorFallback`` — with real gRPC
status codes and a real (un-started) ``AcceleratorDaemon``. Nothing is faked: the
errors are genuine ``grpc.aio.AioRpcError`` objects (a subclass of
``grpc.RpcError``, so they exercise the same classifier the sync and async
clients both hit at runtime), and the "daemon is dead" signal comes from a real
``AcceleratorDaemon`` that was never started (``is_running`` is ``False``).

This covers the behavior the live system tests cannot induce against a real
backend — the ``UNIMPLEMENTED``-driven fallback that trips the breaker
immediately and stickily, mirroring the Go client's
``session.UnimplementedErrorInterceptor``.
"""

import grpc
import grpc.aio
import pytest

from google.api_core import exceptions as core_exceptions

from google.cloud.bigtable.data._accelerator._daemon import AcceleratorDaemon
from google.cloud.bigtable.data._accelerator._fallback import (
    AcceleratorBreaker,
    _AcceleratorFallback,
    handle_accelerator_error,
)


def _aio_error(code: grpc.StatusCode, details: str = "boom") -> grpc.aio.AioRpcError:
    """Build a real ``AioRpcError`` carrying a real gRPC status code."""
    return grpc.aio.AioRpcError(
        code, grpc.aio.Metadata(), grpc.aio.Metadata(), details=details
    )


def _dead_daemon() -> AcceleratorDaemon:
    """A real, never-started daemon: ``is_running`` is ``False``.

    ``__init__`` validates the binary path is a regular file but never spawns it,
    so pointing it at any existing file (this test module) yields a genuine
    wrapper in the "process not running" state without needing a bundled binary.
    """
    return AcceleratorDaemon(binary_path=__file__)


# ---------------------------------------------------------------------------
# AcceleratorBreaker
# ---------------------------------------------------------------------------


class TestAcceleratorBreaker:
    def test_starts_untripped(self):
        assert AcceleratorBreaker().bypass() is False

    def test_trip_is_immediate_and_sticky(self):
        breaker = AcceleratorBreaker()
        breaker.trip()
        assert breaker.bypass() is True
        # Sticky: nothing un-trips a tripped breaker for the client's lifetime.
        breaker.trip()
        assert breaker.bypass() is True


# ---------------------------------------------------------------------------
# handle_accelerator_error
# ---------------------------------------------------------------------------


class TestHandleAcceleratorError:
    def test_unimplemented_falls_back_and_trips_immediately(self):
        # A single UNIMPLEMENTED signals the accelerator can't serve this call
        # and permanently routes the client to the native path — no threshold.
        breaker = AcceleratorBreaker()
        exc = _aio_error(grpc.StatusCode.UNIMPLEMENTED, "no sessions available")
        with pytest.raises(_AcceleratorFallback):
            handle_accelerator_error(exc, daemon=None, breaker=breaker)
        assert breaker.bypass() is True

    def test_dead_daemon_falls_back_and_trips_regardless_of_status(self):
        breaker = AcceleratorBreaker()
        # UNAVAILABLE, not UNIMPLEMENTED: liveness is checked first, so a dead
        # subprocess falls back and trips immediately whatever the status code.
        exc = _aio_error(grpc.StatusCode.UNAVAILABLE, "connection refused")
        with pytest.raises(_AcceleratorFallback):
            handle_accelerator_error(exc, daemon=_dead_daemon(), breaker=breaker)
        assert breaker.bypass() is True

    @pytest.mark.parametrize(
        "code,expected",
        [
            (grpc.StatusCode.UNAVAILABLE, core_exceptions.ServiceUnavailable),
            (grpc.StatusCode.ABORTED, core_exceptions.Aborted),
            (grpc.StatusCode.DEADLINE_EXCEEDED, core_exceptions.DeadlineExceeded),
            (grpc.StatusCode.NOT_FOUND, core_exceptions.NotFound),
        ],
    )
    def test_other_grpc_error_translated_and_raised(self, code, expected):
        # A live daemon serving a normal gRPC error: translate to the api_core
        # type and raise it. Not a fallback, so the breaker stays untripped.
        breaker = AcceleratorBreaker()
        with pytest.raises(expected):
            handle_accelerator_error(_aio_error(code), daemon=None, breaker=breaker)
        assert breaker.bypass() is False

    def test_non_grpc_exception_reraised_unchanged(self):
        breaker = AcceleratorBreaker()
        sentinel = ValueError("merge machinery bug")
        with pytest.raises(ValueError) as ei:
            handle_accelerator_error(sentinel, daemon=None, breaker=breaker)
        # Re-raised as-is, never masked as a fallback.
        assert ei.value is sentinel
        assert breaker.bypass() is False
