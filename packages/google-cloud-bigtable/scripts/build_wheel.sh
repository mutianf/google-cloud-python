#!/usr/bin/env bash
# Build a linux wheel that bundles the accelerator daemon, building the daemon
# from source in the google-cloud-go monorepo.
#
# The daemon source of truth is the monorepo:
#   https://github.com/googleapis/google-cloud-go
#     -> bigtable/internal/accelerator/cmd   (module: cloud.google.com/go/bigtable)
#
# This script shallow-, sparse-, blobless-clones just the `bigtable/` submodule
# at ${ACCEL_GO_REF}, cross-compiles the command for the target platform, drops
# the binary at
#   google/cloud/bigtable/data/_accelerator/bin/accelerator
# and then builds the platform-tagged wheel.
#
# We build from source rather than `go install ...@version`: the command lives
# in an `internal/` directory of the separate `cloud.google.com/go/bigtable`
# submodule and is not part of any tagged release, so `go install pkg@version`
# cannot resolve it.
#
# Requirements on the build host: git, go, file, python3 (with `build`).
#
# Overridable via env:
#   ACCEL_GO_REPO  git URL of the monorepo
#                  (default: https://github.com/googleapis/google-cloud-go)
#   ACCEL_GO_REF   branch, tag, or commit SHA to build from (default: main)
#   GOOS / GOARCH  cross-compile target (default: linux / amd64)
#
# NOTE on version lockstep: the daemon should be built from the monorepo ref
# that matches this package's version. `main` is the default for development;
# release builds should pin ACCEL_GO_REF to the corresponding tag/SHA.
#
# Output: dist/google_cloud_bigtable-<version>-py3-none-<plat>.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${PKG_ROOT}/google/cloud/bigtable/data/_accelerator/bin"
BIN_PATH="${BIN_DIR}/accelerator"

ACCEL_GO_REPO="${ACCEL_GO_REPO:-https://github.com/googleapis/google-cloud-go}"
ACCEL_GO_REF="${ACCEL_GO_REF:-main}"
GOOS="${GOOS:-linux}"
GOARCH="${GOARCH:-amd64}"
# Package path of the daemon command, relative to the `bigtable/` module root.
ACCEL_CMD_PKG="./internal/accelerator/cmd"

# This script produces linux wheels only. macOS/Windows wheels are built by the
# CI matrix with the appropriate GOOS and a matching cibuildwheel platform.
if [[ "${GOOS}" != "linux" ]]; then
    echo "ERROR: build_wheel.sh only builds linux wheels (got GOOS=${GOOS})." >&2
    exit 1
fi

# Map GOARCH -> manylinux platform tag + the `file(1)` arch string we expect.
case "${GOARCH}" in
    amd64) PLAT_NAME="manylinux_2_17_x86_64";  EXPECTED_ARCH="x86-64" ;;
    arm64) PLAT_NAME="manylinux_2_17_aarch64"; EXPECTED_ARCH="aarch64" ;;
    *)
        echo "ERROR: unsupported GOARCH=${GOARCH} (expected amd64 or arm64)." >&2
        exit 1
        ;;
esac

for tool in git go file python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "ERROR: required tool '${tool}' not found on PATH." >&2
        exit 1
    fi
done

# --- Fetch the daemon source from the monorepo -----------------------------
WORKDIR="$(mktemp -d -t bt-accel-build.XXXXXX)"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

SRC="${WORKDIR}/google-cloud-go"
echo "Fetching ${ACCEL_GO_REPO} @ ${ACCEL_GO_REF} (sparse: bigtable/) ..."
# init + fetch (rather than `clone --branch`) so ACCEL_GO_REF may be a branch,
# tag, or commit SHA. Blobless + sparse keeps the download to just `bigtable/`.
git -C "${WORKDIR}" init -q "google-cloud-go"
git -C "${SRC}" remote add origin "${ACCEL_GO_REPO}"
git -C "${SRC}" sparse-checkout init --cone
git -C "${SRC}" sparse-checkout set bigtable
git -C "${SRC}" fetch --depth 1 --filter=blob:none origin "${ACCEL_GO_REF}"
git -C "${SRC}" checkout -q FETCH_HEAD

CMD_DIR="${SRC}/bigtable/internal/accelerator/cmd"
if [[ ! -d "${CMD_DIR}" ]]; then
    echo "ERROR: accelerator command not found at bigtable/internal/accelerator/cmd" >&2
    echo "  in ${ACCEL_GO_REPO} @ ${ACCEL_GO_REF}." >&2
    exit 1
fi

# --- Build the daemon ------------------------------------------------------
mkdir -p "${BIN_DIR}"
echo "Building accelerator daemon (GOOS=${GOOS} GOARCH=${GOARCH}) ..."
(
    cd "${SRC}/bigtable"
    # CGO off -> fully static binary; -trimpath + -s -w for a smaller,
    # reproducible artifact with no local paths embedded.
    GOOS="${GOOS}" GOARCH="${GOARCH}" CGO_ENABLED=0 \
        go build -trimpath -ldflags="-s -w" -o "${BIN_PATH}" "${ACCEL_CMD_PKG}"
)

# Verify the binary matches the intended target -- catches a misconfigured
# GOOS/GOARCH before the bytes get sealed into a mislabeled wheel.
FILE_DESC="$(file -b "${BIN_PATH}")"
if [[ "${FILE_DESC}" != ELF*64-bit*"${EXPECTED_ARCH}"* ]]; then
    echo "ERROR: ${BIN_PATH} is not a linux/${GOARCH} ELF binary." >&2
    echo "  file reports: ${FILE_DESC}" >&2
    exit 1
fi

# --- Build the wheel -------------------------------------------------------
cd "${PKG_ROOT}"

# Clean any prior outputs so we don't ship a stale wheel.
rm -rf build dist

python3 -m build \
    --wheel \
    --config-setting=--build-option=--plat-name="${PLAT_NAME}"

echo
echo "Built wheel(s):"
ls -lh dist/
