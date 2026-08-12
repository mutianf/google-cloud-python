#!/usr/bin/env bash
# Build the platform-tagged wheels that bundle the accelerator daemon, building
# the daemon from source in the google-cloud-go monorepo.
#
# The daemon source of truth is the monorepo:
#   https://github.com/googleapis/google-cloud-go
#     -> bigtable/internal/accelerator/cmd   (module: cloud.google.com/go/bigtable)
#
# This script shallow-, sparse-, blobless-clones just the `bigtable/` submodule
# at ${ACCEL_GO_REF}, then cross-compiles the command once per entry in TARGETS
# below. Go cross-compiles from a single host with CGO disabled, so the whole
# matrix (linux/darwin/windows) is built here without per-OS runners. For each
# target the binary is dropped at
#   google/cloud/bigtable/data/_accelerator/bin/accelerator[.exe]
# and a platform-tagged wheel is built around it.
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
#   TARGETS        space-separated "<goos> <goarch>" pairs to build (defaults to
#                  the full matrix below)
#
# NOTE on version lockstep: the daemon should be built from the monorepo ref
# that matches this package's version. `main` is the default for development;
# release builds should pin ACCEL_GO_REF to the corresponding tag/SHA.
#
# Output: dist/google_cloud_bigtable-<version>-py3-none-<plat>.whl (one per target)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${PKG_ROOT}/google/cloud/bigtable/data/_accelerator/bin"

ACCEL_GO_REPO="${ACCEL_GO_REPO:-https://github.com/googleapis/google-cloud-go}"
ACCEL_GO_REF="${ACCEL_GO_REF:-main}"
# Package path of the daemon command, relative to the `bigtable/` module root.
ACCEL_CMD_PKG="./internal/accelerator/cmd"

# Go build targets -> one wheel each. Cross-compiled from a single host.
TARGETS=(
  # Linux Builds
  "linux 386"
  "linux amd64"
  "linux arm64"

  # Mac Builds
  "darwin amd64"
  "darwin arm64"

  # Windows Builds
  "windows 386"
  "windows amd64"
)
# Allow the caller to override the matrix, e.g. TARGETS="linux amd64".
if [[ -n "${TARGETS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    read -ra TARGETS <<<"${TARGETS_OVERRIDE}"
fi

# Map "<goos> <goarch>" -> the wheel platform tag to stamp on the wheel.
plat_name_for() {
    case "$1" in
        "linux 386")     echo "manylinux_2_17_i686" ;;
        "linux amd64")   echo "manylinux_2_17_x86_64" ;;
        "linux arm64")   echo "manylinux_2_17_aarch64" ;;
        "darwin amd64")  echo "macosx_10_9_x86_64" ;;
        "darwin arm64")  echo "macosx_11_0_arm64" ;;
        "windows 386")   echo "win32" ;;
        "windows amd64") echo "win_amd64" ;;
        *) return 1 ;;
    esac
}

# Map "<goos> <goarch>" -> a distinctive glob of file(1)'s output, so we can
# confirm the cross-compiled bytes really match the intended target before they
# get sealed into a mislabeled wheel. file(1) is magic-based, so it identifies
# ELF/Mach-O/PE alike regardless of the host OS.
file_glob_for() {
    case "$1" in
        "linux 386")     echo "ELF*32-bit*80386*" ;;
        "linux amd64")   echo "ELF*64-bit*x86-64*" ;;
        "linux arm64")   echo "ELF*64-bit*aarch64*" ;;
        "darwin amd64")  echo "Mach-O*x86_64*" ;;
        "darwin arm64")  echo "Mach-O*arm64*" ;;
        "windows 386")   echo "PE32 *Intel 80386*" ;;
        "windows amd64") echo "PE32+*x86-64*" ;;
        *) return 1 ;;
    esac
}

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

# --- Build the matrix ------------------------------------------------------
cd "${PKG_ROOT}"

# Clean prior wheel outputs once, up front: a full run starts clean, but the
# wheels this run produces accumulate across targets in dist/.
rm -rf dist

# Every cross-compiled binary is kept here so a later target never clobbers an
# earlier one. The package tree's bin/ only ever holds the *current* target's
# binary at wheel-build time (MANIFEST bundles whatever is in bin/).
STAGE_DIR="${WORKDIR}/binaries"
mkdir -p "${STAGE_DIR}"

for target in "${TARGETS[@]}"; do
    read -r GOOS GOARCH <<<"${target}"
    if [[ -z "${GOOS}" || -z "${GOARCH}" ]]; then
        echo "ERROR: malformed target entry '${target}' (want '<goos> <goarch>')." >&2
        exit 1
    fi
    PLAT_NAME="$(plat_name_for "${target}")" || {
        echo "ERROR: no wheel platform tag known for target '${target}'." >&2
        exit 1
    }
    FILE_GLOB="$(file_glob_for "${target}")"
    EXT=""
    [[ "${GOOS}" == "windows" ]] && EXT=".exe"

    STAGED_BIN="${STAGE_DIR}/accelerator-${GOOS}-${GOARCH}${EXT}"
    echo "Building accelerator daemon (GOOS=${GOOS} GOARCH=${GOARCH}) ..."
    (
        cd "${SRC}/bigtable"
        # CGO off -> fully static binary; -trimpath + -s -w for a smaller,
        # reproducible artifact with no local paths embedded.
        GOOS="${GOOS}" GOARCH="${GOARCH}" CGO_ENABLED=0 \
            go build -trimpath -ldflags="-s -w" -o "${STAGED_BIN}" "${ACCEL_CMD_PKG}"
    )

    # Verify the binary matches the intended target -- catches a misconfigured
    # GOOS/GOARCH before the bytes get sealed into a mislabeled wheel.
    FILE_DESC="$(file -b "${STAGED_BIN}")"
    # shellcheck disable=SC2053  # glob match is intentional
    if [[ "${FILE_DESC}" != ${FILE_GLOB} ]]; then
        echo "ERROR: ${STAGED_BIN} does not look like a ${GOOS}/${GOARCH} binary." >&2
        echo "  file reports:      ${FILE_DESC}" >&2
        echo "  expected to match: ${FILE_GLOB}" >&2
        exit 1
    fi

    # Stage exactly this target's binary into the package tree, then build its
    # wheel. Drop any binary left by a previous iteration first, and clear the
    # setuptools build/ scratch dir, so the wheel bundles exactly one daemon.
    rm -rf build
    mkdir -p "${BIN_DIR}"
    rm -f "${BIN_DIR}/accelerator" "${BIN_DIR}/accelerator.exe"
    cp "${STAGED_BIN}" "${BIN_DIR}/accelerator${EXT}"

    python3 -m build \
        --wheel \
        --config-setting=--build-option=--plat-name="${PLAT_NAME}"
done

# Leave no stray binary behind in the (git-ignored) source tree.
rm -f "${BIN_DIR}/accelerator" "${BIN_DIR}/accelerator.exe"

echo
echo "Built wheel(s):"
ls -lh dist/
