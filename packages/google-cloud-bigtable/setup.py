# -*- coding: utf-8 -*-
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
#
import io
import json
import os
import re

import setuptools  # type: ignore


class _BinaryDistribution(setuptools.Distribution):
    """Forces bdist_wheel to emit a platform-tagged wheel instead of
    py3-none-any. The wheel ships the prebuilt accelerator daemon binary
    (google/cloud/bigtable/data/_accelerator/bin/accelerator) when present.
    Use --plat-name on bdist_wheel to set the actual platform tag.
    """

    def has_ext_modules(self):  # noqa: D401 - setuptools API
        return True


# Name of the provenance file dropped next to the bundled daemon so each wheel
# records which daemon binary it shipped (see _BDistWheel.run below).
_BUILD_INFO_NAME = "build_info.json"
_ACCEL_BIN_RELPATH = "google/cloud/bigtable/data/_accelerator/bin"


# The bundled binary is a standalone executable, not a CPython extension
# module — so the wheel's Python+ABI tag should be (py3, none), not
# (cp311, cp311). One linux/amd64 wheel works for any Python 3.x interpreter.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class _BDistWheel(_bdist_wheel):
        # Configuration hooks an external build script passes in via
        #   python -m build \
        #     --config-setting=--build-option=--wheel-version=<pep440> \
        #     --config-setting=--build-option=--go-binary-source=<url> \
        #     --config-setting=--build-option=--go-binary-version=<ref> \
        #     --config-setting=--build-option=--plat-name=<tag>
        # The external script fetches and stages the prebuilt daemon into
        # bin/accelerator[.exe] before invoking the build; setup.py only bundles
        # it. --wheel-version overrides the version stamped on the wheel;
        # --go-binary-source/--go-binary-version are recorded as build
        # provenance (not used to fetch or build anything here).
        user_options = _bdist_wheel.user_options + [
            (
                "wheel-version=",
                None,
                "PEP 440 version to stamp on the wheel "
                "(default: __version__ from gapic_version.py).",
            ),
            (
                "go-binary-source=",
                None,
                "Source the bundled accelerator daemon was fetched from "
                "(recorded as build provenance).",
            ),
            (
                "go-binary-version=",
                None,
                "Version of the bundled accelerator daemon "
                "(recorded as build provenance).",
            ),
        ]

        def initialize_options(self):
            super().initialize_options()
            self.wheel_version = None
            self.go_binary_source = None
            self.go_binary_version = None

        def finalize_options(self):
            # Apply the version override before super() reads metadata to build
            # the dist-info and the wheel filename.
            if self.wheel_version:
                self.distribution.metadata.version = self.wheel_version
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            return ("py3", "none", plat)

        def run(self):
            # Write provenance before build_py copies package data so it lands
            # in the wheel alongside the daemon binary.
            self._write_build_info()
            super().run()

        def _write_build_info(self):
            here = os.path.abspath(os.path.dirname(__file__))
            bin_dir = os.path.join(here, *_ACCEL_BIN_RELPATH.split("/"))
            # Only stamp provenance when a daemon binary is actually staged for
            # bundling. `packages` is computed at import time (before this runs);
            # creating build_info.json in an otherwise-empty/absent bin/ would
            # materialize a data file in a package directory that wasn't declared
            # then, which setuptools rejects as an ambiguous configuration. A
            # provenance record for a binary we aren't shipping is meaningless
            # anyway.
            if not any(
                os.path.isfile(os.path.join(bin_dir, name))
                for name in ("accelerator", "accelerator.exe")
            ):
                return
            info = {
                "wheel_version": self.distribution.metadata.version,
                "go_binary_source": self.go_binary_source or "",
                "go_binary_version": self.go_binary_version or "",
            }
            with open(os.path.join(bin_dir, _BUILD_INFO_NAME), "w") as fp:
                json.dump(info, fp, indent=2, sort_keys=True)
                fp.write("\n")

    _cmdclass = {"bdist_wheel": _BDistWheel}
except ImportError:
    _cmdclass = {}

package_root = os.path.abspath(os.path.dirname(__file__))

name = "google-cloud-bigtable"


description = "Google Cloud Bigtable API client library"

version = None

with open(os.path.join(package_root, "google/cloud/bigtable/gapic_version.py")) as fp:
    # Accept any PEP 440 version string (pre-releases, local segments, etc.),
    # not just bare X.Y.Z.
    match = re.search(r'__version__\s*=\s*"([^"]+)"', fp.read())
    assert match, "could not find __version__ in gapic_version.py"
    version = match.group(1)

if version[0] == "0":
    release_status = "Development Status :: 4 - Beta"
else:
    release_status = "Development Status :: 5 - Production/Stable"

dependencies = [
    "google-api-core[grpc] >= 2.25.0, <3.0.0",
    # Exclude incompatible versions of `google-auth`
    # See https://github.com/googleapis/google-cloud-python/issues/12364
    "google-auth >= 2.14.1, <3.0.0,!=2.24.0,!=2.25.0",
    "grpcio >= 1.59.0, < 2.0.0",
    "grpcio >= 1.75.1, < 2.0.0; python_version >= '3.14'",
    "proto-plus >= 1.26.1, <2.0.0",
    "protobuf >= 6.33.5, < 8.0.0",
    "google-cloud-core >= 2.0.0, <3.0.0",
    "grpc-google-iam-v1 >= 0.14.2, <1.0.0",
    "google-crc32c>=1.6.0, < 2.0.0",
]
extras = {
    "libcst": "libcst >= 0.2.5",
}

url = "https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-bigtable"

package_root = os.path.abspath(os.path.dirname(__file__))

readme_filename = os.path.join(package_root, "README.rst")
with io.open(readme_filename, encoding="utf-8") as readme_file:
    readme = readme_file.read()

packages = [
    package
    for package in setuptools.find_namespace_packages()
    if package.startswith("google")
]

# The accelerator daemon binary ships in a `bin/` directory whose name is a
# valid Python identifier. When an external build script stages the binary
# there, `find_namespace_packages()` normally discovers it -- but depending on
# the build tool (isolated sdist->wheel, git export, staging order) discovery
# can miss `bin` while MANIFEST.in still bundles the binary, and setuptools
# then rejects the build as an ambiguous "importable package absent from
# packages" configuration. Declaring it explicitly makes that deterministic.
#
# But only declare it when the directory actually exists: a pure build with no
# staged binary (e.g. the base wheel, or a checkout that doesn't carry the
# placeholder dir) has no `bin/`, and naming a nonexistent package directory
# makes setuptools fail with "package directory ... does not exist". Gating on
# existence satisfies both cases -- binary staged => dir present => declared =>
# no ambiguity; no binary => dir absent => omitted => nothing to bundle anyway.
_accelerator_bin_pkg = "google.cloud.bigtable.data._accelerator.bin"
_accelerator_bin_dir = os.path.join(
    package_root, *"google/cloud/bigtable/data/_accelerator/bin".split("/")
)
if os.path.isdir(_accelerator_bin_dir) and _accelerator_bin_pkg not in packages:
    packages.append(_accelerator_bin_pkg)

setuptools.setup(
    name=name,
    version=version,
    description=description,
    long_description=readme,
    author="Google LLC",
    author_email="googleapis-packages@google.com",
    license="Apache-2.0",
    url=url,
    classifiers=[
        release_status,
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: POSIX :: Linux",
        "Topic :: Internet",
    ],
    platforms="Linux",
    packages=packages,
    package_data={
        # Bundle the prebuilt accelerator daemon (staged at build time) and its
        # provenance manifest. Empty when no binary is staged (pure sdist).
        _accelerator_bin_pkg: [
            "accelerator",
            "accelerator.exe",
            "build_info.json",
        ],
    },
    python_requires=">=3.10",
    install_requires=dependencies,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            # Standalone YCSB-style benchmark driver for the data (V3) client,
            # bundled in the wheel so it runs without a source checkout.
            "bigtable-ycsb = google.cloud.bigtable.data._benchmarks.ycsb_perf:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    distclass=_BinaryDistribution,
    cmdclass=_cmdclass,
)
