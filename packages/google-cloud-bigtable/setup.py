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


# The bundled binary is a standalone executable, not a CPython extension
# module — so the wheel's Python+ABI tag should be (py3, none), not
# (cp311, cp311). One linux/amd64 wheel works for any Python 3.x interpreter.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class _BDistWheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            return ("py3", "none", plat)

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
    python_requires=">=3.10",
    install_requires=dependencies,
    extras_require=extras,
    include_package_data=True,
    zip_safe=False,
    distclass=_BinaryDistribution,
    cmdclass=_cmdclass,
)
