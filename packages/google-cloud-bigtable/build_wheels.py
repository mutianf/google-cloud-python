#!/usr/bin/env python3
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

"""Multi-platform wheel assembler with embedded Go accelerator binaries."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

DEFAULT_PLATFORM_BINARY_MAP: dict[str, str] = {
    "manylinux2014_x86_64": "accelerator_linux_x86_64",
    "manylinux2014_aarch64": "accelerator_linux_aarch64",
    "manylinux2014_i686": "accelerator_linux_386",
    "macosx_10_9_x86_64": "accelerator_darwin_x86_64",
    "macosx_11_0_arm64": "accelerator_darwin_arm64",
}


def load_platform_binary_map(
    bins_dir: str, explicit_json: str | None = None
) -> dict[str, str]:
  """Loads platform-to-binary mapping from targets.json if available."""
  json_path = explicit_json or os.path.join(bins_dir, "targets.json")
  if os.path.isfile(json_path):
    print(f"Loading platform binary mapping from manifest: {json_path}")
    try:
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      if "targets" in data and isinstance(data["targets"], dict):
        return data["targets"]
      if isinstance(data, dict):
        return data
    except Exception as e:  # pylint: disable=broad-except
      print(f"Warning: Failed to parse {json_path} ({e}); using defaults.")

  print(
      "No targets.json manifest found; using default built-in platform mapping."
  )
  return DEFAULT_PLATFORM_BINARY_MAP


def assemble_platform_wheel(
    base_wheel: str,
    output_wheel: str,
    platform_tag: str,
    binary_source: str,
) -> None:
  """Unpacks base wheel, embeds platform binary, updates tags and repacks."""
  with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(base_wheel, "r") as zf:
      zf.extractall(tmpdir)

    # 1. Update WHEEL metadata
    dist_info_dirs = [d for d in os.listdir(tmpdir) if d.endswith(".dist-info")]
    if not dist_info_dirs:
      raise ValueError(f"No .dist-info directory found in {base_wheel}")
    dist_info_path = os.path.join(tmpdir, dist_info_dirs[0])

    wheel_file = os.path.join(dist_info_path, "WHEEL")
    if os.path.exists(wheel_file):
      with open(wheel_file, "r", encoding="utf-8") as f:
        content = f.read()
      content = re.sub(
          r"Root-Is-Purelib:\s*true", "Root-Is-Purelib: false", content
      )
      content = re.sub(r"Tag:\s*.*", f"Tag: py3-none-{platform_tag}", content)
      with open(wheel_file, "w", encoding="utf-8") as f:
        f.write(content)

    # 2. Place binary at the exact expected location relative to _daemon.py
    accelerator_dirs: list[str] = []
    for root, _, files in os.walk(tmpdir):
      if ".dist-info" in root:
        continue
      if (
          os.path.basename(root) == "_accelerator"
          or "_daemon.py" in files
          or "daemon.py" in files
      ):
        accelerator_dirs.append(root)

    if not accelerator_dirs:
      default_accel = os.path.join(
          tmpdir, "google", "cloud", "bigtable", "data", "_accelerator"
      )
      os.makedirs(default_accel, exist_ok=True)
      accelerator_dirs.append(default_accel)

    for adir in accelerator_dirs:
      bin_dir = os.path.join(adir, "bin")
      os.makedirs(bin_dir, exist_ok=True)
      dst_accel = os.path.join(bin_dir, "accelerator")
      shutil.copy2(binary_source, dst_accel)
      os.chmod(dst_accel, 0o755)

    # 3. Recalculate RECORD checksums and file sizes
    record_file = os.path.join(dist_info_path, "RECORD")
    record_relpath = os.path.relpath(record_file, tmpdir).replace("\\", "/")
    records: list[tuple[str, str, int | str]] = []

    for root, _, files in sorted(os.walk(tmpdir)):
      for filename in sorted(files):
        fpath = os.path.join(root, filename)
        relpath = os.path.relpath(fpath, tmpdir).replace("\\", "/")
        if relpath == record_relpath:
          records.append((relpath, "", ""))
        else:
          with open(fpath, "rb") as f:
            data = f.read()
          digest = (
              base64.urlsafe_b64encode(hashlib.sha256(data).digest())
              .decode("ascii")
              .rstrip("=")
          )
          records.append((relpath, f"sha256={digest}", len(data)))

    with open(record_file, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerows(records)

    # 4. Pack into output wheel preserving executable permissions
    if os.path.exists(output_wheel):
      os.remove(output_wheel)

    with zipfile.ZipFile(
        output_wheel, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
      for root, _, files in sorted(os.walk(tmpdir)):
        for filename in sorted(files):
          fpath = os.path.join(root, filename)
          relpath = os.path.relpath(fpath, tmpdir)
          zinfo = zipfile.ZipInfo.from_file(fpath, relpath)
          st = os.stat(fpath)
          zinfo.external_attr = (st.st_mode & 0xFFFF) << 16
          with open(fpath, "rb") as f:
            zf.writestr(zinfo, f.read())

  print(f"Successfully assembled: {output_wheel}")


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Assemble multi-platform Python wheels with Go binaries."
  )
  parser.add_argument(
      "--dist-dir",
      required=True,
      help=(
          "Directory containing built distributions and where to write"
          " assembled wheels."
      ),
  )
  parser.add_argument(
      "--bins-dir",
      required=True,
      help="Directory containing downloaded Go accelerator binaries.",
  )
  parser.add_argument(
      "--targets-json",
      required=False,
      default=None,
      help="Optional path to targets.json manifest file.",
  )
  args = parser.parse_args()

  dist_dir = os.path.abspath(args.dist_dir)
  bins_dir = os.path.abspath(args.bins_dir)
  platform_binary_map = load_platform_binary_map(bins_dir, args.targets_json)

  whl_files = [
      os.path.join(dist_dir, f)
      for f in os.listdir(dist_dir)
      if f.endswith(".whl") and "-manylinux" not in f and "-macosx" not in f
  ]
  if not whl_files:
    whl_files = [
        os.path.join(dist_dir, f)
        for f in os.listdir(dist_dir)
        if f.endswith(".whl")
    ]
  if not whl_files:
    print(f"ERROR: No base wheel found in {dist_dir}", file=sys.stderr)
    return 1

  base_wheel = whl_files[0]
  base_filename = os.path.basename(base_wheel)
  pkg_prefix = re.sub(r"-[^-]+-[^-]+-[^-]+\.whl$", "", base_filename)

  print("=" * 60)
  print(f"Assembling multi-platform binary wheels from base: {base_filename}")
  print(f"Using Go binaries from: {bins_dir}")
  print("=" * 60)

  assembled_count = 0
  for platform_tag, bin_filename in platform_binary_map.items():
    bin_path = os.path.join(bins_dir, bin_filename)
    if not os.path.isfile(bin_path):
      print(
          f"Warning: Prebuilt binary '{bin_filename}' not found in {bins_dir}."
          f" Skipping {platform_tag}."
      )
      continue

    output_wheel = os.path.join(
        dist_dir, f"{pkg_prefix}-py3-none-{platform_tag}.whl"
    )
    print(f"Assembling wheel for '{platform_tag}' using '{bin_filename}'...")
    assemble_platform_wheel(
        base_wheel=base_wheel,
        output_wheel=output_wheel,
        platform_tag=platform_tag,
        binary_source=bin_path,
    )
    assembled_count += 1

  if assembled_count > 0:
    if "none-any.whl" in base_wheel or "linux_x86_64.whl" in base_wheel:
      print(f"Removing intermediate base wheel: {base_wheel}")
      try:
        os.remove(base_wheel)
      except OSError:
        pass

  print(
      f"Multi-platform wheel assembly completed ({assembled_count} wheels"
      " created)."
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
