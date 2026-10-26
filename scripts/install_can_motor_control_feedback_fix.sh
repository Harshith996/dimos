#!/usr/bin/env bash
# Copyright 2026 Dimensional Inc.
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

# Build the 0.0.8 feedback decoder fix from the Dimensional upstream source.
# Run after provisioning .venv; no motors are accessed by this installer.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$project_dir/.venv/bin/python"
upstream_revision=ef5c082b94d77fbfbf50b4a73a41dca44f578acf
patch_file="$project_dir/dimos/hardware/whole_body/damiao/patches/can_motor_control_feedback.patch"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/dimos-can-feedback.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
source_dir="$build_dir/source"

if [[ ! -x "$python_bin" ]]; then
    echo "Provision $project_dir/.venv before running this installer." >&2
    exit 1
fi

git init --quiet "$source_dir"
git -C "$source_dir" remote add origin https://github.com/dimensionalOS/can-motor-control.git
git -C "$source_dir" fetch --quiet --depth 1 origin "$upstream_revision"
git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
git -C "$source_dir" apply --check "$patch_file"
git -C "$source_dir" apply "$patch_file"

# Match upstream's release process: its Rust workspace uses 0.1.0, while
# v0.0.8's published Python distribution overrides the dynamic version.
"$python_bin" - "$source_dir/crates/can-motor-control-py/pyproject.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = 'dynamic = ["version"]'
if text.count(old) != 1:
    raise RuntimeError("Upstream Python version metadata changed")
path.write_text(text.replace(old, 'version = "0.0.8"'))
PY

export CARGO_TARGET_DIR="$project_dir/target/can-motor-control-feedback"
cargo test --locked --manifest-path "$source_dir/Cargo.toml" \
    -p can-motor-damiao-codec -p can-motor-control
uv tool run --from 'maturin>=1.7,<2' maturin build --release --strip --locked \
    --manifest-path "$source_dir/crates/can-motor-control-py/Cargo.toml" \
    --interpreter "$python_bin" --out "$build_dir/wheels"
wheels=("$build_dir"/wheels/can_motor_control-*.whl)

# Running DimOS under sudo can leave root-owned bytecode caches. Preserve
# those outside site-packages so uv can replace the user-owned package.
"$python_bin" - <<'PY'
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

package = Path(sysconfig.get_path("purelib")) / "can_motor_control"
if any(not os.access(cache, os.W_OK) for cache in package.rglob("__pycache__")):
    backup = Path(tempfile.mkdtemp(prefix=".can_motor_control_old_cache-", dir=sys.prefix))
    package.rename(backup / "can_motor_control")
PY

uv pip install --python "$python_bin" --no-deps --reinstall "${wheels[0]}"

echo "Installed the tested Dimensional can-motor-control feedback fix into $project_dir/.venv."
