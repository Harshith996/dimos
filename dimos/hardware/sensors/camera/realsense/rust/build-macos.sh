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

set -euo pipefail
cd "$(dirname "$0")"

realsense_prefix="$(brew --prefix librealsense)"
realsense_version="$(brew list --versions librealsense | awk '{print $2}')"
pc_dir="$(mktemp -d)"
trap 'rm -f "$pc_dir/realsense2.pc"; rmdir "$pc_dir"' EXIT

# Homebrew librealsense 2.59.1 ships a Linux libdir in its pkg-config file.
# Supply the actual Homebrew paths to realsense-sys and this crate's build.rs.
cat > "$pc_dir/realsense2.pc" <<EOF
prefix=$realsense_prefix
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: realsense2
Description: RealSense SDK 2.0
Version: $realsense_version
Libs: -L\${libdir} -lrealsense2
Cflags: -I\${includedir}
EOF

PKG_CONFIG_PATH="$pc_dir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
    cargo build --release --locked --target-dir "$PWD/target"
