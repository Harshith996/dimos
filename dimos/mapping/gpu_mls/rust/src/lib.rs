// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GPU-resident voxel mapping and MLS surface extraction on wgpu compute: a
//! port of the ray tracer's update_map and the MLS planner's surface stages.
//!
//! # Jetson (Vulkan)
//!
//! Build and run with the host toolchain, not `nix develop`. The nix shell's
//! glibc (2.39) cannot load the Tegra Vulkan driver, so wgpu silently falls
//! back to llvmpipe on the CPU:
//!
//! ```text
//! export PATH=$HOME/.cargo/bin:$PATH
//! cargo test --release && cargo run --release --example bench
//! ```
//!
//! Check the `adapter:` line the tests print: it must name the Orin, not
//! llvmpipe. Brick pools are bounded by the adapter's storage binding limit
//! (about 2 GB on Orin); see `BrickMap::max_brick_capacity`.

pub mod brick_map;
pub mod gpu;
pub mod surfaces;
pub mod synthetic;
