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

//! GPU-resident sparse voxel map: 8x8x8 bricks in a fixed pool, found through
//! an open-addressing hash table from brick coordinate to pool slot. A frame
//! uploads only its points and sensor origin.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::gpu::{groups_1d, Gpu, PassTimer};

pub type VoxelKey = (i32, i32, i32);

const BRICK_VOXELS: u64 = 512;
const MOMENT_WORDS: u64 = 10;
const STAMPS_PER_VOXEL: u64 = 3;
const FINE_WORDS: u64 = 2;
const LISTS: u64 = 5;
const COUNTERS: u64 = 8;
/// Indirect argument slots: the five lists, then the brick pool.
const DISPATCH_SLOTS: u64 = 6;
const POOL_DISPATCH_SLOT: u64 = 5;
const FIXED_SCALE: f64 = (1u64 << 16) as f64;
const EMPTY_KEY: u32 = u32::MAX;
const S_EXISTS: u32 = 1;
const S_STALE: u32 = 2;
const S_PLANAR: u32 = 8;
/// Brick coordinates the 32-bit table key can hold, per axis, half-open.
const BRICK_RANGE_XY: (i32, i32) = (-1024, 1023);
const BRICK_RANGE_Z: (i32, i32) = (-512, 511);

const MODE_TRACE: u32 = 0;
const MODE_SEED: u32 = 1;
const MODE_INGEST: u32 = 2;

/// The CPU ray tracer's update parameters, plus the GPU pool sizes.
#[derive(Clone, Debug)]
pub struct MapConfig {
    pub voxel_size: f32,
    pub fine_divisor: u32,
    pub max_range: f32,
    pub ray_subsample: u32,
    pub shadow_depth: f32,
    pub grace_depth: f32,
    pub min_health: i32,
    pub max_health: i32,
    pub graze_cos: f32,
    /// Bricks the pool holds; voxel buffers are sized to this.
    pub brick_capacity: u32,
    /// Most points one frame may upload.
    pub max_points: u32,
    /// Entries in each per-frame work list (hits, fits, misses, ...).
    pub list_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MapParams {
    origin: [f32; 4],
    voxel_size: f32,
    inv_voxel: f32,
    shadow_depth: f32,
    grace_depth: f32,
    graze_cos: f32,
    min_health: i32,
    max_health: i32,
    fine_divisor: u32,
    point_count: u32,
    ray_step: u32,
    frame: u32,
    mode: u32,
    table_mask: u32,
    brick_capacity: u32,
    list_capacity: u32,
    trace_pass: u32,
}

/// One voxel read back from the GPU, moments in meters about its center.
#[derive(Clone, Debug)]
pub struct VoxelSnapshot {
    pub key: VoxelKey,
    pub health: i32,
    pub num_pts: u32,
    pub sum: [f64; 3],
    /// xx, xy, xz, yy, yz, zz.
    pub m2: [f64; 6],
    pub fine: u64,
    pub stale: bool,
    pub normal: Option<[f32; 3]>,
}

/// Per-frame list overflow and pool use, read back on demand.
#[derive(Clone, Debug)]
pub struct MapStats {
    pub bricks: u32,
    pub hits: u32,
    pub fits: u32,
    pub misses: u32,
    pub changed: u32,
    /// Bit per list that overflowed, 256 table full, 512 brick pool full.
    pub overflow: u32,
}

pub(crate) struct MapBuffers {
    pub params: wgpu::Buffer,
    /// The params the second, miss-listing walk reads.
    pub miss_params: wgpu::Buffer,
    pub table_keys: wgpu::Buffer,
    pub table_slots: wgpu::Buffer,
    pub brick_coords: wgpu::Buffer,
    pub counters: wgpu::Buffer,
    pub lists: wgpu::Buffer,
    pub points: wgpu::Buffer,
    pub voxel_state: wgpu::Buffer,
    pub health: wgpu::Buffer,
    pub stamps: wgpu::Buffer,
    pub fine: wgpu::Buffer,
    pub moments: wgpu::Buffer,
    pub next_fit: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub dispatch: wgpu::Buffer,
}

struct Pipelines {
    prepare: wgpu::ComputePipeline,
    insert_bricks: wgpu::ComputePipeline,
    assign_bricks: wgpu::ComputePipeline,
    mark_hits: wgpu::ComputePipeline,
    apply_hits: wgpu::ComputePipeline,
    accumulate: wgpu::ComputePipeline,
    settle_hits: wgpu::ComputePipeline,
    apply_misses: wgpu::ComputePipeline,
    mark_stale: wgpu::ComputePipeline,
    trace_rays: wgpu::ComputePipeline,
    fit_normals: wgpu::ComputePipeline,
}

pub struct BrickMap {
    gpu: Arc<Gpu>,
    pub config: MapConfig,
    pub(crate) buffers: MapBuffers,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) bind_group: wgpu::BindGroup,
    miss_bind_group: wgpu::BindGroup,
    dispatch_group: wgpu::BindGroup,
    pipelines: Pipelines,
    table_mask: u32,
    frame: u32,
    pub timer: Option<PassTimer>,
}

pub(crate) const MAP_COMMON: &str = include_str!("shaders/map_common.wgsl");

/// The 14 map bindings: params uniform, then the storage buffers in order.
pub(crate) const MAP_BINDINGS: u32 = 14;

impl BrickMap {
    /// Bytes each voxel holds across the per-voxel buffers.
    pub const VOXEL_BYTES: u64 = 4 * (1 + 1 + STAMPS_PER_VOXEL + FINE_WORDS + MOMENT_WORDS + 1 + 1);

    /// The largest brick pool whose widest buffer, the moments, fits one
    /// storage binding on this device.
    pub fn max_brick_capacity(gpu: &Gpu) -> u32 {
        let limits = gpu.device.limits();
        let largest = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        (largest / (4 * MOMENT_WORDS * BRICK_VOXELS)).min(u32::MAX as u64 / BRICK_VOXELS) as u32
    }

    /// Fails, without touching the device, when a buffer the config asks for
    /// exceeds this device's binding or buffer limits.
    pub fn new(gpu: Arc<Gpu>, config: MapConfig) -> Result<Self, String> {
        let cap = config.brick_capacity as u64;
        let voxels = cap * BRICK_VOXELS;
        // Half-full at capacity keeps linear probes short.
        let table_entries = (2 * cap).next_power_of_two();
        let list_cap = config.list_capacity as u64;
        let limits = gpu.device.limits();
        let largest = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        for (label, bytes) in [
            ("moments", 4 * MOMENT_WORDS * voxels),
            ("lists", 4 * LISTS * list_cap),
            ("points", 16 * config.max_points as u64),
            ("table", 4 * table_entries),
        ] {
            if bytes > largest {
                return Err(format!(
                    "brick map {label} buffer needs {bytes} bytes but {} allows {largest} per binding; \
                     lower brick_capacity to at most {} (or max_points / list_capacity)",
                    gpu.adapter_info.name,
                    Self::max_brick_capacity(&gpu)
                ));
            }
        }
        let buffers = MapBuffers {
            params: gpu.uniform("map params", std::mem::size_of::<MapParams>() as u64),
            miss_params: gpu.uniform("miss params", std::mem::size_of::<MapParams>() as u64),
            table_keys: gpu.storage("table keys", 4 * table_entries),
            table_slots: gpu.storage("table slots", 4 * table_entries),
            brick_coords: gpu.storage("brick coords", 16 * cap),
            counters: gpu.storage("counters", 4 * COUNTERS),
            lists: gpu.storage("lists", 4 * LISTS * list_cap),
            points: gpu.storage("points", 16 * config.max_points as u64),
            voxel_state: gpu.storage("voxel state", 4 * voxels),
            health: gpu.storage("health", 4 * voxels),
            stamps: gpu.storage("stamps", 4 * STAMPS_PER_VOXEL * voxels),
            fine: gpu.storage("fine", 4 * FINE_WORDS * voxels),
            moments: gpu.storage("moments", 4 * MOMENT_WORDS * voxels),
            next_fit: gpu.storage("next fit", 4 * voxels),
            normals: gpu.storage("normals", 4 * voxels),
            dispatch: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dispatch args"),
                size: 12 * DISPATCH_SLOTS,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
        };
        gpu.queue.write_buffer(
            &buffers.table_keys,
            0,
            bytemuck::cast_slice(&vec![EMPTY_KEY; table_entries as usize]),
        );

        let layout = gpu.layout("map", MAP_BINDINGS, &[0], false);
        let dispatch_layout = gpu.layout("dispatch", 1, &[], false);
        let b = &buffers;
        let storage = [
            &b.table_keys,
            &b.table_slots,
            &b.brick_coords,
            &b.counters,
            &b.lists,
            &b.points,
            &b.voxel_state,
            &b.health,
            &b.stamps,
            &b.fine,
            &b.moments,
            &b.next_fit,
            &b.normals,
        ];
        let with_params = |params: &wgpu::Buffer| {
            let all: Vec<&wgpu::Buffer> = std::iter::once(params).chain(storage).collect();
            gpu.bind_group("map", &layout, &all)
        };
        let bind_group = with_params(&b.params);
        let miss_bind_group = with_params(&b.miss_params);
        let dispatch_group = gpu.bind_group("dispatch", &dispatch_layout, &[&b.dispatch]);

        let ingest = gpu.module(
            "ingest",
            &[MAP_COMMON, include_str!("shaders/ingest.wgsl")].concat(),
        );
        let trace = gpu.module(
            "trace",
            &[MAP_COMMON, include_str!("shaders/raytrace.wgsl")].concat(),
        );
        let fit = gpu.module(
            "fit",
            &[MAP_COMMON, include_str!("shaders/fit.wgsl")].concat(),
        );
        let map_only = [&layout];
        let pipelines = Pipelines {
            prepare: gpu.pipeline(&ingest, "prepare_dispatch", &[&layout, &dispatch_layout]),
            insert_bricks: gpu.pipeline(&ingest, "insert_bricks", &map_only),
            assign_bricks: gpu.pipeline(&ingest, "assign_bricks", &map_only),
            mark_hits: gpu.pipeline(&ingest, "mark_hits", &map_only),
            apply_hits: gpu.pipeline(&ingest, "apply_hits", &map_only),
            accumulate: gpu.pipeline(&ingest, "accumulate", &map_only),
            settle_hits: gpu.pipeline(&ingest, "settle_hits", &map_only),
            apply_misses: gpu.pipeline(&ingest, "apply_misses", &map_only),
            mark_stale: gpu.pipeline(&ingest, "mark_stale", &map_only),
            trace_rays: gpu.pipeline(&trace, "trace_rays", &map_only),
            fit_normals: gpu.pipeline(&fit, "fit_normals", &map_only),
        };

        Ok(Self {
            gpu,
            config,
            buffers,
            layout,
            bind_group,
            miss_bind_group,
            dispatch_group,
            pipelines,
            table_mask: (table_entries - 1) as u32,
            frame: 0,
            timer: None,
        })
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// One lidar frame: hits, ray-traced clearing, health and moments. The
    /// same filter as the CPU update_map runs while packing the upload.
    pub fn integrate(&mut self, origin: (f32, f32, f32), points: &[(f32, f32, f32)]) {
        let packed = self.pack(points, Some(origin));
        self.run(origin, &packed, MODE_TRACE);
    }

    /// `integrate` without the clearing walk: occupancy, health and moments only.
    pub fn ingest(&mut self, origin: (f32, f32, f32), points: &[(f32, f32, f32)]) {
        let packed = self.pack(points, Some(origin));
        self.run(origin, &packed, MODE_INGEST);
    }

    /// The CPU seed_points: create absent voxels at seed health, without rays.
    pub fn seed(&mut self, points: &[(f32, f32, f32)]) {
        let packed = self.pack(points, None);
        self.run((0.0, 0.0, 0.0), &packed, MODE_SEED);
    }

    /// Points the map can hold, as vec4s. With an origin, also the CPU's
    /// nonzero and max-range filter.
    fn pack(&self, points: &[(f32, f32, f32)], origin: Option<(f32, f32, f32)>) -> Vec<[f32; 4]> {
        let inv = 1.0 / self.config.voxel_size;
        let max_range_sq = if self.config.max_range > 0.0 {
            self.config.max_range * self.config.max_range
        } else {
            f32::INFINITY
        };
        let brick_ok = |c: f32, (lo, hi): (i32, i32)| {
            let b = ((c * inv).floor() as i32) >> 3;
            (c * inv).is_finite() && b >= lo && b < hi
        };
        points
            .iter()
            .filter(|&&(x, y, z)| {
                if !(brick_ok(x, BRICK_RANGE_XY)
                    && brick_ok(y, BRICK_RANGE_XY)
                    && brick_ok(z, BRICK_RANGE_Z))
                {
                    return false;
                }
                origin.is_none_or(|o| {
                    let (dx, dy, dz) = (x - o.0, y - o.1, z - o.2);
                    let d2 = dx * dx + dy * dy + dz * dz;
                    d2 > 0.0 && d2 <= max_range_sq
                })
            })
            .map(|&(x, y, z)| [x, y, z, 0.0])
            .collect()
    }

    fn run(&mut self, origin: (f32, f32, f32), packed: &[[f32; 4]], mode: u32) {
        assert!(
            packed.len() <= self.config.max_points as usize,
            "frame of {} points exceeds max_points {}",
            packed.len(),
            self.config.max_points
        );
        self.frame += 1;
        let gpu = Arc::clone(&self.gpu);
        let c = &self.config;
        let params = MapParams {
            origin: [origin.0, origin.1, origin.2, 0.0],
            voxel_size: c.voxel_size,
            inv_voxel: 1.0 / c.voxel_size,
            shadow_depth: c.shadow_depth,
            grace_depth: c.grace_depth,
            graze_cos: c.graze_cos,
            min_health: c.min_health,
            max_health: c.max_health,
            fine_divisor: if c.fine_divisor >= 2 {
                c.fine_divisor
            } else {
                0
            },
            point_count: packed.len() as u32,
            ray_step: c.ray_subsample.max(1),
            frame: self.frame,
            mode,
            table_mask: self.table_mask,
            brick_capacity: c.brick_capacity,
            list_capacity: c.list_capacity,
            trace_pass: 0,
        };
        gpu.queue
            .write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&params));
        let miss_params = MapParams {
            trace_pass: 1,
            ..params
        };
        gpu.queue.write_buffer(
            &self.buffers.miss_params,
            0,
            bytemuck::bytes_of(&miss_params),
        );
        gpu.queue
            .write_buffer(&self.buffers.points, 0, bytemuck::cast_slice(packed));
        // List counters restart each frame; the brick count carries over.
        gpu.queue
            .write_buffer(&self.buffers.counters, 4, bytemuck::cast_slice(&[0u32; 5]));

        let point_groups = groups_1d(packed.len() as u32);
        let rays = (packed.len() as u32).div_ceil(params.ray_step);
        let ray_groups = groups_1d(rays);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());

        self.pass(&mut encoder, "bricks+hits", |map, pass| {
            map.dispatch(pass, &map.pipelines.insert_bricks, point_groups);
            map.prepare(pass);
            map.dispatch_list(pass, &map.pipelines.assign_bricks, 0);
            map.dispatch(pass, &map.pipelines.mark_hits, point_groups);
            map.prepare(pass);
        });

        if mode == MODE_TRACE {
            self.pass(&mut encoder, "trace walk", |map, pass| {
                map.dispatch(pass, &map.pipelines.trace_rays, ray_groups);
                map.prepare(pass);
            });
            self.pass(&mut encoder, "trace fit", |map, pass| {
                map.dispatch_list(pass, &map.pipelines.fit_normals, 2);
            });
            self.pass(&mut encoder, "trace walk 2", |map, pass| {
                pass.set_bind_group(0, &map.miss_bind_group, &[]);
                map.dispatch(pass, &map.pipelines.trace_rays, ray_groups);
                map.prepare(pass);
            });
        }

        self.pass(&mut encoder, "update", |map, pass| {
            map.dispatch_list(pass, &map.pipelines.apply_hits, 1);
            map.dispatch(pass, &map.pipelines.accumulate, point_groups);
            map.dispatch_list(pass, &map.pipelines.settle_hits, 1);
            map.dispatch_list(pass, &map.pipelines.apply_misses, 3);
            map.prepare(pass);
            map.dispatch_list(pass, &map.pipelines.mark_stale, 4);
        });
        if let Some(timer) = &self.timer {
            timer.resolve(&mut encoder);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    fn pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        body: impl FnOnce(&Self, &mut wgpu::ComputePass),
    ) {
        let timestamp_writes = self.timer.as_mut().and_then(|t| t.next(label));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        body(self, &mut pass);
    }

    fn dispatch(
        &self,
        pass: &mut wgpu::ComputePass,
        pipeline: &wgpu::ComputePipeline,
        (x, y): (u32, u32),
    ) {
        if x == 0 {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn dispatch_list(
        &self,
        pass: &mut wgpu::ComputePass,
        pipeline: &wgpu::ComputePipeline,
        slot: u64,
    ) {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups_indirect(&self.buffers.dispatch, 12 * slot);
    }

    /// Refresh the indirect sizes of every list and the brick pool.
    pub(crate) fn prepare(&self, pass: &mut wgpu::ComputePass) {
        pass.set_pipeline(&self.pipelines.prepare);
        pass.set_bind_group(1, &self.dispatch_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// One thread per pool voxel, sized on the GPU by the last `prepare`.
    pub(crate) fn dispatch_pool(
        &self,
        pass: &mut wgpu::ComputePass,
        pipeline: &wgpu::ComputePipeline,
    ) {
        self.dispatch_list(pass, pipeline, POOL_DISPATCH_SLOT);
    }

    pub fn finish(&self) {
        self.gpu.wait();
    }

    pub fn stats(&self) -> MapStats {
        let c: Vec<u32> = self.gpu.read(&self.buffers.counters, COUNTERS as usize);
        MapStats {
            bricks: c[0],
            hits: c[2],
            fits: c[3],
            misses: c[4],
            changed: c[5],
            overflow: c[6],
        }
    }

    /// Every existing voxel. Reads the whole used pool back.
    pub fn snapshot(&self) -> Vec<VoxelSnapshot> {
        let bricks = self.stats().bricks.min(self.config.brick_capacity) as usize;
        let voxels = bricks * BRICK_VOXELS as usize;
        let gpu = &self.gpu;
        let coords: Vec<[i32; 4]> = gpu.read(&self.buffers.brick_coords, bricks);
        let state: Vec<u32> = gpu.read(&self.buffers.voxel_state, voxels);
        let health: Vec<i32> = gpu.read(&self.buffers.health, voxels);
        let fine: Vec<u32> = gpu.read(&self.buffers.fine, voxels * FINE_WORDS as usize);
        let moments: Vec<u32> = gpu.read(&self.buffers.moments, voxels * MOMENT_WORDS as usize);
        let normals: Vec<u32> = gpu.read(&self.buffers.normals, voxels);
        let vs = self.config.voxel_size as f64;
        let fixed = |v: usize, word: usize| {
            moments[v * MOMENT_WORDS as usize + 1 + word] as i32 as f64 / FIXED_SCALE
        };
        let mut out = Vec::new();
        for (v, &s) in state.iter().enumerate() {
            if s & S_EXISTS == 0 {
                continue;
            }
            let b = coords[v / BRICK_VOXELS as usize];
            let l = (v % BRICK_VOXELS as usize) as i32;
            let key = (
                (b[0] << 3) | (l >> 6),
                (b[1] << 3) | ((l >> 3) & 7),
                (b[2] << 3) | (l & 7),
            );
            out.push(VoxelSnapshot {
                key,
                health: health[v],
                num_pts: moments[v * MOMENT_WORDS as usize],
                sum: [0, 1, 2].map(|w| fixed(v, w) * vs),
                m2: [3, 4, 5, 6, 7, 8].map(|w| fixed(v, w) * vs * vs),
                fine: (fine[2 * v] as u64) | ((fine[2 * v + 1] as u64) << 32),
                stale: s & S_STALE != 0,
                normal: (s & S_PLANAR != 0).then(|| unpack_normal(normals[v])),
            });
        }
        out
    }

    /// Bytes the map holds on the GPU.
    pub fn gpu_bytes(&self) -> u64 {
        let b = &self.buffers;
        [
            &b.table_keys,
            &b.table_slots,
            &b.brick_coords,
            &b.lists,
            &b.points,
            &b.voxel_state,
            &b.health,
            &b.stamps,
            &b.fine,
            &b.moments,
            &b.next_fit,
            &b.normals,
        ]
        .iter()
        .map(|b| b.size())
        .sum()
    }
}

/// The shader's octahedral unpack, for snapshots.
fn unpack_normal(packed: u32) -> [f32; 3] {
    let snorm = |bits: u32| ((bits & 0xffff) as u16 as i16 as f32 / 32767.0).max(-1.0);
    let (qx, qy) = (snorm(packed), snorm(packed >> 16));
    let mut n = [qx, qy, 1.0 - qx.abs() - qy.abs()];
    let t = (-n[2]).max(0.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|c| c / len)
}
