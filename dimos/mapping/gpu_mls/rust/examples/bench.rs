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

//! CPU against GPU, per stage, on a 30 m x 30 m x 3 m synthetic room.
//! Run with `cargo run --release --example bench`.

use std::collections::BTreeMap;
use std::time::Instant;

use ahash::AHashSet;
use dimos_gpu_mls::brick_map::{BrickMap, MapConfig, VoxelKey};
use dimos_gpu_mls::gpu::{Gpu, PassTimer};
use dimos_gpu_mls::surfaces::{SurfaceConfig, SurfaceExtractor, Window};
use dimos_gpu_mls::synthetic::{Lcg, Scene};
use dimos_mls_planner::{surface_cells, wall_distances};
use dimos_voxel_ray_tracing::voxel_ray_tracer::{update_map, Config, VoxelMap};

const ROOM_M: f32 = 30.0;
const ROOM_HEIGHT_M: f32 = 3.0;
const FRAMES: usize = 30;
/// Frames before timing starts, so the map is populated and pipelines warm.
const WARMUP: usize = 5;
const REAL_MAP: &str = "go2_hongkong_office_twopass_map.pc2.lcm";

/// The xyz floats of an LCM-encoded sensor_msgs/PointCloud2.
fn pointcloud2_xyz(bytes: &[u8]) -> Vec<(f32, f32, f32)> {
    let cloud = lcm_msgs::sensor_msgs::PointCloud2::decode(bytes).expect("a PointCloud2");
    let offset = |name: &str| {
        cloud
            .fields
            .iter()
            .find(|f| f.name == name)
            .expect("xyz")
            .offset as usize
    };
    let (ox, oy, oz) = (offset("x"), offset("y"), offset("z"));
    let at =
        |p: usize, o: usize| f32::from_le_bytes(cloud.data[p + o..p + o + 4].try_into().unwrap());
    (0..(cloud.width * cloud.height) as usize)
        .map(|i| i * cloud.point_step as usize)
        .map(|p| (at(p, ox), at(p, oy), at(p, oz)))
        .filter(|p| p.0.is_finite() && p.1.is_finite() && p.2.is_finite())
        .collect()
}

fn cpu_config(voxel_size: f32) -> Config {
    Config {
        voxel_size,
        fine_divisor: 3,
        emit_fine: false,
        max_range: 30.0,
        ray_subsample: 1,
        shadow_depth: 0.1,
        grace_depth: 0.2,
        min_health: -1,
        max_health: 5,
        graze_cos: 0.7,
        support_min: 0,
        emit_every: 1,
        global_emit_every: 1,
        region_percentile: 95.0,
        world_frame: "world".to_string(),
        tf_match_tolerance_s: 0.1,
        max_cloud_age_s: 0.0,
        tf_wait_timeout_s: 0.05,
        worker_threads: 8,
        region_m: 4.0,
        viz_cell_m: 4.0,
        viz_emit_every: 0,
        viz_sweep_cells: 0,
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

#[derive(Default)]
struct Mean(BTreeMap<String, (f64, usize)>);

impl Mean {
    fn add(&mut self, key: &str, v: f64) {
        let e = self.0.entry(key.to_string()).or_default();
        e.0 += v;
        e.1 += 1;
    }

    fn get(&self, key: &str) -> f64 {
        self.0.get(key).map_or(f64::NAN, |(s, n)| s / *n as f64)
    }
}

fn main() {
    let gpu = Gpu::new().expect("a GPU");
    println!(
        "adapter {} ({:?}), timestamps {}, rayon threads {}",
        gpu.adapter_info.name,
        gpu.adapter_info.backend,
        gpu.timestamp_period.is_some(),
        rayon::current_num_threads()
    );
    let scene = Scene::room(ROOM_M, ROOM_HEIGHT_M);
    println!("| voxel | pts/frame | CPU update_map | GPU frame wall | GPU upload (host) | GPU bricks+hits | GPU walk 1 / fits / walk 2 | GPU update | map GPU MB |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut surface_rows = Vec::new();
    for voxel_size in [0.05f32, 0.08] {
        for points in [20_000usize, 50_000, 100_000] {
            let cfg = cpu_config(voxel_size);
            let mut rng = Lcg(3);
            let frames: Vec<_> = scene
                .trajectory(FRAMES, 0.4)
                .into_iter()
                .map(|o| (o, scene.scan(o, points, &mut rng, 0.01, 35.0)))
                .collect();
            let mut cpu = VoxelMap::default();
            let mut map = BrickMap::new(
                gpu.clone(),
                MapConfig {
                    voxel_size,
                    fine_divisor: cfg.fine_divisor,
                    max_range: cfg.max_range,
                    ray_subsample: 1,
                    shadow_depth: cfg.shadow_depth,
                    grace_depth: cfg.grace_depth,
                    min_health: cfg.min_health,
                    max_health: cfg.max_health,
                    graze_cos: cfg.graze_cos,
                    brick_capacity: 1 << 15,
                    max_points: points as u32 + 64,
                    list_capacity: 1 << 22,
                },
            )
            .expect("brick map fits the device");
            map.timer = PassTimer::new(&gpu, 16);
            let mut mean = Mean::default();
            for (i, (origin, pts)) in frames.iter().enumerate() {
                let t = Instant::now();
                update_map(&mut cpu, *origin, pts, &cfg);
                let cpu_ms = ms(t);

                let t = Instant::now();
                map.integrate(*origin, pts);
                let submit_ms = ms(t);
                map.finish();
                let wall_ms = ms(t);
                let passes = map
                    .timer
                    .as_mut()
                    .map(|tm| tm.take(&gpu))
                    .unwrap_or_default();
                if i < WARMUP {
                    continue;
                }
                mean.add("cpu", cpu_ms);
                mean.add("wall", wall_ms);
                mean.add("submit", submit_ms);
                for (label, t) in passes {
                    mean.add(label, t);
                }
                mean.add("fits", map.stats().fits as f64);
            }
            let stats = map.stats();
            println!(
                "| {voxel_size} | {}k | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} / {:.2} ({:.0} voxels) / {:.2} ms | {:.2} ms | {} (bricks {}) |",
                points / 1000,
                mean.get("cpu"),
                mean.get("wall"),
                mean.get("submit"),
                mean.get("bricks+hits"),
                mean.get("trace walk"),
                mean.get("trace fit"),
                mean.get("fits"),
                mean.get("trace walk 2"),
                mean.get("update"),
                map.gpu_bytes() / (1 << 20),
                stats.bricks,
            );
            assert_eq!(stats.overflow, 0, "lists or pool overflowed");
            if points == 100_000 {
                surface_rows.push(bench_surfaces(
                    &gpu,
                    &mut map,
                    voxel_size,
                    "synthetic 30x30x3",
                ));
            }
        }
    }
    let real = dimos_gpu_mls::data_dir().join(REAL_MAP);
    if let Ok(bytes) = std::fs::read(&real) {
        let cloud = pointcloud2_xyz(&bytes);
        for voxel_size in [0.05f32, 0.08] {
            let mut map = BrickMap::new(
                gpu.clone(),
                MapConfig {
                    voxel_size,
                    fine_divisor: 3,
                    max_range: 30.0,
                    ray_subsample: 1,
                    shadow_depth: 0.1,
                    grace_depth: 0.2,
                    min_health: -1,
                    max_health: 5,
                    graze_cos: 0.7,
                    brick_capacity: 1 << 15,
                    max_points: cloud.len() as u32,
                    list_capacity: 1 << 22,
                },
            )
            .expect("brick map fits the device");
            map.timer = PassTimer::new(&gpu, 16);
            map.seed(&cloud);
            map.finish();
            map.timer.as_mut().map(|tm| tm.take(&gpu));
            surface_rows.push(bench_surfaces(&gpu, &mut map, voxel_size, REAL_MAP));
        }
    } else {
        eprintln!(
            "skipping the real map: {} is missing; set DIMOS_DATA_DIR",
            real.display()
        );
    }
    println!();
    println!("| map | voxel | window | CPU surfaces | CPU wall dist | GPU plane fit | GPU standable | GPU hole fill | GPU planes | GPU wall dist | GPU wall clock | surface readback |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for row in surface_rows {
        println!("{row}");
    }
}

fn bench_surfaces(
    gpu: &std::sync::Arc<Gpu>,
    map: &mut BrickMap,
    voxel_size: f32,
    label: &str,
) -> String {
    let occupied: AHashSet<VoxelKey> = map
        .snapshot()
        .into_iter()
        .filter(|v| v.health > 0)
        .map(|v| v.key)
        .collect();
    let config = SurfaceConfig::mls_defaults();
    let (clearance, closing, step) = (
        config.clearance_cells(voxel_size),
        config.closing_cells(voxel_size),
        config.step_cells(voxel_size),
    );
    let window = Window::covering(occupied.iter().copied(), closing + 1).expect("an occupied map");
    let t = Instant::now();
    let surface = surface_cells(&occupied, clearance, closing as u32);
    let cpu_surface_ms = ms(t);
    let t = Instant::now();
    wall_distances(&occupied, &surface, clearance, step, voxel_size);
    let cpu_wall_ms = ms(t);

    let mut extractor = SurfaceExtractor::new(map, window, config).expect("window fits the device");
    let mut mean = Mean::default();
    for i in 0..6 {
        let t = Instant::now();
        extractor.extract(map);
        map.finish();
        let wall = ms(t);
        let passes = map
            .timer
            .as_mut()
            .map(|tm| tm.take(gpu))
            .unwrap_or_default();
        let t = Instant::now();
        extractor.read_surface_words();
        let readback = ms(t);
        if i == 0 {
            continue;
        }
        mean.add("wall", wall);
        mean.add("readback", readback);
        for (label, t) in passes {
            mean.add(label, t);
        }
    }
    format!(
        "| {label} | {voxel_size} | {}x{}x{} | {:.1} ms | {:.1} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        window.size.0,
        window.size.1,
        window.size.2,
        cpu_surface_ms,
        cpu_wall_ms,
        mean.get("plane fit"),
        mean.get("standable"),
        mean.get("hole fill"),
        mean.get("planes"),
        mean.get("wall distance"),
        mean.get("wall"),
        mean.get("readback"),
    )
}
