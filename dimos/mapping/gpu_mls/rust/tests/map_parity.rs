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

//! The GPU brick map against the CPU VoxelMap on synthetic lidar frames.

mod common;

use common::{cpu_config, diff_maps, gpu, gpu_config};
use dimos_gpu_mls::brick_map::BrickMap;
use dimos_gpu_mls::synthetic::{Lcg, Scene};
use dimos_voxel_ray_tracing::voxel_ray_tracer::{seed_points, update_map, Config, VoxelMap};

const FRAMES: usize = 8;
const POINTS_PER_FRAME: usize = 20_000;
const MOMENT_TOLERANCE: f64 = 1e-4;
/// Metal fuses `p - k * v` into one fma, so a point on a fine-cell boundary
/// can land in the neighboring fine cell. Allowed per voxel, as a fraction.
const FINE_TOLERANCE: f64 = 1e-4;

fn frames(scene: &Scene, cfg: &Config) -> Vec<((f32, f32, f32), Vec<(f32, f32, f32)>)> {
    let mut rng = Lcg(42);
    scene
        .trajectory(FRAMES, 0.4)
        .into_iter()
        .map(|o| {
            let pts = scene.scan(o, POINTS_PER_FRAME, &mut rng, 0.01, cfg.max_range + 5.0);
            (o, pts)
        })
        .collect()
}

#[test]
fn ingest_without_clearing_matches_cpu() {
    for voxel_size in [0.05, 0.08] {
        // A grace region longer than any ray and no shadow: the CPU walk clears nothing.
        let cfg = Config {
            grace_depth: 1e6,
            shadow_depth: 0.0,
            ..cpu_config(voxel_size)
        };
        let scene = Scene::room(12.0, 2.5);
        let mut cpu = VoxelMap::default();
        let mut map = BrickMap::new(gpu(), gpu_config(&cfg, 16384, POINTS_PER_FRAME as u32))
            .expect("brick map fits the device");
        for (origin, pts) in frames(&scene, &cfg) {
            update_map(&mut cpu, origin, &pts, &cfg);
            map.ingest(origin, &pts);
        }
        map.finish();
        let d = diff_maps(&cpu, &map.snapshot(), voxel_size);
        eprintln!("ingest at {voxel_size}: {d:?}", d = summary(&d));
        assert_eq!(map.stats().overflow, 0);
        assert!(d.cpu_voxels > 10_000);
        assert_eq!(d.mismatched() - d.fine, 0, "{}", summary(&d));
        assert!(
            d.fine as f64 <= FINE_TOLERANCE * d.cpu_voxels as f64,
            "{}",
            summary(&d)
        );
        assert!(d.moment_rel_err < MOMENT_TOLERANCE, "{}", summary(&d));
    }
}

#[test]
fn ray_traced_clearing_matches_cpu() {
    for voxel_size in [0.05, 0.08] {
        let cfg = cpu_config(voxel_size);
        let scene = Scene::room(12.0, 2.5);
        let mut cpu = VoxelMap::default();
        let mut map = BrickMap::new(gpu(), gpu_config(&cfg, 16384, POINTS_PER_FRAME as u32))
            .expect("brick map fits the device");
        let mut misses = 0;
        for (origin, pts) in frames(&scene, &cfg) {
            update_map(&mut cpu, origin, &pts, &cfg);
            map.integrate(origin, &pts);
            let stats = map.stats();
            assert_eq!(stats.overflow, 0);
            misses += stats.misses;
        }
        map.finish();
        let snapshot = map.snapshot();
        let d = diff_maps(&cpu, &snapshot, voxel_size);
        let fitted = snapshot.iter().filter(|v| !v.stale).count();
        eprintln!(
            "trace at {voxel_size}: {}; gpu misses {misses}, fitted normals {fitted}",
            summary(&d)
        );
        assert!(misses > 0, "the frames must exercise clearing");
        // Grazing-ray sparing hinges on f32 Jacobi fits and fused DDA math,
        // so boundary cases may flip; the rest must agree voxel for voxel.
        assert!(
            d.mismatched() as f64 <= 1e-3 * d.cpu_voxels as f64,
            "{}",
            summary(&d)
        );
        assert!(d.moment_rel_err < MOMENT_TOLERANCE, "{}", summary(&d));
    }
}

#[test]
fn seed_matches_cpu() {
    let cfg = cpu_config(0.05);
    let scene = Scene::room(12.0, 2.5);
    let cloud: Vec<(f32, f32, f32)> = frames(&scene, &cfg)
        .into_iter()
        .flat_map(|(_, p)| p)
        .collect();
    let mut cpu = VoxelMap::default();
    seed_points(&mut cpu, &cloud, &cfg);
    let mut map = BrickMap::new(gpu(), gpu_config(&cfg, 16384, cloud.len() as u32))
        .expect("brick map fits the device");
    map.seed(&cloud);
    // A second seed over existing voxels must change nothing.
    seed_points(&mut cpu, &cloud[..1000], &cfg);
    map.seed(&cloud[..1000]);
    map.finish();
    let d = diff_maps(&cpu, &map.snapshot(), cfg.voxel_size);
    eprintln!("seed: {}", summary(&d));
    assert_eq!(d.mismatched(), 0, "{}", summary(&d));
    assert!(d.moment_rel_err < MOMENT_TOLERANCE, "{}", summary(&d));
}

pub fn summary(d: &common::MapDiff) -> String {
    format!(
        "cpu {} gpu {} only_cpu {} only_gpu {} health {} count {} fine {} moment_rel_err {:.2e}",
        d.cpu_voxels,
        d.gpu_voxels,
        d.only_cpu.len(),
        d.only_gpu.len(),
        d.health,
        d.count,
        d.fine,
        d.moment_rel_err
    )
}
