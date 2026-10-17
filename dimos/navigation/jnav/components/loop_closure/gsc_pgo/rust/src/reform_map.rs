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

//! The corrected global map a gsc deployment would keep next to the PGO, for `pgo-eval --map-voxel`:
//! every scan goes into a voxel grid in the raw odom frame (first point and time per voxel), and is
//! appended to the corrected map through the newest keyframe's correction. After a loop closure that
//! moves some keyframe more than `min_shift` since the map last saw it, the whole raw map is re-warped
//! with fast-reform (graph thinned to a voxel of error), replacing the corrected map.

use std::collections::HashSet;

use dimos_gsc_pgo::gsc_pgo::KeyPoseWithCloud;
use dimos_gsc_pgo::mat3::{self, Mat3, Vec3};
use fast_reform::{GraphDelta3D, Node, PointCloud2, Quat, Reformer, Transform};

/// The keyframe's correction (raw odom -> corrected): global * local⁻¹.
fn correction(key_pose: &KeyPoseWithCloud) -> (Mat3, Vec3) {
    let rotation = mat3::mat_mul(
        &key_pose.rotation_global,
        &mat3::transpose(&key_pose.rotation_local),
    );
    let moved = mat3::mat_vec(&rotation, &key_pose.translation_local);
    let translation = [0, 1, 2].map(|axis| key_pose.translation_global[axis] - moved[axis]);
    (rotation, translation)
}

fn apply(rotation: &Mat3, translation: &Vec3, point: [f32; 3]) -> [f32; 3] {
    let rotated = mat3::mat_vec(rotation, &point.map(|value| value as f64));
    [0, 1, 2].map(|axis| (rotated[axis] + translation[axis]) as f32)
}

pub struct ReformMap {
    voxel: f64,
    min_shift: f64,
    cells: HashSet<[i32; 3]>,
    raw: PointCloud2,
    corrected: Vec<[f32; 3]>,
    /// Each keyframe's global position as the corrected map has it.
    in_map: Vec<Vec3>,
    reformer: Reformer,
}

impl ReformMap {
    pub fn new(voxel: f64, min_shift: f64) -> Self {
        ReformMap {
            voxel,
            min_shift,
            cells: HashSet::new(),
            raw: PointCloud2 {
                timestamps: Some(Vec::new()),
                ..PointCloud2::default()
            },
            corrected: Vec::new(),
            in_map: Vec::new(),
            reformer: Reformer::new(GraphDelta3D::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.corrected.len()
    }

    /// Per scan: a body-frame cloud at its raw odom pose.
    pub fn add(
        &mut self,
        cloud: &[[f32; 3]],
        rotation: &Mat3,
        translation: &Vec3,
        ts: f64,
        key_poses: &[KeyPoseWithCloud],
    ) {
        for key_pose in &key_poses[self.in_map.len()..] {
            self.in_map.push(key_pose.translation_global);
        }
        let newest = key_poses.last().map(correction);
        for &point in cloud {
            let world = apply(rotation, translation, point);
            let cell = world.map(|value| (value as f64 / self.voxel).floor() as i32);
            if !self.cells.insert(cell) {
                continue;
            }
            self.raw.points.push(world);
            self.raw.timestamps.as_mut().unwrap().push(ts);
            self.corrected.push(match &newest {
                Some((correction_rotation, correction_translation)) => {
                    apply(correction_rotation, correction_translation, world)
                }
                None => world,
            });
        }
    }

    /// After a closure: the largest keyframe move since the map saw it, and re-warp the map when it
    /// passes `min_shift`. Returns (shift, Some(kept graph nodes) if it reformed).
    pub fn after_closure(&mut self, key_poses: &[KeyPoseWithCloud]) -> (f64, Option<usize>) {
        let shift = key_poses
            .iter()
            .zip(&self.in_map)
            .map(|(key_pose, seen)| {
                (0..3)
                    .map(|axis| (key_pose.translation_global[axis] - seen[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max);
        if shift <= self.min_shift {
            return (shift, None);
        }
        let mut delta = GraphDelta3D::default();
        for (id, key_pose) in key_poses.iter().enumerate() {
            let (rotation, translation) = correction(key_pose);
            let [w, x, y, z] = mat3::quat_from_mat(&rotation);
            let [px, py, pz] = key_pose.translation_local;
            delta.nodes.push(Node {
                id: id as u64,
                ts: key_pose.time,
                position: fast_reform::Vec3::new(px, py, pz),
            });
            delta.transforms.push(Transform {
                translation: fast_reform::Vec3::new(translation[0], translation[1], translation[2]),
                rotation: Quat::new(x, y, z, w),
            });
        }
        self.reformer.update_within_error(delta, self.voxel);
        self.corrected = self.reformer.apply(&self.raw).points;
        self.in_map = key_poses
            .iter()
            .map(|key_pose| key_pose.translation_global)
            .collect();
        (shift, Some(self.reformer.graph_delta().len()))
    }
}
