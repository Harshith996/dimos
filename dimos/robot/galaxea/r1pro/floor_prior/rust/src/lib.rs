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

//! The R1 Pro's head depth: `depth2depth_cloud` with a floor prior. The chassis lidar never sees the floor near the
//! robot, so unanchored the near field reads 2-5x short; each frame, pixels the calibration puts on or just over
//! `height_frame`'s floor (its z = 0) are anchored on it, in rounds, before the frame is calibrated.

use std::sync::Arc;

use dimos_depth2depth_cloud::depth2depth::{Anchor, Calibration};
use dimos_depth2depth_cloud::module::{Config as DepthConfig, Depth2DepthCloud, FrameAnchors};
use dimos_depth2depth_cloud::undistort::UndistortMap;
use dimos_module::{Builder, Module, NativeConfig};
use nalgebra::{Isometry3, Vector3};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use validator::Validate;

/// Depth2DepthCloud's config plus the floor prior's, one flat object as the Python wrapper sends it.
#[derive(Debug, Serialize, Validate)]
pub struct Config {
    /// Pixels probed one per block of this many.
    #[validate(range(min = 1, max = 256))]
    pub floor_anchor_step: i64,
    /// A probed pixel the calibration puts at most this above the floor, and no deeper under it than
    /// `floor_min_height_m`, is anchored on it.
    #[validate(range(min = 0.0, max = 10.0))]
    pub floor_tolerance_m: f64,
    /// Deeper than this under the floor is a drop, left as measured rather than anchored on the floor.
    #[validate(range(min = -10.0, max = 0.0))]
    pub floor_min_height_m: f64,
    /// Probe-and-anchor rounds per frame: each round's anchors recalibrate the pixels the next one probes.
    #[validate(range(min = 1, max = 16))]
    pub floor_anchor_rounds: i64,
    #[serde(flatten)]
    #[validate(nested)]
    pub depth: DepthConfig,
    /// Depth2DepthCloud's `max_range_m`: the floor is counted out to it.
    #[serde(skip)]
    max_range_m: f64,
}

impl NativeConfig for Config {}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = serde_json::Map::deserialize(deserializer)?;
        let mut take = |name: &'static str| {
            fields
                .remove(name)
                .ok_or_else(|| D::Error::missing_field(name))
        };
        let floor_anchor_step = take("floor_anchor_step")?;
        let floor_tolerance_m = take("floor_tolerance_m")?;
        let floor_min_height_m = take("floor_min_height_m")?;
        let floor_anchor_rounds = take("floor_anchor_rounds")?;
        let max_range_m = fields
            .get("max_range_m")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| D::Error::missing_field("max_range_m"))?;
        Ok(Config {
            floor_anchor_step: serde_json::from_value(floor_anchor_step)
                .map_err(D::Error::custom)?,
            floor_tolerance_m: serde_json::from_value(floor_tolerance_m)
                .map_err(D::Error::custom)?,
            floor_min_height_m: serde_json::from_value(floor_min_height_m)
                .map_err(D::Error::custom)?,
            floor_anchor_rounds: serde_json::from_value(floor_anchor_rounds)
                .map_err(D::Error::custom)?,
            depth: serde_json::from_value(serde_json::Value::Object(fields))
                .map_err(D::Error::custom)?,
            max_range_m,
        })
    }
}

/// Depth2DepthCloud, with FloorPrior adding to every frame's anchors.
pub struct R1ProHeadDepth {
    depth: Depth2DepthCloud,
}

impl Module for R1ProHeadDepth {
    type Config = Config;

    fn build(builder: &mut Builder, config: Config) -> Self {
        let floor = FloorPrior {
            step: config.floor_anchor_step as usize,
            tolerance: config.floor_tolerance_m,
            min_height: config.floor_min_height_m,
            rounds: config.floor_anchor_rounds as usize,
            max_range: config.max_range_m,
        };
        Self {
            depth: Depth2DepthCloud::build(builder, config.depth)
                .with_frame_anchors(Arc::new(floor)),
        }
    }

    async fn setup(&mut self) {
        self.depth.setup().await
    }

    async fn handle(&mut self) {
        self.depth.handle().await
    }

    async fn teardown(&mut self) {
        self.depth.teardown().await
    }
}

impl FrameAnchors for FloorPrior {
    fn anchors(
        &self,
        calibration: &Calibration,
        pred: &[f32],
        map: &UndistortMap,
        visible: &[Anchor],
        height_from_camera: Option<&Isometry3<f64>>,
    ) -> Vec<Anchor> {
        height_from_camera.map_or_else(Vec::new, |floor_from_camera| {
            self.floor_anchors(calibration, pred, map, visible, floor_from_camera)
        })
    }
}

/// Floor prior: the Config fields of the same names (`height_frame`'s z = 0 is the floor), and the floor counted out to `max_range` along the optical axis.
struct FloorPrior {
    step: usize,
    tolerance: f64,
    min_height: f64,
    rounds: usize,
    max_range: f64,
}

impl FloorPrior {
    /// Floor anchors (`floor_from_camera`'s z = 0) for the probed pixels the calibration puts on, just over or just under the floor.
    fn floor_anchors(
        &self,
        calibration: &Calibration,
        pred: &[f32],
        map: &UndistortMap,
        visible: &[Anchor],
        floor_from_camera: &Isometry3<f64>,
    ) -> Vec<Anchor> {
        let pin = &map.camera;
        let camera_z = floor_from_camera.translation.vector.z;
        let candidates: Vec<(usize, usize, f64, f64)> = (0..map.height)
            .step_by(self.step)
            .flat_map(|row| {
                (0..map.width)
                    .step_by(self.step)
                    .map(move |column| (column, row))
            })
            .filter_map(|(column, row)| {
                let ray = Vector3::new(
                    (column as f64 - pin.cx as f64) / pin.fx as f64,
                    (row as f64 - pin.cy as f64) / pin.fy as f64,
                    1.0,
                );
                // Height gained per unit of depth along the optical axis; the floor is only below the camera.
                let down = (floor_from_camera.rotation * ray).z;
                let depth = -camera_z / down;
                (down < 0.0 && depth <= self.max_range).then_some((column, row, depth, down))
            })
            .collect();
        let pixels: Vec<(usize, usize)> = candidates
            .iter()
            .map(|&(column, row, _, _)| (column, row))
            .collect();
        let mut anchored = vec![false; candidates.len()];
        let mut floor: Vec<Anchor> = Vec::new();
        // Floor read too near drops onto the floor once its neighbours are anchored; an obstacle keeps its height.
        for _ in 0..self.rounds {
            let with_floor: Vec<Anchor> = visible.iter().chain(&floor).copied().collect();
            let probed = calibration.probe(pred, map.height, map.width, &with_floor, &pixels);
            let before = floor.len();
            for (index, (&(column, row, depth, down), probed)) in
                candidates.iter().zip(probed).enumerate()
            {
                let height = camera_z + probed as f64 * down;
                if !anchored[index] && (self.min_height..=self.tolerance).contains(&height) {
                    anchored[index] = true;
                    floor.push(Anchor {
                        u: column as f32,
                        v: row as f32,
                        z: depth as f32,
                    });
                }
            }
            if floor.len() == before {
                break;
            }
        }
        floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dimos_depth2depth_cloud::depth2depth::CalibrationConfig;
    use dimos_depth2depth_cloud::undistort::{Lens, UndistortMap};
    use lcm_msgs::sensor_msgs::CameraInfo;
    use nalgebra::Point3;

    /// A camera 1 m up, pitched 30 degrees down: its 160x120 pinhole map and its pose in the floor frame.
    fn camera() -> (UndistortMap, Isometry3<f64>) {
        // Optical axes (x right, y down, z ahead) to base (x ahead, z up).
        let optical = nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
            0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
        ));
        let pitch = nalgebra::Rotation3::from_axis_angle(&Vector3::y_axis(), 30f64.to_radians());
        let floor_from_camera = Isometry3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::from_rotation_matrix(&(pitch * optical)),
        );
        let lens = Lens::from_info(
            &CameraInfo {
                width: 160,
                height: 120,
                K: [100.0, 0.0, 79.5, 0.0, 100.0, 59.5, 0.0, 0.0, 1.0],
                D: vec![0.0; 5],
                ..Default::default()
            },
            1.0,
        )
        .unwrap();
        (UndistortMap::new(&lens, 160, 120, 100.0), floor_from_camera)
    }

    /// Depth along the optical axis to the floor for a pixel, and whether that floor point is in the 1.5-1.8 m patch ahead.
    fn floor_hit(
        map: &UndistortMap,
        floor_from_camera: &Isometry3<f64>,
        column: usize,
        row: usize,
    ) -> Option<(f64, bool)> {
        let pin = &map.camera;
        let ray = Vector3::new(
            (column as f64 - pin.cx as f64) / pin.fx as f64,
            (row as f64 - pin.cy as f64) / pin.fy as f64,
            1.0,
        );
        let down = (floor_from_camera.rotation * ray).z;
        let depth = -floor_from_camera.translation.vector.z / down;
        let hit = floor_from_camera * Point3::from(ray * depth);
        (down < 0.0).then_some((depth, (1.5..1.8).contains(&hit.x) && hit.y.abs() < 0.2))
    }

    /// The model's depth for every pixel: `on_floor` and `in_patch` scale the true floor depth, 20 m where there is no floor.
    fn predicted(
        map: &UndistortMap,
        floor_from_camera: &Isometry3<f64>,
        depth_of: impl Fn(f64, bool) -> f64,
    ) -> Vec<f32> {
        (0..map.height)
            .flat_map(|row| (0..map.width).map(move |column| (column, row)))
            .map(
                |(column, row)| match floor_hit(map, floor_from_camera, column, row) {
                    Some((depth, in_patch)) => depth_of(depth, in_patch) as f32,
                    None => 20.0,
                },
            )
            .collect()
    }

    fn prior(min_height: f64) -> FloorPrior {
        FloorPrior {
            step: 4,
            tolerance: 0.05,
            min_height,
            rounds: 3,
            max_range: 6.0,
        }
    }

    #[test]
    fn floor_read_too_far_is_anchored_on_the_floor_and_a_box_on_it_is_not() {
        let (map, floor_from_camera) = camera();
        // The model reads the floor 30% too far and the box as a block 20% nearer than the floor behind it.
        let pred = predicted(&map, &floor_from_camera, |depth, on_box| {
            depth * if on_box { 0.8 } else { 1.3 }
        });
        let calibration = Calibration::new(CalibrationConfig::default());

        let floor = prior(-1.0).floor_anchors(&calibration, &pred, &map, &[], &floor_from_camera);

        assert!(floor.len() > 100, "{}", floor.len());
        for anchor in &floor {
            let (depth, on_box) = floor_hit(
                &map,
                &floor_from_camera,
                anchor.u as usize,
                anchor.v as usize,
            )
            .unwrap();
            assert!(!on_box, "anchored the box at {anchor:?}");
            assert!(
                (anchor.z as f64 - depth).abs() < 1e-4,
                "{anchor:?} vs {depth}"
            );
        }
    }

    #[test]
    fn a_drop_deeper_than_min_height_is_not_anchored_on_the_floor() {
        let (map, floor_from_camera) = camera();
        // The floor read right, and a pit 30 cm deep in the patch: its rays run on to z = -0.3.
        let pred = predicted(&map, &floor_from_camera, |depth, in_pit| {
            depth * if in_pit { 1.3 } else { 1.0 }
        });
        let calibration = Calibration::new(CalibrationConfig::default());

        let floor = prior(-0.15).floor_anchors(&calibration, &pred, &map, &[], &floor_from_camera);

        assert!(floor.len() > 100, "{}", floor.len());
        for anchor in &floor {
            let (_, in_pit) = floor_hit(
                &map,
                &floor_from_camera,
                anchor.u as usize,
                anchor.v as usize,
            )
            .unwrap();
            assert!(!in_pit, "anchored the pit at {anchor:?}");
        }
    }
}
