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

//! A local_map covers only the cylinder its scans observed. Inside it the new map
//! replaces everything; outside it the points last seen stand, out to the planning reach.

use std::collections::HashSet;

use lcm_msgs::geometry_msgs::PoseStamped;

/// Snap for matching a kept point to a new one; map points are voxel centers.
const KEY_M: f64 = 0.01;

/// The cylinder a local_map covers, as region_bounds carries it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl Crop {
    /// Position is the center, orientation holds radius, z_min, z_max.
    pub fn of_bounds(msg: &PoseStamped) -> Self {
        Crop {
            cx: msg.pose.position.x,
            cy: msg.pose.position.y,
            radius: msg.pose.orientation.x,
            z_min: msg.pose.orientation.y,
            z_max: msg.pose.orientation.z,
        }
    }

    fn covers(&self, p: &[f32; 3]) -> bool {
        let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
        (x - self.cx).hypot(y - self.cy) <= self.radius && (self.z_min..=self.z_max).contains(&z)
    }
}

fn key(p: &[f32; 3]) -> (i64, i64, i64) {
    let k = |v: f32| (v as f64 / KEY_M).round() as i64;
    (k(p[0]), k(p[1]), k(p[2]))
}

/// `fresh` plus the `kept` points outside `crop` within `reach` of the robot.
pub fn merge(
    kept: &[[f32; 3]],
    fresh: Vec<[f32; 3]>,
    crop: &Crop,
    robot: (f64, f64),
    reach: f64,
) -> Vec<[f32; 3]> {
    let seen: HashSet<_> = fresh.iter().map(key).collect();
    let mut out = fresh;
    out.extend(kept.iter().filter(|p| {
        !crop.covers(p)
            && (p[0] as f64 - robot.0).hypot(p[1] as f64 - robot.1) <= reach
            && !seen.contains(&key(p))
    }));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop(radius: f64) -> Crop {
        Crop {
            cx: 0.0,
            cy: 0.0,
            radius,
            z_min: -1.0,
            z_max: 2.0,
        }
    }

    #[test]
    fn a_wall_past_a_shrunk_crop_stays() {
        let wall = vec![[3.7, 0.0, 0.5], [3.7, 0.1, 0.5]];
        let near = vec![[1.0, 0.0, 0.5]];
        let first = merge(
            &[],
            [near.clone(), wall.clone()].concat(),
            &crop(4.0),
            (0.0, 0.0),
            6.0,
        );
        let second = merge(&first, near.clone(), &crop(3.4), (0.0, 0.0), 6.0);
        assert_eq!(second.len(), 3);
        assert!(wall.iter().all(|w| second.contains(w)));
    }

    #[test]
    fn inside_the_crop_the_new_map_wins() {
        let gone = [[2.0, 0.0, 0.5]];
        let merged = merge(&gone, vec![[1.0, 0.0, 0.5]], &crop(4.0), (0.0, 0.0), 6.0);
        assert_eq!(merged, vec![[1.0, 0.0, 0.5]]);
    }

    #[test]
    fn points_past_the_reach_are_dropped() {
        let far = [[5.0, 0.0, 0.5]];
        assert!(merge(&far, vec![], &crop(3.0), (0.0, 0.0), 4.5).is_empty());
    }

    #[test]
    fn a_point_above_the_crop_is_kept() {
        let high = [[1.0, 0.0, 2.5]];
        assert_eq!(
            merge(&high, vec![], &crop(3.0), (0.0, 0.0), 4.5),
            high.to_vec()
        );
    }

    #[test]
    fn a_point_the_new_map_repeats_is_not_doubled() {
        let p = [3.6, 0.0, 0.5];
        let mut merged = vec![p];
        for _ in 0..5 {
            merged = merge(&merged, vec![p], &crop(3.0), (0.0, 0.0), 6.0);
        }
        assert_eq!(merged, vec![p]);
    }
}
