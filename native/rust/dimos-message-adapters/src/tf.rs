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

//! Canonical generated TF messages at the SDK numeric transform boundary.

use dimos_generated_messages::builtin_interfaces::msg::time::Time;
use dimos_generated_messages::std_msgs::msg::header::Header;
use dimos_module::cdr;
use nalgebra::Isometry3;

pub fn decode(bytes: &[u8]) -> std::io::Result<Vec<dimos_module::Transform>> {
    from_message(cdr::decode(bytes)?)
}

pub fn from_message(
    message: dimos_generated_messages::tf2_msgs::msg::tf_message::TFMessage,
) -> std::io::Result<Vec<dimos_module::Transform>> {
    use nalgebra::{Quaternion, Translation3, UnitQuaternion};
    message
        .transforms
        .into_iter()
        .map(|edge| {
            let p = edge.transform.translation;
            let q = edge.transform.rotation;
            if ![p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "nonfinite TF components",
                ));
            }
            let rotation = UnitQuaternion::try_new(Quaternion::new(q.w, q.x, q.y, q.z), 1e-9)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "zero TF quaternion")
                })?;
            if edge.header.stamp.nanosec >= 1_000_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unnormalized TF stamp",
                ));
            }
            Ok(dimos_module::Transform::from_nanoseconds(
                edge.header.frame_id,
                edge.child_frame_id,
                edge.header.stamp.sec as i64 * 1_000_000_000 + edge.header.stamp.nanosec as i64,
                Isometry3::from_parts(Translation3::new(p.x, p.y, p.z), rotation),
            ))
        })
        .collect()
}

pub fn encode(transforms: &[dimos_module::Transform]) -> std::io::Result<Vec<u8>> {
    cdr::encode(&to_message(transforms)?)
}

pub fn to_message(
    transforms: &[dimos_module::Transform],
) -> std::io::Result<dimos_generated_messages::tf2_msgs::msg::tf_message::TFMessage> {
    use dimos_generated_messages::{
        geometry_msgs::msg::{
            quaternion::Quaternion, transform::Transform, transform_stamped::TransformStamped,
            vector3::Vector3,
        },
        tf2_msgs::msg::tf_message::TFMessage,
    };
    let transforms = transforms
        .iter()
        .map(|value| {
            let ns = value.stamp_nanoseconds().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "TF timestamp is outside range",
                )
            })?;
            let sec = i32::try_from(ns.div_euclid(1_000_000_000)).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "TF seconds exceed ROS int32",
                )
            })?;
            let p = value.translation();
            let q = value.rotation();
            if ![p.x, p.y, p.z, q.i, q.j, q.k, q.w]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "nonfinite TF components",
                ));
            }
            Ok(TransformStamped {
                header: Header {
                    stamp: Time {
                        sec,
                        nanosec: ns.rem_euclid(1_000_000_000) as u32,
                    },
                    frame_id: value.parent.clone(),
                },
                child_frame_id: value.child.clone(),
                transform: Transform {
                    translation: Vector3 {
                        x: p.x,
                        y: p.y,
                        z: p.z,
                    },
                    rotation: Quaternion {
                        x: q.i,
                        y: q.j,
                        z: q.k,
                        w: q.w,
                    },
                },
            })
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(TFMessage { transforms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dimos_module::Transform;
    use nalgebra::Isometry3;

    #[test]
    fn canonical_tf_preserves_exact_stamps_and_values_in_both_wire_orders() {
        for ns in [
            0,
            -1,
            1_700_000_000_000_000_001,
            1_700_000_000_000_000_002,
            1_700_000_000_123_456_789,
        ] {
            let edge = Transform::from_nanoseconds(
                "map",
                "camera",
                ns,
                Isometry3::translation(1.0, 2.0, 3.0),
            );
            let message = to_message(&[edge]).unwrap();
            assert_eq!(
                message.transforms[0].header.stamp.sec as i64 * 1_000_000_000
                    + message.transforms[0].header.stamp.nanosec as i64,
                ns
            );
            for little in [false, true] {
                let decoded = decode(&cdr::encode_endian(&message, little).unwrap()).unwrap();
                assert_eq!(decoded[0].stamp_nanoseconds(), Some(ns));
                assert_eq!(decoded[0].translation(), nalgebra::Vector3::new(1., 2., 3.));
                let restored = cdr::decode::<
                    dimos_generated_messages::tf2_msgs::msg::tf_message::TFMessage,
                >(&encode(&decoded).unwrap())
                .unwrap();
                assert_eq!(restored, message);
            }
        }
    }

    #[test]
    fn reject_invalid_components_and_time_before_returning_any_edges() {
        let good =
            to_message(&[Transform::new("map", "camera", 0., Isometry3::identity())]).unwrap();
        let mut bad_stamp = good.clone();
        bad_stamp.transforms[0].header.stamp.nanosec = 1_000_000_000;
        assert_eq!(
            from_message(bad_stamp).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let mut bad_rotation = good.clone();
        bad_rotation.transforms[0].transform.rotation.w = 0.;
        assert_eq!(
            from_message(bad_rotation).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let mut bad_position = good;
        bad_position.transforms[0].transform.translation.x = f64::NAN;
        assert_eq!(
            from_message(bad_position).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        let overflow =
            Transform::from_nanoseconds("map", "camera", i64::MAX, Isometry3::identity());
        assert_eq!(
            encode(&[overflow]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
