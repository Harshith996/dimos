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

import pytest

from dimos.robot.galaxea.r1pro.lio import R1ProLioConfig, mount_transforms


def test_pointlio_frame_is_the_mid360_imu_under_the_lidar() -> None:
    config = R1ProLioConfig()
    imu_to_base, base_to_lidar = mount_transforms(config)
    # Point-LIO reports the IMU's pose and cloud, which sits extrinsic_t's opposite away from the lidar.
    imu_in_lidar = (-imu_to_base).translation - base_to_lidar.translation
    expected = [
        -config.extrinsic_t[0],
        -config.extrinsic_t[1],
        config.mount_z_trim_m - config.extrinsic_t[2],
    ]
    assert [imu_in_lidar.x, imu_in_lidar.y, imu_in_lidar.z] == pytest.approx(expected, abs=1e-9)
