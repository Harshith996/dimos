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

import json

import numpy as np

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.experimental.object import Object


def test_agent_geometry_matches_the_rendered_observation_without_sensor_payloads():
    observed = Object(
        name="cup",
        object_id="observed-cup",
        center=Vector3(0.3, 0.1, 0.2),
        size=Vector3(0.08, 0.12, 0.2),
        pose=PoseStamped(frame_id="world", orientation=Quaternion(1, 0, 0, 0)),
        pointcloud=PointCloud2.from_numpy(np.zeros((1, 3)), frame_id="world"),
        image=Image(),
        bbox=(0.0, 0.0, 1.0, 1.0),
        track_id=1,
        class_id=0,
        confidence=0.9,
        ts=12.0,
        last_seen_ts=12.0,
    )

    encoded = json.loads(json.dumps(observed.agent_encode()))
    rendered = observed.to_detection3d_msg()

    assert encoded["object_id"] == rendered.id
    assert encoded["frame_id"] == "world"
    assert encoded["position"] == [0.3, 0.1, 0.2]
    assert encoded["orientation"] == [1.0, 0.0, 0.0, 0.0]
    assert encoded["size"] == [0.08, 0.12, 0.2]
    assert encoded["last_seen_ts"] == 12.0
    assert "pointcloud" not in encoded and "image" not in encoded
