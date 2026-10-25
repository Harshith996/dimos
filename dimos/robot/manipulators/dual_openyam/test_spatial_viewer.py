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

"""Control must use fresh observations and explicit, independent hand grabs."""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop
from dimos.teleop.webxr.controller_types import Hand


@pytest.fixture
def viewer(mocker):
    module = OpenYamSpatialTeleop(artifact_directory=Path("/unused"), max_points=2)
    for port in (
        module.teleop_buttons,
        module.button_pressed,
        module.button_released,
        module.left_gripper_command,
        module.right_gripper_command,
        module.perception_focus,
    ):
        mocker.patch.object(port, "publish")
    module._feedback = {
        "ts": time.time(),
        "arms": {"left": {"grasp": [0, 0.31, 0.4]}, "right": {"grasp": [0, -0.31, 0.4]}},
    }
    module._cloud = {"ts": time.time(), "positions": [0, 0, 0], "colors": [1, 0, 0]}
    yield module
    module.stop()


def send(module, sequence, *, left=None, right=None, observed_at=None, focus=None):
    hands = {}
    for side, grab in [("left", left), ("right", right)]:
        if grab is not None:
            hands[side] = {
                "position": module._feedback["arms"][side]["grasp"],
                "orientation": [0, 0, 0, 1],
                "grab": grab,
                "opening": 0.5,
            }
    return module._dispatch_text_message(
        json.dumps(
            {
                "type": "spatial_input",
                "sequence": sequence,
                "observed_at": time.time() if observed_at is None else observed_at,
                "hands": hands,
                "focus": focus,
            }
        )
    )


def test_hands_grab_independently_and_tracking_loss_requires_release(viewer):
    assert send(viewer, 0, left=True)  # Arriving already pinched cannot engage.
    assert not viewer._controllers[Hand.LEFT].grip
    assert send(viewer, 1, left=False, right=False)
    assert send(viewer, 2, left=True, right=False)
    assert viewer._controllers[Hand.LEFT].grip
    assert not viewer._controllers[Hand.RIGHT].grip
    assert send(viewer, 3, right=False)  # Left tracking disappears.
    assert send(viewer, 4, left=True, right=False)
    assert not viewer._controllers[Hand.LEFT].grip
    assert send(viewer, 5, left=False)
    assert send(viewer, 6, left=True)
    assert viewer._controllers[Hand.LEFT].grip


def test_old_view_stops_control_and_replayed_sequence_cannot_reengage(viewer):
    assert send(viewer, 1, left=False)
    assert send(viewer, 2, left=True)
    assert not send(viewer, 2, left=True)
    assert viewer._controllers[Hand.LEFT] is None
    assert not send(viewer, 3, left=False, observed_at=time.time() - 2)
    assert send(viewer, 4, left=True)
    assert not viewer._controllers[Hand.LEFT].grip


def test_control_and_display_use_world_frame_without_quest_rotation(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    pose = viewer._get_output_pose(Hand.LEFT)
    assert pose.frame_id == "world"
    assert pose.position.to_list() == [0, 0.31, 0.4]
    assert pose.orientation.to_list() == [0, 0, 0, 1]


def test_cloud_needs_calibration_filters_invalid_depth_and_bounds_payload(viewer):
    cloud = PointCloud2.from_numpy(
        np.array([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1], [0, 0, 3], [0, 0, -1]], dtype=np.float32),
        frame_id="optical",
    )
    cloud.ts = time.time()
    viewer._on_cloud(cloud)
    assert viewer._cloud["positions"] == [0, 0, 0]  # No fabricated transform.
    viewer._transforms.receive_transform(
        Transform(
            translation=Vector3(1, 2, 3), frame_id="world", child_frame_id="optical", ts=cloud.ts
        )
    )
    viewer._on_cloud(cloud)
    np.testing.assert_allclose(
        np.asarray(viewer._cloud["positions"]).reshape(-1, 3), [[1, 2, 4], [1.2, 2, 4]]
    )
    assert viewer._cloud["frame"] == "world"


def test_perception_focus_must_lie_on_an_observed_surface(viewer):
    assert send(viewer, 0, focus=[4, 4, 4])
    viewer.perception_focus.publish.assert_not_called()
    assert send(viewer, 1, focus=[0, 0, 0])
    point = viewer.perception_focus.publish.call_args.args[0]
    assert (point.x, point.y, point.z, point.frame_id) == (0, 0, 0, "world")
