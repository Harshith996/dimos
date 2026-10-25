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
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop
from dimos.teleop.webxr.controller_types import Hand
from dimos.teleop.webxr.rgbd import SurfaceFrame


@pytest.fixture
def viewer(mocker):
    module = OpenYamSpatialTeleop(artifact_directory=Path("/unused"))
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
        "arms": {
            side: {"grasp": [0, y, 0.4], "orientation": [0, 0, 0, 1], "opening": 0.5}
            for side, y in [("left", 0.31), ("right", -0.31)]
        },
    }
    module._surface = SurfaceFrame(time.time(), b"", np.zeros((1, 3)))
    yield module
    module.stop()


def send(
    module,
    sequence,
    *,
    left=None,
    right=None,
    observed_at=None,
    focus=None,
    action="move",
    offset=(0, 0, 0),
):
    hands = {}
    for side, grab in [("left", left), ("right", right)]:
        if grab is not None:
            hands[side] = {
                "position": (np.asarray(module._feedback["arms"][side]["grasp"]) + offset).tolist(),
                "orientation": [0, 0, 0, 1],
                "pinched": grab,
                "action": action if grab else None,
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


def test_opening_drag_holds_arm_starts_at_feedback_and_stops_on_release(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True, action="opening", offset=(0, 0.14, 0))
    assert not viewer._controllers[Hand.LEFT].grip
    assert viewer._controllers[Hand.LEFT].trigger == 0.5
    assert send(viewer, 2, left=True, action="opening", offset=(-0.03, 0.14, 0))
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], None)
    assert viewer.left_gripper_command.publish.call_args.args[0].data == pytest.approx(0.25)
    viewer.right_gripper_command.publish.assert_not_called()
    viewer.left_gripper_command.publish.reset_mock()
    assert send(viewer, 3, left=False)
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], None)
    viewer.left_gripper_command.publish.assert_not_called()


def test_move_cannot_command_jaws_or_switch_to_opening_without_release(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], None)
    viewer.left_gripper_command.publish.assert_not_called()
    assert send(viewer, 2, left=True, action="opening", offset=(0, 0.14, 0))
    assert viewer._actions[Hand.LEFT] is None
    assert not viewer._controllers[Hand.LEFT].grip
    assert send(viewer, 3, left=True, action="opening", offset=(0, 0.14, 0))
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], None)
    viewer.left_gripper_command.publish.assert_not_called()


def test_lost_opening_tracking_requires_release_before_any_regrab(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True, action="opening", offset=(0, 0.14, 0))
    assert send(viewer, 2)
    assert send(viewer, 3, left=True, action="opening", offset=(0, 0.14, 0))
    assert viewer._actions[Hand.LEFT] is None
    assert send(viewer, 4, left=False)
    assert send(viewer, 5, left=True)
    assert viewer._controllers[Hand.LEFT].grip


def test_perception_focus_must_lie_on_an_observed_surface(viewer):
    assert send(viewer, 0, focus=[4, 4, 4])
    viewer.perception_focus.publish.assert_not_called()
    assert send(viewer, 1, focus=[0, 0, 0])
    point = viewer.perception_focus.publish.call_args.args[0]
    assert (point.x, point.y, point.z, point.frame_id) == (0, 0, 0, "world")


def test_rgbd_requires_timestamped_camera_mount_and_expires_when_depth_disappears(viewer):
    now = time.time()
    color = Image(np.zeros((2, 2, 3), dtype=np.uint8), ImageFormat.RGB, "camera", now)
    depth = Image(np.ones((2, 2), dtype=np.float32), ImageFormat.DEPTH, "camera", now)
    viewer._on_camera_info(CameraInfo.from_intrinsics(100, 100, 0, 0, 2, 2, frame_id="camera"))
    viewer._surface = None
    viewer._on_rgbd((color, depth))
    assert viewer._surface is None
    assert "Camera-to-world" in viewer._observation_error
    viewer._transforms.receive_transform(
        Transform(frame_id="world", child_frame_id="camera", ts=now)
    )
    viewer._on_rgbd((color, depth))
    assert viewer._surface.ts == now
    depth.data.fill(0)
    viewer._on_rgbd((color, depth))
    assert viewer._observation_error == "No valid workspace depth observations"
    viewer._surface = SurfaceFrame(now - 1, b"", np.zeros((1, 3)))
    assert not send(viewer, 0, left=True)
    assert viewer._controllers[Hand.LEFT] is None
