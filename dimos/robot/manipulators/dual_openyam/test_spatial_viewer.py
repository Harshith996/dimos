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

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from dimos.imitation.collection.spec import EpisodeCollectionSpec
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
        if isinstance(grab, dict):
            hands[side] = grab
        elif grab is not None:
            hands[side] = {
                "position": (np.asarray(module._feedback["arms"][side]["grasp"]) + offset).tolist(),
                "orientation": [0, 0, 0, 1],
                "pinched": grab,
                "action": action if grab else None,
                "arm": side if grab else None,
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
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 1, left=False, right=False)
    assert send(viewer, 2, left=True, right=False)
    assert viewer._controllers[Hand.LEFT].grip
    assert viewer._controllers[Hand.RIGHT] is None
    assert send(viewer, 3, right=False)  # Left tracking disappears.
    assert send(viewer, 4, left=True, right=False)
    assert viewer._controllers[Hand.LEFT] is None
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
    assert viewer._controllers[Hand.LEFT] is None


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
    assert Hand.LEFT not in viewer._claims
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 3, left=True, action="opening", offset=(0, 0.14, 0))
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], None)
    viewer.left_gripper_command.publish.assert_not_called()


def test_lost_opening_tracking_requires_release_before_any_regrab(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True, action="opening", offset=(0, 0.14, 0))
    assert send(viewer, 2)
    assert send(viewer, 3, left=True, action="opening", offset=(0, 0.14, 0))
    assert Hand.LEFT not in viewer._claims
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


def test_spatial_episode_controls_call_the_existing_owner_and_report_rejected_saves(viewer, mocker):
    owner = mocker.create_autospec(EpisodeCollectionSpec, instance=True)
    owner.get_status.return_value = {"recording": None, "episodes_saved": 2}
    owner.record.return_value = {"recording": "take-3", "episodes_saved": 2}
    owner.finish.side_effect = RuntimeError("Camera stopped; discard this take")
    viewer.set_module_ref("_collection", owner)
    viewer._web_server = mocker.Mock(app=FastAPI())
    viewer._setup_routes()
    with TestClient(viewer._web_server.app) as client:
        assert client.get("/spatial/collection").json()["episodes_saved"] == 2
        result = client.post("/spatial/collection", json={"action": "record", "task": "pick cube"})
        assert result.json()["recording"] == "take-3"
        owner.record.assert_called_once_with("pick cube")
        result = client.post("/spatial/collection", json={"action": "save"})
        assert result.status_code == 409
        assert result.json()["detail"] == "Camera stopped; discard this take"
        owner.finish.assert_called_once_with(True)
        assert client.post("/spatial/collection", json={"action": "move"}).status_code == 422


def tool_sample(viewer, arm, action, *, pinched=True, offset=(0, 0, 0)):
    return {
        "arm": arm if pinched else None,
        "action": action if pinched else None,
        "pinched": pinched,
        "position": (np.asarray(viewer._feedback["arms"][arm]["grasp"]) + offset).tolist(),
        "orientation": [0, 0, 0, 1],
    }


@pytest.mark.parametrize("human,robot", [("left", Hand.RIGHT), ("right", Hand.LEFT)])
def test_either_human_hand_controls_the_opposite_arm(viewer, human, robot):
    target = robot.name.lower()
    assert send(viewer, 0, **{human: False})
    assert send(viewer, 1, **{human: tool_sample(viewer, target, "move")})
    assert viewer._controllers[robot].grip
    assert viewer._controllers[Hand(1 - robot)] is None
    assert (
        viewer._get_output_pose(robot).position.to_list()
        == viewer._feedback["arms"][target]["grasp"]
    )


def test_two_hands_can_move_and_adjust_the_same_arm_independently(viewer):
    move = tool_sample(viewer, "right", "move")
    opening = tool_sample(viewer, "right", "opening", offset=(0, 0.14, 0))
    assert send(viewer, 0, left=False, right=False)
    assert send(viewer, 1, left=move, right=opening)
    assert viewer._controllers[Hand.RIGHT].grip
    assert viewer._controllers[Hand.RIGHT].trigger == 0.5
    move = tool_sample(viewer, "right", "move", offset=(0.02, 0, 0))
    opening = tool_sample(viewer, "right", "opening", offset=(-0.03, 0.14, 0))
    assert send(viewer, 2, left=move, right=opening)
    viewer._publish_button_state(None, viewer._controllers[Hand.RIGHT])
    assert viewer.right_gripper_command.publish.call_args.args[0].data == pytest.approx(0.25)
    assert viewer._get_output_pose(Hand.RIGHT).position.x == pytest.approx(0.02)
    viewer.left_gripper_command.publish.assert_not_called()
    assert send(viewer, 3, left=move, right=False)
    assert viewer._controllers[Hand.RIGHT].grip
    viewer.right_gripper_command.publish.reset_mock()
    viewer._publish_button_state(None, viewer._controllers[Hand.RIGHT])
    viewer.right_gripper_command.publish.assert_not_called()


def test_one_control_cannot_be_stolen_and_transfer_publishes_release_before_next_tick(viewer):
    move = tool_sample(viewer, "left", "move")
    assert send(viewer, 0, left=False, right=False)
    assert send(viewer, 1, left=move, right=move)
    assert viewer._owner(Hand.LEFT, "move") == Hand.LEFT
    assert send(viewer, 2, left=False, right=move)
    assert not viewer.teleop_buttons.publish.call_args.args[0].left_grip
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 3, right=move)
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 4, right=False)
    assert send(viewer, 5, right=move)
    assert viewer._owner(Hand.LEFT, "move") == Hand.RIGHT
    assert viewer._controllers[Hand.LEFT].grip


def test_losing_movement_hand_keeps_only_the_independently_owned_slider(viewer):
    move = tool_sample(viewer, "left", "move")
    opening = tool_sample(viewer, "left", "opening", offset=(0, 0.14, 0))
    assert send(viewer, 0, left=False, right=False)
    assert send(viewer, 1, left=move, right=opening)
    assert send(viewer, 2, right=opening)
    assert not viewer._controllers[Hand.LEFT].grip
    assert viewer._gripper_is_enabled(Hand.LEFT, None)
    assert send(viewer, 3, left=move, right=opening)
    assert not viewer._controllers[Hand.LEFT].grip
    viewer._expire_stale_state(time.monotonic() + 1)
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 4, left=move, right=opening)
    assert viewer._controllers[Hand.LEFT] is None


def test_claim_cannot_change_robot_arm_while_pinched_or_start_far_from_target(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=tool_sample(viewer, "right", "move"))
    assert send(viewer, 2, left=tool_sample(viewer, "left", "move"))
    assert viewer._controllers[Hand.LEFT] is None
    assert viewer._controllers[Hand.RIGHT] is None
    assert send(viewer, 3, left=False)
    assert send(viewer, 4, left=tool_sample(viewer, "right", "opening", offset=(1, 1, 1)))
    assert viewer._controllers[Hand.RIGHT] is None


def test_tool_packets_require_explicit_robot_identity(viewer):
    sample = tool_sample(viewer, "left", "move")
    del sample["arm"]
    assert not send(viewer, 0, left=sample)
    assert viewer._controllers[Hand.LEFT] is None
