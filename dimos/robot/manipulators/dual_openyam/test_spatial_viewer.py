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

"""Delayed observations preserve grabs; control loss invalidates queued input."""

from dataclasses import replace
import json
from pathlib import Path
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from dimos.imitation.collection.spec import EpisodeCollectionSpec
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.protocol.testing.link import Delay, DelayTimeline, DeliveryQueue, LinkConditions
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop
from dimos.teleop.webxr.controller_types import Hand
from dimos.teleop.webxr.rgbd import SurfaceFrame
from dimos.web.robot_web_interface import RobotWebInterface


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


def test_link_reset_discards_queued_grab_and_requires_release_on_reconnect(viewer, mocker):
    mocker.patch(
        "dimos.robot.manipulators.dual_openyam.spatial_viewer.time.time", return_value=123.0
    )
    viewer._feedback["ts"] = 123.0
    viewer._surface = replace(viewer._surface, ts=123.0)
    viewer._web_server = RobotWebInterface(
        link_conditions=LinkConditions(to_robot=Delay(base_s=0.1))
    )
    viewer._setup_routes()
    runtime = viewer._web_server.link_runtime
    queued, processed = threading.Event(), threading.Event()
    original_enqueue = runtime.enqueue
    original_dispatch = viewer._dispatch_text_message

    def enqueue(*args, **kwargs):
        original_enqueue(*args, **kwargs)
        if args[1]["type"] == "websocket.receive":
            queued.set()

    def dispatch(payload):
        result = original_dispatch(payload)
        processed.set()
        return result

    mocker.patch.object(runtime, "enqueue", side_effect=enqueue)
    handler = mocker.patch.object(viewer, "_dispatch_text_message", side_effect=dispatch)

    def message(sequence, pinched, epoch):
        return {
            "type": "spatial_input",
            "sequence": sequence,
            "control_epoch": epoch,
            "observed_at": 123.0,
            "hands": {
                "left": {
                    "position": viewer._feedback["arms"]["left"]["grasp"],
                    "orientation": [0, 0, 0, 1],
                    "pinched": pinched,
                    "action": "move" if pinched else None,
                    "arm": "left" if pinched else None,
                }
            },
        }

    with TestClient(viewer._web_server.app) as client:
        with client.websocket_connect("/ws") as ws:
            state = client.get("/spatial/state").json()
            epoch = state["control"]["epoch"]
            assert state["control"]["robot_ready"]
            ws.send_json(message(0, False, epoch))
            assert processed.wait(2)
            processed.clear()
            queued.clear()
            ws.send_json(message(1, True, epoch))
            assert queued.wait(2)
            status = viewer.reset_link_conditions()
            assert ws.receive()["code"] == 1012
            assert handler.call_count == 1
            assert status["pending_messages"] == 0
            assert viewer._controllers[Hand.LEFT] is None
        with client.websocket_connect("/ws") as ws:
            state = client.get("/spatial/state").json()
            assert state["control"]["epoch"] > epoch
            ws.send_json(message(0, True, state["control"]["epoch"]))
            assert processed.wait(2)
            assert viewer._claims == {}
            assert viewer._controllers[Hand.LEFT] is None


def send(
    module,
    sequence,
    *,
    left=None,
    right=None,
    observed_at=None,
    control_epoch=None,
    focus=None,
    action="move",
    offset=(0, 0, 0),
    mode="tools",
    trigger=0.0,
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
                "trigger": trigger,
            }
    return module._dispatch_text_message(
        json.dumps(
            {
                "type": "spatial_input",
                "sequence": sequence,
                "control_epoch": module._control_epoch if control_epoch is None else control_epoch,
                "observed_at": time.time() if observed_at is None else observed_at,
                "hands": hands,
                "focus": focus,
                "mode": mode,
            }
        )
    )


@pytest.mark.parametrize("side,hand", [("left", Hand.LEFT), ("right", Hand.RIGHT)])
def test_clutch_moves_one_arm_from_free_space_and_trigger_controls_only_its_jaws(
    viewer, side, hand
):
    assert not send(viewer, 0, mode="clutch")  # Mode switch invalidates the previous epoch.
    assert send(viewer, 1, mode="clutch", **{side: False})
    assert send(viewer, 2, mode="clutch", offset=(2, 0, 0), trigger=0.7, **{side: True})
    assert viewer._controllers[hand].trigger == 0.7
    assert viewer._current_poses[hand].position.x == 2
    other = Hand.RIGHT if hand == Hand.LEFT else Hand.LEFT
    assert viewer._controllers[other] is None
    ports = {Hand.LEFT: viewer.left_gripper_command, Hand.RIGHT: viewer.right_gripper_command}
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], viewer._controllers[Hand.RIGHT])
    assert ports[hand].publish.call_args.args[0].data == pytest.approx(0.3)
    ports[other].publish.assert_not_called()
    ports[hand].publish.reset_mock()
    assert send(viewer, 3, mode="clutch", trigger=1, **{side: False})
    assert viewer._controllers[hand] is None
    viewer._publish_button_state(viewer._controllers[Hand.LEFT], viewer._controllers[Hand.RIGHT])
    ports[hand].publish.assert_not_called()


def test_clutch_mode_change_and_tracking_loss_require_release_before_reengaging(viewer):
    send(viewer, 0, left=False)
    send(viewer, 1, left=True)
    epoch = viewer._control_epoch
    assert not send(viewer, 2, mode="clutch", left=True)
    assert viewer._claims == {}
    assert viewer._control_epoch > epoch
    assert not send(viewer, 3, control_epoch=epoch, left=True)
    assert send(viewer, 4, mode="clutch", left=True)
    assert viewer._claims == {}
    assert send(viewer, 5, mode="clutch", left=False, right=False)
    assert send(viewer, 6, mode="clutch", left=True, right=True)
    assert all(controller.grip == 1 for controller in viewer._controllers.values())
    assert send(viewer, 7, mode="clutch", right=True)
    assert send(viewer, 8, mode="clutch", left=True, right=True)
    assert viewer._controllers[Hand.LEFT] is None
    assert viewer._controllers[Hand.RIGHT].grip == 1
    assert send(viewer, 9, mode="clutch", left=False, right=True)
    assert send(viewer, 10, mode="clutch", left=True, right=True)
    assert viewer._controllers[Hand.LEFT].grip == 1


def test_clutch_cannot_cross_assign_arms_or_weaken_tool_proximity(viewer):
    assert not send(viewer, 0, mode="clutch")
    assert send(viewer, 1, mode="clutch", left=False)
    assert not send(
        viewer,
        2,
        mode="clutch",
        left={
            "position": [2, 0, 0],
            "orientation": [0, 0, 0, 1],
            "pinched": True,
            "arm": "right",
            "action": "move",
            "trigger": 0.5,
        },
    )
    assert viewer._claims == {}
    assert not send(viewer, 3, mode="tools")
    assert send(viewer, 4, left=False)
    assert send(viewer, 5, left=True, offset=(2, 0, 0))
    assert viewer._claims == {}


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


def test_replayed_sequence_cannot_reengage(viewer):
    assert send(viewer, 1, left=False)
    assert send(viewer, 2, left=True)
    assert not send(viewer, 2, left=True)
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 4, left=True)
    assert viewer._controllers[Hand.LEFT] is None


def test_delayed_observations_and_missing_depth_do_not_release_a_tracked_tool(viewer, mocker):
    wall = mocker.patch(
        "dimos.robot.manipulators.dual_openyam.spatial_viewer.time.time", return_value=100.0
    )
    clock = mocker.patch(
        "dimos.robot.manipulators.dual_openyam.spatial_viewer.time.monotonic", return_value=0.0
    )
    viewer._feedback["ts"] = 100.0
    viewer._surface = None
    epoch = viewer._control_epoch
    conditions = LinkConditions(
        seed=7,
        to_robot=Delay(base_s=0.1, jitter_std_s=0.02),
        to_operator=Delay(base_s=0.8, jitter_std_s=0.03),
    )
    incoming = DelayTimeline(conditions, "to_robot")
    outgoing = DelayTimeline(conditions, "to_operator")
    queue = DeliveryQueue(max_messages=100, max_bytes=100)
    delivered = []
    for tick in range(225):
        now = tick / 100
        clock.return_value = now
        wall.return_value = 100 + now
        viewer._feedback["ts"] = wall.return_value
        if tick % 2 == 0 and tick < 200:
            observed = wall.return_value - outgoing.delay_at(now)
            queue.push((tick // 2, observed), now=now, delay_s=incoming.delay_at(now), size=1)
        while (delivery := queue.pop_ready(now)) is not None:
            sequence, observed = delivery.value
            assert send(
                viewer,
                sequence,
                left=sequence > 0,
                observed_at=observed,
                control_epoch=epoch,
                offset=(0.04, 0, 0),
            )
            delivered.append(sequence)
        viewer._expire_stale_state(now)
    assert delivered == list(range(100))
    assert viewer._owner(Hand.LEFT, "move") == Hand.LEFT
    assert viewer._get_output_pose(Hand.LEFT).position.x == pytest.approx(0.04)
    assert viewer._control_epoch == epoch


def test_input_stall_invalidates_queued_release_and_regrab_before_recovery(viewer, mocker):
    clock = mocker.patch(
        "dimos.robot.manipulators.dual_openyam.spatial_viewer.time.monotonic", return_value=10.0
    )
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    previous_epoch = viewer._control_epoch
    clock.return_value = 10.4
    # Receive-side expiry also works when no watchdog tick ran during the stall.
    assert not send(viewer, 2, left=False, control_epoch=previous_epoch)
    assert not send(viewer, 3, left=True, control_epoch=previous_epoch)
    assert viewer._controllers[Hand.LEFT] is None
    epoch = viewer._control_epoch
    viewer._expire_stale_state(11)
    assert viewer._control_epoch == epoch == previous_epoch + 1
    assert viewer._hold_reason == "Control input stopped; release and pinch again"
    assert send(viewer, 4, left=True)
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 5, left=False)
    assert send(viewer, 6, left=True)
    assert viewer._controllers[Hand.LEFT].grip
    # A late packet from the invalidated epoch cannot cancel the new grab either.
    assert not send(viewer, 7, left=False, control_epoch=previous_epoch)
    assert viewer._controllers[Hand.LEFT].grip


def test_robot_local_joint_feedback_loss_holds_even_with_continuing_operator_input(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    previous_epoch = viewer._control_epoch
    viewer._feedback["ts"] = time.time() - 1
    assert not send(viewer, 2, left=True, control_epoch=previous_epoch)
    assert viewer._claims == {}
    assert viewer._controllers[Hand.LEFT] is None
    assert viewer._hold_reason == "Robot joint feedback unavailable"
    viewer._feedback["ts"] = time.time()
    assert not send(viewer, 3, left=False, control_epoch=previous_epoch)
    assert send(viewer, 4, left=True)
    assert viewer._controllers[Hand.LEFT] is None
    assert send(viewer, 5, left=False)
    assert send(viewer, 6, left=True)
    assert viewer._controllers[Hand.LEFT].grip


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


def test_rgbd_requires_timestamped_camera_mount_without_gating_robot_teleop(viewer):
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
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    assert viewer._controllers[Hand.LEFT].grip


def test_old_surface_cannot_publish_perception_focus_or_cancel_a_grab(viewer):
    assert send(viewer, 0, left=False)
    assert send(viewer, 1, left=True)
    viewer._surface = replace(viewer._surface, ts=time.time() - 2)
    assert send(viewer, 2, left=True, focus=[0, 0, 0])
    viewer.perception_focus.publish.assert_not_called()
    assert viewer._controllers[Hand.LEFT].grip


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
