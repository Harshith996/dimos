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

from dimos.control.tasks.teleop_ik_task.teleop_ik_task import TeleopIKTaskParams
from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.visualization.viser.config import ViserVisualizationConfig
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    EGOCENTRIC_CAMERA,
    LEFT_WRIST_CAMERA,
    RIGHT_WRIST_CAMERA,
    build_dual_openyam_webxr,
    teleop_webxr_dual_openyam,
)
from dimos.teleop.webxr.camera import TeleopCamera
from dimos.teleop.webxr.extensions import DualArmVideoTeleopModule


def test_visualization_configuration_preserves_teleop_composition(mocker):
    download = mocker.patch(
        "dimos.utils.data.get_data", side_effect=AssertionError("Config must not download models")
    )
    base = teleop_webxr_dual_openyam
    configured = build_dual_openyam_webxr(visualization=ViserVisualizationConfig(host="0.0.0.0"))

    assert configured.remapping_map == base.remapping_map
    assert [atom.module for atom in configured.blueprints] == [
        atom.module for atom in base.blueprints
    ]
    [original] = [atom for atom in base.blueprints if atom.module is ManipulationModule]
    [updated] = [atom for atom in configured.blueprints if atom.module is ManipulationModule]
    assert original.name == updated.name
    assert original.kwargs["visualization"].host == "127.0.0.1"
    assert updated.kwargs == {
        **original.kwargs,
        "visualization": ViserVisualizationConfig(host="0.0.0.0"),
    }
    assert [
        (atom.name, atom.kwargs)
        for atom in configured.blueprints
        if atom.module is not ManipulationModule
    ] == [
        (atom.name, atom.kwargs)
        for atom in base.blueprints
        if atom.module is not ManipulationModule
    ]

    parsed = BlueprintConfigParser(configured).parse(
        ["--manipulationmodule.visualization.host", "127.0.0.1"], environ={}
    )
    assert parsed.module_kwargs(updated.name)["visualization"]["host"] == "127.0.0.1"
    download.assert_not_called()


def test_headset_cameras_feed_their_panels() -> None:
    cameras = {
        atom.name: atom
        for atom in teleop_webxr_dual_openyam.blueprints
        if atom.module is TeleopCamera
    }
    assert set(cameras) == {EGOCENTRIC_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA}
    assert all(atom.kwargs == {"instance_name": atom.name} for atom in cameras.values())
    [video] = [
        atom
        for atom in teleop_webxr_dual_openyam.blueprints
        if atom.module is DualArmVideoTeleopModule
    ]
    panel_inputs = {stream.name for stream in video.streams if stream.direction == "in"}
    remapped = {teleop_webxr_dual_openyam.remapping_map[(name, "color_image")] for name in cameras}
    assert remapped == {"egocentric_image", "left_wrist_image", "right_wrist_image"}
    assert remapped <= panel_inputs

    parsed = BlueprintConfigParser(teleop_webxr_dual_openyam).parse(
        ["--left-wrist-camera.device", "/dev/v4l/by-id/usb-left", "--video-max-fps", "15"],
        environ={},
    )
    assert parsed.module_kwargs(LEFT_WRIST_CAMERA)["device"] == "/dev/v4l/by-id/usb-left"
    assert parsed.module_kwargs(RIGHT_WRIST_CAMERA) == {}
    assert parsed.module_kwargs(video.name)["video_max_fps"] == 15


def test_operator_elbows_steer_the_arms() -> None:
    [coordinator] = [
        atom for atom in teleop_webxr_dual_openyam.blueprints if atom.name == "ControlCoordinator"
    ]
    teleop_tasks = [task for task in coordinator.kwargs["tasks"] if task.type == "teleop_ik"]
    assert len(teleop_tasks) == 2
    for task in teleop_tasks:
        params = TeleopIKTaskParams.model_validate(task.params)
        [binding] = params.bindings
        side = binding.hand.value
        assert binding.target_frame == f"{side}_grasp_frame"
        assert binding.elbow_frame == f"{side}_link3"
        assert params.hint_position_cost == 2.0

    [video] = [
        atom
        for atom in teleop_webxr_dual_openyam.blueprints
        if atom.module is DualArmVideoTeleopModule
    ]
    assert video.kwargs["body_tracking_mode"] == "optional"
    assert {stream.name for stream in coordinator.streams if stream.direction == "in"} >= {
        "left_arm_posture",
        "right_arm_posture",
    }
