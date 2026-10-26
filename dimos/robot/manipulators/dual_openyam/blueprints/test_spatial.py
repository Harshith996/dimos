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

from dataclasses import replace
import json

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.protocol.testing.link import Delay, LinkConditions, Stall
from dimos.robot.assets.model import RobotModel
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.spatial import (
    build_openyam_spatial,
    build_openyam_spatial_sim,
    openyam_spatial,
)
from dimos.robot.manipulators.dual_openyam.collection import DualOpenYamCollection
from dimos.robot.manipulators.dual_openyam.config import dual_openyam_model_config
from dimos.robot.manipulators.dual_openyam.spatial_cameras import OpenYamSpatialCameras
from dimos.robot.manipulators.dual_openyam.spatial_simulation import (
    OpenYamSpatialCoordinator,
    OpenYamSpatialSimulation,
)
from dimos.robot.manipulators.dual_openyam.spatial_viewer import OpenYamSpatialTeleop
from dimos.robot.manipulators.dual_openyam.stations.anvil import ANVIL
from dimos.sim2.sensors.spec import Mount
from dimos.teleop.webxr.extensions import ArmTeleopModule


def test_physical_station_resolves_explicit_devices_and_calibrated_mount_without_opening_them():
    parsed = BlueprintConfigParser(openyam_spatial).parse(
        [
            "--left-can-port",
            "follower_l",
            "--right-can-port",
            "follower_r",
            "--workspace-camera.serial-number",
            "camera-serial",
            "--workspace-mount.frames",
            '[["camera_link","world",[0.2,0,0.8],[0,0.8,0]]]',
            "--dualopenyamcollection.camera-ids",
            '["left-usb","right-usb"]',
            "--openyamspatialteleop.depth-unit-m",
            "0.0005",
        ],
        environ={},
    )
    assert parsed.module_kwargs("ControlCoordinator")["left_can_port"] == "follower_l"
    assert parsed.module_kwargs("workspace_camera")["serial_number"] == "camera-serial"
    assert parsed.module_kwargs("workspace_mount")["frames"][0][2] == (0.2, 0, 0.8)
    assert parsed.module_kwargs("openyamspatialteleop")["depth_unit_m"] == 0.0005
    assert ArmTeleopModule not in [atom.module for atom in openyam_spatial.blueprints]


def test_calibrated_robot_model_is_shared_by_physical_ik_limits_and_display(tmp_path):
    model = dual_openyam_model_config().model_copy(
        update={"model": RobotModel.from_file(tmp_path / "calibrated.urdf")}
    )
    blueprint = build_openyam_spatial(model=model)
    kwargs = {atom.module: atom.kwargs for atom in blueprint.blueprints}
    assert kwargs[OpenYamSpatialTeleop]["robot_model"] == model
    assert kwargs[ManipulationModule]["model"] == model
    coordinator = kwargs[DualOpenYamCoordinator]
    assert coordinator["robot_model"] == model.model
    assert [
        task.params["robot_model"] for task in coordinator["tasks"] if task.type == "teleop_ik"
    ] == [model, model]


def test_station_variant_reaches_physics_cameras_ik_display_and_recording(tmp_path):
    station = replace(
        ANVIL,
        scene=tmp_path / "another_station.xml",
        bases={"left": Mount("world", xyz=(0, 0.4, 0)), "right": Mount("world", xyz=(0, -0.4, 0))},
        wrists={
            side: replace(camera, width=320, height=240) for side, camera in ANVIL.wrists.items()
        },
    )
    blueprint = build_openyam_spatial_sim(station)
    kwargs = {atom.module: atom.kwargs for atom in blueprint.blueprints}
    assert kwargs[OpenYamSpatialSimulation]["station"] == station
    assert kwargs[OpenYamSpatialCoordinator]["base_poses"] == station.bases
    assert kwargs[OpenYamSpatialTeleop]["base_poses"] == station.bases
    assert kwargs[OpenYamSpatialCameras]["sensor"] == station.workspace
    assert kwargs[OpenYamSpatialCameras]["wrists"] == station.wrists
    assert kwargs[DualOpenYamCollection]["width"] == 320
    assert kwargs[DualOpenYamCollection]["height"] == 240


def test_link_conditions_are_optional_application_config_and_parse_from_cli(tmp_path):
    baseline = BlueprintConfigParser(build_openyam_spatial_sim()).parse([], environ={})
    assert baseline.module_kwargs("openyamspatialteleop")["link_conditions"] is None
    conditions = LinkConditions(
        seed=7,
        to_robot=Delay(base_s=0.1, jitter_std_s=0.02),
        stalls=(Stall(at_s=10, duration_s=0.4),),
    )
    blueprint = build_openyam_spatial_sim(link_conditions=conditions)
    kwargs = {atom.module: atom.kwargs for atom in blueprint.blueprints}
    assert kwargs[OpenYamSpatialTeleop]["link_conditions"] == conditions
    assert "link_conditions" not in kwargs[OpenYamSpatialSimulation]
    assert "link_conditions" not in kwargs[OpenYamSpatialCoordinator]
    path = tmp_path / "link.json"
    path.write_text(
        json.dumps(
            {"openyamspatialteleop": {"link_conditions": conditions.model_dump(mode="json")}}
        )
    )
    parsed = BlueprintConfigParser(build_openyam_spatial_sim()).parse(
        ["--openyamspatialteleop.link-conditions.to-operator.base-s", "0.2"],
        config_path=path,
        environ={},
    )
    assert LinkConditions.model_validate(
        parsed.module_kwargs("openyamspatialteleop")["link_conditions"]
    ) == conditions.model_copy(update={"to_operator": Delay(base_s=0.2)})
