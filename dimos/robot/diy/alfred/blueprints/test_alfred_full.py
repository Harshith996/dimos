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

"""alfred-full composes navigation, the pillar and the arms on one coordinator, as commissioned."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, cast
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dimos.control.components import HardwareType
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME
from dimos.core.coordination.blueprints import Blueprint, TransportSpec
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport, ZenohTransport
from dimos.core.transport_factory import transport_topic
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXConfig, GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.planning.utils.roboplan_point_cloud_self_filter import (
    RoboPlanPointCloudSelfFilter,
)
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.relocalization.lidar.module import LocalMapRelocalization
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.navigation.embodiment.alfred import ALFRED
from dimos.navigation.global_planner.mls_planner.mls_planner_native import MLSPlannerNative
from dimos.navigation.local_planner.native import LocalPlannerNative
from dimos.perception.detection.module2D import Detection2DModule
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.module import PerceptionLocalize
from dimos.robot.diy.alfred import alfred_model
from dimos.robot.diy.alfred.alfred_model import (
    ALFRED_ARM_READY,
    ALFRED_FOLLOWER_ARTIFACT,
    ALFRED_FOOTPRINT_RADIUS_M,
    ALFRED_HEIGHT_M,
    ALFRED_LIFT_LOWER_M,
    ALFRED_LIFT_UPPER_M,
    ALFRED_MOVING_LINKS,
    ALFRED_PLANAR_BASE,
    ALFRED_V1_MODEL,
    alfred_arm_joints,
    alfred_follower_artifact,
    alfred_model_config,
    alfred_planar_joint_names,
    alfred_rerun_urdf,
)
from dimos.robot.diy.alfred.blueprints._nav import WALL_CLEARANCE_M
from dimos.robot.diy.alfred.blueprints.alfred_full import (
    ALFRED_BASE_HARDWARE_ID,
    ALFRED_GRIPPER_TASKS,
    BASE_FRAME,
    BASE_TRAJECTORY_TASK_NAME,
    BASE_VELOCITY_TASK_NAME,
    NAV_FOLLOWER_TASK_NAME,
    OPENARM_LEFT_CAN_ENV,
    OPENARM_RIGHT_CAN_ENV,
    AlfredNavCoordinator,
    _openarm_hardware_from_env,
    alfred_full,
)
from dimos.robot.diy.alfred.blueprints.alfred_full_perception import (
    MANIP_VOXEL_MAP,
    alfred_full_perception,
    alfred_perception_modules,
)
from dimos.robot.diy.alfred.blueprints.alfred_manipulation import (
    alfred_manipulation,
    alfred_manipulation_graspgenx,
)
from dimos.robot.diy.alfred.blueprints.alfred_navigation import alfred_navigation
from dimos.robot.diy.alfred.blueprints.alfred_sim import alfred_sim
from dimos.robot.diy.alfred.effector_high_level import AlfredHighLevel
from dimos.robot.diy.alfred.gripper import (
    OPENARM_V10_FINGERTIP_DEPTH_M,
    OPENARM_V10_GRASP_FRAME_TO_TCP,
    OPENARM_V10_SWEEP_VOLUME,
)
from dimos.robot.diy.alfred.pillar_connection import (
    PILLAR_HARDWARE_ID,
    PILLAR_LIFT_JOINT,
    PillarConnection,
)
from dimos.robot.diy.alfred.wheel_odometry_relay import WheelOdometryRelay
from dimos.robot.manipulators.openarm.config import OPENARM_HARDWARE_ID
from dimos.robot.urdf_manager import UrdfManager
from dimos.utils.gpu_utils import is_cuda_available
from dimos.visualization.rerun.bridge import RerunBridgeModule
from dimos.visualization.rerun.websocket_server import RerunWebSocketServer


def _atoms(blueprint: Blueprint, module: type) -> list[Any]:
    return [
        atom
        for atom in blueprint.blueprints
        if isinstance(atom.module, type) and issubclass(atom.module, module)
    ]


def _coordinator_kwargs(blueprint: Blueprint) -> dict[str, Any]:
    (atom,) = _atoms(blueprint, ControlCoordinator)
    return cast("dict[str, Any]", atom.kwargs)


def _tasks(blueprint: Blueprint) -> dict[str, TaskConfig]:
    tasks = cast("list[TaskConfig]", _coordinator_kwargs(blueprint)["tasks"])
    return {task.name: task for task in tasks}


def _channel(blueprint: Blueprint, key: tuple[str, type]) -> tuple[type, str]:
    """The backend class and topic a stream is pinned to."""
    spec = blueprint.transport_map[key]
    assert isinstance(spec, TransportSpec)
    return spec.cls, cast("str", spec.args[0])


# The urdf is LFS data, so these run where it is pulled (like test_alfred.py).
needs_lfs = pytest.mark.self_hosted


def test_alfred_full_composes_nav_pillar_arms_and_the_viewer() -> None:
    modules = {atom.module for atom in alfred_full.blueprints}
    assert {
        AlfredHighLevel,
        PointLio,
        RayTracingVoxelMap,
        MLSPlannerNative,
        LocalPlannerNative,
        LocalMapRelocalization,
        PillarConnection,
        ManipulationModule,
        AlfredNavCoordinator,
        RerunWebSocketServer,
    } <= modules


def test_the_flowbase_is_arbitrated_on_the_coordinator_but_written_by_alfred_high_level() -> None:
    """The coordinator plans for the base but must never be a second Portal writer.

    Its base is a transport adapter: the arbitrated twist goes out on /flowbase/cmd_vel,
    AlfredHighLevel's cmd_vel is pinned to that channel, and the adapter reads the base
    pose back off /flowbase/odom, which StartRelay fills from tf.
    """
    hardware = {hw.hardware_id: hw for hw in _coordinator_kwargs(alfred_full)["hardware"]}
    assert set(hardware) == {PILLAR_HARDWARE_ID, OPENARM_HARDWARE_ID, ALFRED_BASE_HARDWARE_ID}
    base = hardware[ALFRED_BASE_HARDWARE_ID]
    assert base.hardware_type is HardwareType.BASE
    assert base.adapter_type == "transport_lcm", "a direct FlowBase adapter would double-write"

    backend = ZenohTransport if global_config.transport == "zenoh" else LCMTransport
    assert _channel(alfred_full, ("cmd_vel", Twist)) == (
        backend,
        transport_topic(f"/{ALFRED_BASE_HARDWARE_ID}/cmd_vel"),
    )
    assert _channel(alfred_full, ("start_pose", PoseStamped)) == (
        backend,
        transport_topic(f"/{ALFRED_BASE_HARDWARE_ID}/odom"),
    )


def test_the_pillar_meets_its_adapter_on_the_active_backend() -> None:
    backend = ZenohTransport if global_config.transport == "zenoh" else LCMTransport
    for stream, msg_type in (("motor_command", MotorCommandArray), ("motor_states", JointState)):
        assert _channel(alfred_full, (stream, msg_type)) == (
            backend,
            transport_topic(f"/{PILLAR_HARDWARE_ID}/{stream}"),
        )


def test_teleop_enters_the_coordinator_not_the_base() -> None:
    """Whatever publishes tele_cmd_vel (the Controller, the web viewer) is one base claim."""
    assert alfred_full.remapping_map[("ControlCoordinator", "twist_command")] == "tele_cmd_vel"


def test_base_claims_rank_teleop_over_plan_over_navigation() -> None:
    tasks = _tasks(alfred_full)
    teleop = tasks[BASE_VELOCITY_TASK_NAME]
    plan = tasks[BASE_TRAJECTORY_TASK_NAME]
    nav = tasks[NAV_FOLLOWER_TASK_NAME]
    base_joints = [f"{ALFRED_BASE_HARDWARE_ID}/{axis}" for axis in ("vx", "vy", "wz")]
    for task in (teleop, plan, nav):
        assert task.joint_names == base_joints
    assert teleop.priority > plan.priority > nav.priority
    assert teleop.type == "velocity"
    assert plan.type == "planar_base_trajectory"
    assert plan.params["relative"] is True, "a plan's base segment is a motion from here"
    assert nav.type == "holonomic_pose_follower"
    claimants = [name for name, task in tasks.items() if set(task.joint_names) & set(base_joints)]
    assert sorted(claimants) == sorted(
        [BASE_VELOCITY_TASK_NAME, BASE_TRAJECTORY_TASK_NAME, NAV_FOLLOWER_TASK_NAME]
    )


def test_the_follower_runs_on_alfreds_measured_plant() -> None:
    """The artifact is config, not a build output: losing it degrades silently to Go2 gains."""
    nav = _tasks(alfred_full)[NAV_FOLLOWER_TASK_NAME]
    assert nav.params["artifact_path"] == ALFRED_FOLLOWER_ARTIFACT
    artifact = json.loads(Path(ALFRED_FOLLOWER_ARTIFACT).read_text())
    assert artifact["provenance"]["robot_id"] == "alfred"
    assert artifact["valid_for_tuning"]
    assert all(artifact["plant"][axis]["K"] > 0 for axis in ("vx", "vy", "wz"))
    assert alfred_follower_artifact() == ALFRED_FOLLOWER_ARTIFACT


def test_the_follower_falls_back_to_the_go2_artifact_until_alfred_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot that has never been characterized must still start - loudly."""
    missing = tmp_path / "alfred_posedomain.json"
    monkeypatch.setattr(alfred_model, "ALFRED_FOLLOWER_ARTIFACT", str(missing))
    assert alfred_model.alfred_follower_artifact() == alfred_model._GO2_FOLLOWER_ARTIFACT
    missing.write_text("{}")
    assert alfred_model.alfred_follower_artifact() == str(missing)


def test_one_trajectory_task_covers_the_lift_and_both_arms_with_the_hold_under_it() -> None:
    tasks = _tasks(alfred_full)
    trajectory = tasks[JOINT_TRAJECTORY_TASK_NAME]
    hold = tasks["hold_arms"]
    assert trajectory.joint_names == [*alfred_arm_joints(), PILLAR_LIFT_JOINT]
    assert set(trajectory.params["velocity_limits"]) == set(trajectory.joint_names)
    assert hold.type == "joint_hold"
    assert hold.joint_names == alfred_arm_joints(), (
        "the lift holds itself: a leadscrew with a brake"
    )
    assert hold.priority < trajectory.priority
    assert hold.params["always"] is True, "Damiao motors are torque-free until commanded"
    for task_name in ALFRED_GRIPPER_TASKS.values():
        gripper = tasks[task_name]
        assert gripper.type == "gripper"
        assert gripper.joint_names in (["left_arm/gripper"], ["right_arm/gripper"])
        assert gripper.priority > trajectory.priority, "a hand command beats a trajectory's hold"
    (planner,) = _atoms(alfred_full, ManipulationModule)
    assert planner.kwargs["gripper_tasks"] == ALFRED_GRIPPER_TASKS, "one hand per arm"


def test_the_whole_body_is_planned_in_the_robots_own_frame() -> None:
    """Targets are poses on the robot (base_link), so they ride with it and never depend on
    where Point-LIO woke up or whether its odometry jumps; "world" is an identity edge under
    base_link for the Controller and the stock tooling. The planar base joints are the base's
    motion from here: held at zero, never fed from odometry, executed as a relative trajectory."""
    (atom,) = _atoms(alfred_full, ManipulationModule)
    assert atom.kwargs["world_frame"] == BASE_FRAME
    (world,) = atom.kwargs["static_transforms"]
    assert (world.frame_id, world.child_frame_id) == (BASE_FRAME, "world")
    assert (world.translation.x, world.translation.y, world.translation.z) == (0.0, 0.0, 0.0)
    assert list(atom.kwargs["model"].joint_names) == alfred_planar_joint_names()
    assert "joint_state_aliases" not in atom.kwargs
    assert atom.kwargs["joint_state_defaults"] == dict.fromkeys(ALFRED_PLANAR_BASE.joint_names, 0.0)
    assert atom.kwargs["trajectory_tasks"] == {
        JOINT_TRAJECTORY_TASK_NAME: [*alfred_arm_joints(), PILLAR_LIFT_JOINT],
        BASE_TRAJECTORY_TASK_NAME: list(ALFRED_PLANAR_BASE.joint_names),
    }
    assert atom.kwargs["visualization"] == {"backend": "none"}, "the Controller is the arm UI"


def test_the_arms_are_mock_until_both_can_ports_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENARM_LEFT_CAN_ENV, raising=False)
    monkeypatch.delenv(OPENARM_RIGHT_CAN_ENV, raising=False)
    assert _openarm_hardware_from_env().adapter_type == "mock_whole_body"
    monkeypatch.setenv(OPENARM_LEFT_CAN_ENV, "can2")
    monkeypatch.setenv(OPENARM_RIGHT_CAN_ENV, "can3")
    assert _openarm_hardware_from_env().adapter_type == "openarm_damiao"


def test_the_planner_gets_the_real_height_and_a_footprint_under_the_robot() -> None:
    """robot_height decides whether a cell is floor at all (surfaces.rs is_standable), so
    tables and sub-2 m doorways read as blocked; wall_clearance_m is knowingly below the
    footprint radius - 0.2 plans on the robot, 0.28 cannot."""
    (planner,) = _atoms(alfred_full, MLSPlannerNative)
    assert planner.kwargs["robot_height"] == ALFRED.height == ALFRED_HEIGHT_M
    assert planner.kwargs["wall_clearance_m"] == WALL_CLEARANCE_M
    assert WALL_CLEARANCE_M < ALFRED_FOOTPRINT_RADIUS_M


def test_alfred_full_perception_feeds_the_planner_a_filtered_d455_map() -> None:
    """A second, short-range voxel map under its own name, so the nav map's streams stay put."""
    modules = {atom.module for atom in alfred_full_perception.blueprints}
    assert {RealSenseCamera, RoboPlanPointCloudSelfFilter, PointCloudBoxFilter} <= modules
    assert Detection2DModule not in modules, "the perception stack's localize replaced YOLO"
    assert (PerceptionLocalize in modules) == is_cuda_available(), "localize needs the GPU"
    assert len(_atoms(alfred_full_perception, RayTracingVoxelMap)) == 2
    remaps = alfred_full_perception.remapping_map
    assert remaps[(MANIP_VOXEL_MAP, "lidar")] == "filtered_pointcloud"
    assert remaps[(MANIP_VOXEL_MAP, "global_map")] == "manip_global_map"
    assert remaps[(PointCloudBoxFilter.name, "cloud")] == "manip_global_map"
    assert remaps[(PointCloudBoxFilter.name, "filtered_cloud")] == "voxel_map"
    manip_map = next(
        atom
        for atom in _atoms(alfred_full_perception, RayTracingVoxelMap)
        if atom.kwargs.get("instance_name") == MANIP_VOXEL_MAP
    )
    assert manip_map.kwargs["world_frame"] == BASE_FRAME, "the planner's frame, not odom"
    assert manip_map.kwargs["max_range"] <= 1.5, "the bench, not the floor beyond it"
    assert manip_map.kwargs["grace_depth"] <= 0.05 and manip_map.kwargs["graze_cos"] <= 0.3, (
        "clearing must reach a bench seen at 25-54 degrees"
    )
    assert manip_map.kwargs["global_emit_every"] >= 5, "the planner rebuilds its octree per map"
    (self_filter,) = _atoms(alfred_full_perception, RoboPlanPointCloudSelfFilter)
    assert self_filter.kwargs["padding_m"] >= 0.044, "an open finger is 4.4 cm outside the model"
    (camera,) = _atoms(alfred_full_perception, RealSenseCamera)
    assert camera.kwargs["enable_pointcloud"] and camera.kwargs["enable_color"]
    localize = next(
        atom
        for fragment in alfred_perception_modules(gpu=True)
        for atom in fragment.blueprints
        if atom.module is PerceptionLocalize
    )
    assert localize.kwargs["world_frame"] == BASE_FRAME, "a found object is an arm target"
    assert localize.kwargs["geometry"] == "depth" and localize.kwargs["mobile"] is False
    assert localize.kwargs["params"].min_views == 1, "one fixed camera: one viewpoint"
    assert "float16" not in localize.kwargs["detector"], "the registry's fp16 build is for a GPU"
    assert localize.kwargs["segmenter"] == "none", "EdgeTAM needs a GPU"
    with_gpu = {atom.module for f in alfred_perception_modules(gpu=True) for atom in f.blueprints}
    assert {ObjectSceneRegistrationModule, PickAndPlaceModule, HeuristicGraspModule} <= with_gpu
    without = {atom.module for f in alfred_perception_modules(gpu=False) for atom in f.blueprints}
    assert not ({ObjectSceneRegistrationModule, PickAndPlaceModule} & without)
    scene = next(
        atom
        for f in alfred_perception_modules(gpu=True)
        for atom in f.blueprints
        if atom.module is ObjectSceneRegistrationModule
    )
    assert scene.kwargs["target_frame"] == BASE_FRAME, "objects where the arm plans"
    assert scene.kwargs["detector_confidence"] < 0.6, "OWLv2 scores 0.3-0.6 on this bench"
    assert scene.kwargs["detect_on_request"] is True, "a scan, not a stream"
    pick = next(
        atom
        for f in alfred_perception_modules(gpu=True)
        for atom in f.blueprints
        if atom.module is PickAndPlaceModule
    )
    for side in ("left", "right"):
        reach = pick.kwargs["reach"][f"{side}_manipulator"]
        assert reach.shoulder_frame == f"openarm_{side}_link1", "a frame the planner puts on tf"
        assert 0.40 <= reach.radius <= 0.45 and abs(reach.wrist_to_tip - 0.186) < 1e-9
    assert not alfred_perception_modules(gpu=False) or all(
        atom.module is not PerceptionLocalize
        for fragment in alfred_perception_modules(gpu=False)
        for atom in fragment.blueprints
    ), "on a CPU-only torch localize would peg the box"


def test_the_bench_stack_runs_without_the_lidar() -> None:
    """alfred-manipulation: arms, lift, planner, camera; the base pose from wheel odometry."""
    modules = {atom.module for atom in alfred_manipulation.blueprints}
    assert {
        AlfredHighLevel,
        WheelOdometryRelay,
        PillarConnection,
        ManipulationModule,
        AlfredNavCoordinator,
        RealSenseCamera,
        RoboPlanPointCloudSelfFilter,
    } <= modules
    assert not {PointLio, MLSPlannerNative, LocalPlannerNative, LocalMapRelocalization} & modules
    assert PerceptionLocalize not in modules, "the bench picks through the scene registration"
    tasks = _tasks(alfred_manipulation)
    assert NAV_FOLLOWER_TASK_NAME not in tasks, "no path to follow here"
    assert {BASE_VELOCITY_TASK_NAME, BASE_TRAJECTORY_TASK_NAME, "hold_arms"} <= set(tasks)
    assert alfred_manipulation.remapping_map[("ControlCoordinator", "twist_command")] == (
        "tele_cmd_vel"
    )
    (relay,) = _atoms(alfred_manipulation, WheelOdometryRelay)
    assert (relay.kwargs["world_frame"], relay.kwargs["base_frame"]) == ("odom", BASE_FRAME)


def test_the_driving_stack_runs_without_the_arms() -> None:
    """alfred-navigation: the lidar chain, teleop and the follower; the pillar parked, no planner."""
    modules = {atom.module for atom in alfred_navigation.blueprints}
    assert {
        PointLio,
        MLSPlannerNative,
        LocalPlannerNative,
        AlfredNavCoordinator,
        PillarConnection,
    } <= (modules)
    assert not {ManipulationModule, RealSenseCamera, WheelOdometryRelay} & modules
    hardware_ids = {hw.hardware_id for hw in _coordinator_kwargs(alfred_navigation)["hardware"]}
    assert ALFRED_BASE_HARDWARE_ID in hardware_ids
    assert not any("openarm" in hw_id or "arm" in hw_id for hw_id in hardware_ids), "no arms' CAN"
    tasks = _tasks(alfred_navigation)
    assert {BASE_VELOCITY_TASK_NAME, NAV_FOLLOWER_TASK_NAME} <= set(tasks)
    assert "hold_arms" not in tasks
    assert alfred_navigation.remapping_map[(MLSPlannerNative.name, "goal")] == "clicked_point"


def test_alfred_sim_still_composes() -> None:
    hardware_ids = {hw.hardware_id for hw in _coordinator_kwargs(alfred_sim)["hardware"]}
    assert {PILLAR_HARDWARE_ID, OPENARM_HARDWARE_ID} <= hardware_ids


def test_alfred_model_uses_the_pillar_joint_convention() -> None:
    """Lift is negative below the top switch, exactly as pillar_connection reports it."""
    config = alfred_model_config()
    assert config.joint_names[0] == PILLAR_LIFT_JOINT
    assert ALFRED_LIFT_LOWER_M == -0.5
    assert ALFRED_LIFT_UPPER_M == -0.002
    groups = {group.name: group for group in config.planning_groups}
    assert groups["lift"].joint_names == (PILLAR_LIFT_JOINT,)
    assert set(groups) == {"lift", "left_manipulator", "right_manipulator"}


@needs_lfs
def test_home_pose_is_inside_every_joint_limit() -> None:
    """The lift's zero is the top switch, above its reachable range; Home must not send it."""
    for wheels in (False, True):
        config = alfred_model_config(wheels=wheels)
        limits = {j.name: j for j in config.model.load().joints}
        assert config.home_joints is not None
        for name, value in zip(config.joint_names, config.home_joints, strict=True):
            joint = limits[name]
            if joint.lower is not None and joint.upper is not None:
                assert joint.lower <= value <= joint.upper, f"{name}: home {value} outside limits"


@needs_lfs
def test_rerun_urdf_is_materialized_with_resolved_meshes_and_coordinator_joints() -> None:
    xml = alfred_rerun_urdf().read_text()
    assert "package://" not in xml
    assert 'name="pillar/lift"' in xml, "renamed lift joint must reach the rerun model"
    meshes = re.findall(r'filename="([^"]+)"', xml)
    assert meshes and all(Path(m).is_file() for m in meshes), "unresolved mesh path"


def test_the_controller_gets_the_urdf_without_a_second_tf_tree() -> None:
    """UrdfManager publishes the XML and serves meshes; Point-LIO and the mount tree keep tf,
    and the planner puts the moving links on it."""
    (urdf,) = _atoms(alfred_full, UrdfManager)
    assert urdf.kwargs["publish_joint_tf"] is False
    assert urdf.kwargs["self_filter"] is False, "the perception layer filters clouds"
    (planner,) = _atoms(alfred_full, ManipulationModule)
    assert planner.kwargs["model"].tf_extra_links == ALFRED_MOVING_LINKS


@needs_lfs
def test_the_moving_links_are_every_link_under_a_movable_joint() -> None:
    root = ElementTree.fromstring(ALFRED_V1_MODEL.load().xml)
    under_movable = [
        joint.find("child").attrib["link"]  # type: ignore[union-attr]
        for joint in root.findall("joint")
        if joint.attrib["type"] != "fixed"
    ]
    assert ALFRED_MOVING_LINKS == under_movable


def test_a_click_is_the_goal() -> None:
    """The Controller's "Go here" and the rerun viewer publish clicked_point; with no
    MovementManager to relay it, the planner listens there itself."""
    assert alfred_full.remapping_map[(MLSPlannerNative.name, "goal")] == "clicked_point"
    assert (RerunWebSocketServer.name, "clicked_point") not in alfred_full.remapping_map


def test_mock_arms_start_in_the_ready_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-zero arms hang straight down, a singular pose the planner's IK cannot leave."""
    monkeypatch.delenv(OPENARM_LEFT_CAN_ENV, raising=False)
    monkeypatch.delenv(OPENARM_RIGHT_CAN_ENV, raising=False)
    hardware = _openarm_hardware_from_env()
    start = dict(zip(hardware.joints, hardware.adapter_kwargs["initial_positions"], strict=True))
    assert {name: start[name] for name in ALFRED_ARM_READY} == ALFRED_ARM_READY
    assert all(value == 0.0 for name, value in start.items() if name not in ALFRED_ARM_READY)
    home = dict(
        zip(alfred_model_config().joint_names, alfred_model_config().home_joints or [], strict=True)
    )
    assert {name: home[name] for name in ALFRED_ARM_READY} == ALFRED_ARM_READY, (
        "home is the ready pose"
    )


def test_the_agent_drives_the_same_stack_the_controller_does() -> None:
    from dimos.agents.mcp.mcp_client import McpClient
    from dimos.agents.mcp.mcp_server import McpServer
    from dimos.agents.skills.point_nav import PointNavSkillContainer
    from dimos.manipulation.manipulation_skills import ManipulationSkills
    from dimos.robot.diy.alfred.blueprints.alfred_full_agentic import alfred_full_agentic
    from dimos.robot.diy.alfred.system_prompt import ALFRED_SYSTEM_PROMPT

    modules = {atom.module for atom in alfred_full_agentic.blueprints}
    assert {McpServer, McpClient, PointNavSkillContainer, ManipulationSkills} <= modules
    (client,) = _atoms(alfred_full_agentic, McpClient)
    assert client.kwargs["system_prompt"] is ALFRED_SYSTEM_PROMPT
    # go_to publishes where "Go here" does, so one planner input serves both.
    assert (
        alfred_full_agentic.remapping_map[(PointNavSkillContainer.name, "goal")] == "clicked_point"
    )
    (nav,) = _atoms(alfred_full_agentic, PointNavSkillContainer)
    assert nav.kwargs["world_frame"] == "odom"


@needs_lfs
def test_an_arm_plans_on_the_planar_whole_body_model() -> None:
    """RoboPlan's native RRT rejects every start on a model with the planar base's line and
    circle joints ("Invalid start configuration"), whatever the selection; the planner must
    route such a model to the shared Python RRT, which bounds them itself."""
    from dimos.manipulation.planning.planners.roboplan_config import RoboPlanPlannerConfig
    from dimos.manipulation.planning.planners.roboplan_planner import RoboPlanPlanner
    from dimos.manipulation.planning.spec.validation import prepare_robot_model
    from dimos.manipulation.planning.world.roboplan_world import RoboPlanWorld
    from dimos.msgs.sensor_msgs.JointState import JointState
    from dimos.robot.diy.alfred.alfred_model import alfred_planar_model_config

    config = alfred_planar_model_config()
    world = RoboPlanWorld()
    world.load_model(prepare_robot_model(config))
    world.finalize()
    assert config.home_joints is not None
    world.sync_from_joint_state(
        JointState(name=list(config.joint_names), position=config.home_joints)
    )
    planner = RoboPlanPlanner(world, RoboPlanPlannerConfig())
    selection = world._planning_groups.select(("left_manipulator",))
    assert planner._selection_requires_shared_planner(selection)

    left = [name for name in config.joint_names if name.startswith("openarm_left")]
    home = dict(zip(config.joint_names, config.home_joints, strict=True))
    start = JointState(name=left, position=[home[name] for name in left])
    goal = JointState(
        name=left,
        position=[home[name] + (0.2 if name.endswith("joint1") else 0.0) for name in left],
    )
    result = planner.plan_selected_joint_path(
        world=world, selection=selection, start=start, goal=goal, timeout=5.0
    )
    assert result.is_success(), result.message


def test_graspgenx_sees_the_v10_gripper_the_urdf_carries() -> None:
    """GraspGenX's gripper frame is the wrist turned half a turn about x (approach +z, jaws
    along x); its sweep volume and the step to grasp_frame come from the URDF's numbers."""
    config = GraspGenXConfig.model_validate(
        {"gripper": OPENARM_V10_SWEEP_VOLUME, "grasp_frame_to_tcp": OPENARM_V10_GRASP_FRAME_TO_TCP}
    )
    # The URDF: ee_base_link -> grasp_frame is rpy (pi, 0, pi/2) at (0, 0, -0.186).
    wrist_to_tcp = np.eye(4)
    wrist_to_tcp[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0.0, np.pi / 2]).as_matrix()
    wrist_to_tcp[:3, 3] = [0.0, 0.0, -0.186]
    wrist_to_graspgenx = np.eye(4)
    wrist_to_graspgenx[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    expected = np.linalg.inv(wrist_to_graspgenx) @ wrist_to_tcp
    assert np.allclose(np.asarray(config.grasp_frame_to_tcp), expected, atol=1e-9)
    # Approach +z, closing along x, the way GraspGenX wants them.
    tcp = np.asarray(config.grasp_frame_to_tcp)
    assert np.allclose(tcp[:3, 2], [0.0, 0.0, 1.0]), "grasp_frame's z is the approach"
    assert np.allclose(tcp[:3, 1], [1.0, 0.0, 0.0]), "grasp_frame's y is across the jaw"
    gripper = config.gripper
    jaw_open, pad_width, pad_length = gripper.extents_open
    assert 0.080 < jaw_open < 0.088, "88 mm of slider travel less the pads' overlap"
    assert gripper.extents_half_open[0] < jaw_open / 2 + 1e-9
    assert gripper.extents_half_open[1:] == (pad_width, pad_length), "prismatic: pads only slide"
    assert gripper.offset_open == gripper.offset_half_open
    assert gripper.fingertip_depth == OPENARM_V10_FINGERTIP_DEPTH_M
    far_face = gripper.offset_open[2] + pad_length / 2
    assert abs(far_face - gripper.fingertip_depth) < 1e-9, "the box ends at the fingertips"
    assert gripper.offset_open[2] < 0.186 < gripper.fingertip_depth, "the TCP sits in the pads"
    assert gripper.family == "parallel_2f"


def test_the_bench_stack_can_run_graspgenx_instead_of_the_heuristic() -> None:
    """alfred-manipulation-graspgenx: the same stack, GraspGenX answering propose_grasps."""
    heuristic = {atom.module for atom in alfred_manipulation.blueprints}
    assert HeuristicGraspModule in heuristic and GraspGenXModule not in heuristic
    learned = {atom.module for atom in alfred_manipulation_graspgenx.blueprints}
    assert GraspGenXModule in learned and HeuristicGraspModule not in learned
    assert learned - {GraspGenXModule} == heuristic - {HeuristicGraspModule}, "only the generator"
    (graspgenx,) = _atoms(alfred_manipulation_graspgenx, GraspGenXModule)
    assert graspgenx.kwargs["gripper"] == OPENARM_V10_SWEEP_VOLUME
    assert graspgenx.kwargs["grasp_frame_to_tcp"] == OPENARM_V10_GRASP_FRAME_TO_TCP
    assert GraspGenXModule not in {
        atom.module
        for f in alfred_perception_modules(gpu=False, grasp="graspgenx")
        for atom in f.blueprints
    }, "GraspGenX is a GPU model"


def test_the_rerun_bridge_keeps_little_for_late_viewers() -> None:
    """A quarter of the Orin's RAM (the bridge's default) is what the models run in: that
    worker reached 6.4 GB and was OOM-killed, with the box thrashing first."""
    (bridge,) = _atoms(alfred_full, RerunBridgeModule)
    assert bridge.kwargs["memory_limit"] == "256MB"


@pytest.mark.parametrize("stack", [alfred_manipulation, alfred_navigation], ids=["bench", "nav"])
def test_the_split_stacks_run_no_rerun_bridge(stack: Blueprint) -> None:
    """The Desktop reads zenoh itself; the bridge only cost the Orin memory (the OOM above)."""
    modules = {atom.module for atom in stack.blueprints}
    assert not ({RerunBridgeModule, RerunWebSocketServer} & modules)
