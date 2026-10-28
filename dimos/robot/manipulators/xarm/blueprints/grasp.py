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

"""The xArm grasping stacks, on hardware by default.

``dimos run xarm-grasp --config robot.json``            xArm7, heuristic grasps
``dimos run xarm-grasp-graspgenx --config robot.json``  xArm7, learned grasps
``dimos run xarm6-grasp --config robot.json``           xArm6, heuristic grasps
``dimos run xarm6-grasp-graspgenx --config robot.json`` xArm6, learned grasps
``dimos run xarm-grasp --simulation mujoco``            the same stack in MuJoCo

Only the grasp provider separates the two blueprints of one arm, and only the
planning model, the flange the wrist camera hangs from and the MuJoCo scene
separate the arms. The arm-versus-sim split is decided here at import time,
because composition runs before module config is applied. What it swaps is the
hardware adapter, the base pose, the camera and the engine behind it, the
detector backends and the home pose. Everything else -- the coordinator,
pick-and-place, scene registration -- is the same stack either way.
"""

from __future__ import annotations

from dimos.control.components import HardwareComponent
from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import (
    coordinator,
    manipulation_mapping,
    trajectory_task,
)
from dimos.robot.manipulators.xarm.config import (
    XARM6_DOF,
    XARM6_SIM_HOME,
    XARM6_SIM_PATH,
    XARM7_DOF,
    XARM_HEURISTIC_AIM,
    XARM_SIM_BASE_POSE,
    make_xarm_model_config,
    make_xarm_sim_hardware,
    make_xarm_sim_module_kwargs,
    make_xarm_sim_robot_config,
    xarm_flange_link,
    xarm_hardware,
)
from dimos.simulation.engines.mujoco_sim_module import MujocoSimModule
from dimos.utils.data import LfsPath
from dimos.visualization.rerun.bridge import RerunBridgeModule

SIMULATED = bool(global_config.simulation)

XARM_GRASP_SCENE_PATH = LfsPath("xarm_grasp_sim/scene.xml")
# The stock xArm home points the narrow wrist-camera frustum between the widely
# spaced targets. This collision-free top-down pose raises the camera enough to
# put every mesh in one frame, without changing the configured base pose.
XARM_GRASP_SCAN_JOINTS = [0.0, -0.04609, 0.0, 1.83940, 0.0, 1.87106, 0.0]
# The mapper and planner octree use the same resolution.
XARM_GRASP_VOXEL_SIZE = 0.025

XARM_GRASP_PROMPTS = [
    "black bottle",
    "gray can",
    "red cup",
    "green tape roll",
    "blue marker",
    "brown box",
    # The wrist camera sees the tape almost directly from above, where it reads
    # as a ring instead of a roll. Keep a shape-word fallback for that view.
    "green ring",
]
# data/xarm6's table. The cup is the pick target; from above it reads as a white disc.
XARM6_GRASP_PROMPTS = ["red apple", "orange", "white ball"]

# Measured off data/xarm_grasp_sim (mj_forward at the driver joint limits) and the
# gripper URDF, expressed in GraspGenX's convention -- approach along +Z, jaws
# closing along X -- with the origin on xarm_gripper_base_link, the frame
# GraspGenX predicts into. The geometry is the real gripper's, so it holds on
# hardware too.
XARM_GRIPPER_SWEEP_VOLUME = {
    "extents_open": (0.0889, 0.030, 0.0370),
    "offset_open": (0.0, 0.0, 0.1421),
    "extents_half_open": (0.0479, 0.030, 0.0370),
    "offset_half_open": (0.0, 0.0, 0.1530),
    "fingertip_depth": 0.1606,
}
# xarm_gripper_base_link -> link_tcp, the planning tip frame: +0.172 m along the
# approach axis (xarm_gripper.urdf.xacro joint_tcp) plus the quarter turn that
# takes GraspGenX's X closing axis onto the xArm gripper's Y.
XARM_GRASP_FRAME_TO_TCP = (
    (0.0, 1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.172),
    (0.0, 0.0, 0.0, 1.0),
)


def _wrist_camera_transform(dof: int) -> Transform:
    """Hand-eye calibration for the eye-in-hand RealSense, off the arm's flange.

    RealSenseCamera publishes only its own subtree, so without this edge
    camera_link has no parent, nothing resolves into world, and every cloud the
    camera produces is silently unusable. The flange is a frame the model already
    publishes, so ManipulationModule emits the whole chain from one loop at one
    rate.
    """
    return Transform(
        translation=Vector3(x=0.06693724, y=-0.0309563, z=0.00691482),
        rotation=Quaternion(0.70513398, 0.00535696, 0.70897578, -0.01052180),  # xyzw
        frame_id=xarm_flange_link(dof),
        child_frame_id="camera_link",
    )


def _model_and_hardware(
    dof: int, *, sim_base_pose: PoseStamped, sim_home: list[float]
) -> tuple[RobotModelConfig, HardwareComponent]:
    """The planning model and the arm it drives, on hardware or in MuJoCo."""
    if SIMULATED:
        return (
            make_xarm_sim_robot_config(dof, base_pose=sim_base_pose, tf_extra_links=[]),
            make_xarm_sim_hardware(dof, home_joints=sim_home),
        )
    return (
        make_xarm_model_config(
            dof,
            add_gripper=True,
            gripper_hardware_id="arm",
            base_pose=PoseStamped(frame_id="world"),
            tf_extra_links=[xarm_flange_link(dof)],
        ),
        xarm_hardware(dof, "arm", gripper=True),
    )


def _sensing(dof: int, *, scene_path: LfsPath, sim_home: list[float]) -> tuple[Blueprint, ...]:
    """The camera, and in sim the engine that produces its frames."""
    if SIMULATED:
        return (
            MujocoSimModule.blueprint(
                **{
                    **make_xarm_sim_module_kwargs(dof, scene_path),
                    "headless": True,
                    # Publish the simulated camera pose directly in the planning
                    # frame. A wrist-relative TF would need a second
                    # asynchronously stamped world->flange edge and can make an
                    # otherwise valid scan unregistrable.
                    "base_frame_id": "world",
                    "reset_joint_positions": sim_home,
                    # Off by default, and the voxel chain has nothing to map
                    # without it. Scene registration reads colour and depth.
                    "enable_pointcloud": True,
                }
            ),
        )
    return (
        # enable_pointcloud is off by default here too, and the voxel chain has
        # nothing to map without it.
        RealSenseCamera.blueprint(enable_pointcloud=True),
    )


def _scene_registration() -> Blueprint:
    """Detector settings differ: synthetic renders score far below real images."""
    if SIMULATED:
        return ObjectSceneRegistrationModule.blueprint(
            target_frame="world",
            detector_backend="owlv2",
            # OWLv2 is box-only; YOLO-E visual prompts refine its boxes into masks.
            segmentation_backend="yolo",
            detector_confidence=0.07,
            segmentation_confidence=0.05,
            # Keep adjacent tabletop targets distinct instead of merging by label.
            distance_threshold=0.05,
            detect_on_request=True,
            # The obstacle stream carries permanent objects only, so one explicit
            # scan must promote its first sightings immediately.
            min_detections_for_permanent=1,
        )
    return ObjectSceneRegistrationModule.blueprint(
        target_frame="world",
        detector_backend="moondream",
        segmentation_backend="edgetam",
        detect_on_request=True,
        distance_threshold=0.08,
        min_detections_for_permanent=3,
        max_distance=1.0,
        use_aabb=True,
        max_obstacle_width=0.06,
    )


def _grasp_modules(
    dof: int, *, scene_path: LfsPath, sim_base_pose: PoseStamped, sim_home: list[float]
) -> tuple[Blueprint, ...]:
    """Everything but the grasp provider, for one arm.

    Exactly one provider may be composed in: PickAndPlaceModule resolves its
    generator by spec, so two would be ambiguous.
    """
    model, hardware = _model_and_hardware(dof, sim_base_pose=sim_base_pose, sim_home=sim_home)
    return (
        ManipulationModule.blueprint(
            model=model,
            static_transforms=[] if SIMULATED else [_wrist_camera_transform(dof)],
            planning_timeout=10.0,
            visualization={"backend": "viser"},
            world_frame="world",
            voxel_map_resolution=XARM_GRASP_VOXEL_SIZE,
        ),
        ManipulationSkills.blueprint(),
        PickAndPlaceModule.blueprint(planning_frame="world", robot=model),
        *_sensing(dof, scene_path=scene_path, sim_home=sim_home),
        _scene_registration(),
        manipulation_mapping(
            # Filter only the arm; gripper returns may remain in the map.
            self_filter_model=make_xarm_model_config(dof, add_gripper=False).model,
            voxel_size=XARM_GRASP_VOXEL_SIZE,
        ),
        RerunBridgeModule.blueprint(),
        coordinator(
            hardware=[hardware],
            tasks=[
                trajectory_task(hardware),
                TaskConfig(
                    name="arm_gripper",
                    type="gripper",
                    joint_names=["arm/gripper"],
                    priority=20,
                ),
            ],
        ),
    )


# data/xarm_grasp_sim/xarm7.xml bolts link_base to the world origin instead of
# the 12 cm pedestal data/xarm7 uses. Inheriting that offset would put the
# planning model 12 cm above the arm MuJoCo simulates.
_XARM7_GRASP_MODULES = _grasp_modules(
    XARM7_DOF,
    scene_path=XARM_GRASP_SCENE_PATH,
    sim_base_pose=PoseStamped(frame_id="world"),
    sim_home=XARM_GRASP_SCAN_JOINTS,
)
_XARM6_GRASP_MODULES = _grasp_modules(
    XARM6_DOF,
    scene_path=XARM6_SIM_PATH,
    sim_base_pose=XARM_SIM_BASE_POSE,
    sim_home=XARM6_SIM_HOME,
)

_heuristic_grasps = HeuristicGraspModule.blueprint(**XARM_HEURISTIC_AIM)
_graspgenx_grasps = GraspGenXModule.blueprint(
    gripper=XARM_GRIPPER_SWEEP_VOLUME,
    grasp_frame_to_tcp=XARM_GRASP_FRAME_TO_TCP,
    # Return every sample: the pick's contact check chooses among them.
    max_candidates=500,
)

xarm_grasp = autoconnect(*_XARM7_GRASP_MODULES, _heuristic_grasps)
xarm_grasp_graspgenx = autoconnect(*_XARM7_GRASP_MODULES, _graspgenx_grasps)
xarm6_grasp = autoconnect(*_XARM6_GRASP_MODULES, _heuristic_grasps)
xarm6_grasp_graspgenx = autoconnect(*_XARM6_GRASP_MODULES, _graspgenx_grasps)
