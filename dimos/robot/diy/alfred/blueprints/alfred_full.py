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

"""Alfred, the whole robot: lidar navigation on the coordinator's holonomic follower, the pillar and
both arms planned on that same coordinator, and Desktop's Controller (or the rerun web viewer) as the UI.

    dimos run alfred-full                                                # arms mocked, pillar real
    OPENARM_LEFT_CAN=can2 OPENARM_RIGHT_CAN=can3 dimos run alfred-full   # real arms; `dimos hardware can setup` first

Navigation is the Go2 pattern on Point-LIO odometry: voxel map, MLS global planner, native local
planner, then ``path`` into the coordinator's holonomic pose follower, which runs on Alfred's measured
plant model (alfred_posedomain.json). The follower and teleop (``tele_cmd_vel``) are ordinary joint
claims on the FlowBase's three virtual joints, arbitrated by priority on one coordinator. The
arbitrated twist leaves on ``/flowbase/cmd_vel`` and AlfredHighLevel stays the only FlowBase writer.
The pillar, the OpenArms and the base are planned by the ManipulationModule on the whole-body model in
the robot's own frame (``base_link``): arm targets ride with the robot and never depend on odometry,
and a plan that drives the base is a motion from where the robot stands, which the coordinator's base
task lays down from its odometry pose only while it runs. The Controller's arm view drives that
planner, so there is no viser.

Everything rides the default transport (zenoh), which is what Desktop reads.
"""

from __future__ import annotations

import math
import os
from typing import Any

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import TaskConfig
from dimos.control.path_following_coordinator import PathFollowingCoordinator
from dimos.control.tasks.joint_hold_task.joint_hold_task import joint_hold_task
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JOINT_TRAJECTORY_TASK_NAME,
    joint_trajectory_task,
)
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.pointlio.module import PointLio
from dimos.hardware.sensors.lidar.pointlio.pointlio_blueprints import mid360_for_pointlio
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.relocalization.lidar.module import LocalMapRelocalization
from dimos.mapping.relocalization.lidar.relocalize import MID360_NAV
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.embodiment.alfred import ALFRED
from dimos.navigation.global_planner.mls_planner.mls_planner_native import MLSPlannerNative
from dimos.navigation.global_planner.mls_planner.start_relay import StartRelay
from dimos.navigation.local_planner.native import LocalPlannerNative
from dimos.robot.diy.alfred.alfred_model import (
    ALFRED_ARM_READY,
    ALFRED_MOVING_LINKS,
    ALFRED_PLANAR_BASE,
    ALFRED_V1_MODEL,
    alfred_arm_joints,
    alfred_follower_artifact,
    alfred_planar_model_config,
    alfred_rerun_urdf,
)
from dimos.robot.diy.alfred.blueprints._nav import (
    ALFRED_RERUN_ROOT,
    PLANNER_VIZ_HZ,
    STEP_THRESHOLD_M,
    VOXEL_SIZE_METERS,
    WALL_CLEARANCE_M,
    _alfred_rerun_config,
)
from dimos.robot.diy.alfred.blueprints.alfred_hardware import _alfred_lidar_hardware
from dimos.robot.diy.alfred.blueprints.alfred_nav import MID360_MAX_RANGE_METERS
from dimos.robot.diy.alfred.blueprints.pillar import (
    PILLAR_LIFT_VELOCITY_LIMIT_M_S,
    pillar_motor_transports,
)
from dimos.robot.diy.alfred.channels import channel_spec
from dimos.robot.diy.alfred.config import ALFRED_HARDWARE, LIO_BLIND_M
from dimos.robot.diy.alfred.pillar_connection import (
    PILLAR_LIFT_JOINT,
    PillarConnection,
    pillar_hardware,
)
from dimos.robot.manipulators.common.blueprints import planner
from dimos.robot.manipulators.openarm.config import openarm_hardware
from dimos.robot.urdf_manager import UrdfManager
from dimos.visualization.rerun.urdf_robot import (
    UrdfRobotJointStateRerunFactory,
    UrdfRobotStaticRerunFactory,
)
from dimos.visualization.vis_module import vis_module

OPENARM_LEFT_CAN_ENV = "OPENARM_LEFT_CAN"
OPENARM_RIGHT_CAN_ENV = "OPENARM_RIGHT_CAN"
ARM_VELOCITY_LIMIT_RAD_S = 0.5  # the speed the arms were hardware-tested at

ODOM_FRAME = "odom"
BASE_FRAME = "base_link"

ALFRED_BASE_HARDWARE_ID = "flowbase"
BASE_TRAJECTORY_TASK_NAME = "base_trajectory"
BASE_VELOCITY_TASK_NAME = "vel_flowbase"
NAV_FOLLOWER_TASK_NAME = "holonomic_follower"
# Three claims on the same base joints: a person outranks a plan, a plan outranks navigation.
TELEOP_PRIORITY = 20
PLAN_PRIORITY = 15
NAV_PRIORITY = 10


def _openarm_hardware_from_env() -> HardwareComponent:
    """Real Damiao arms only when both CAN ports are set; the mock adapter otherwise.

    Mock arms start in the ready pose: hanging straight down is a singular configuration
    the planner's IK cannot move a hand out of.
    """
    hardware = openarm_hardware(
        left_can_port=os.environ.get(OPENARM_LEFT_CAN_ENV) or None,
        right_can_port=os.environ.get(OPENARM_RIGHT_CAN_ENV) or None,
    )
    if hardware.adapter_type == "mock_whole_body":
        hardware.adapter_kwargs = {
            "initial_positions": [ALFRED_ARM_READY.get(joint, 0.0) for joint in hardware.joints]
        }
    return hardware


# The coordinator publishes the arbitrated twist through this adapter; AlfredHighLevel
# consumes it and stays the only Portal writer.
_flowbase_hardware = HardwareComponent(
    hardware_id=ALFRED_BASE_HARDWARE_ID,
    hardware_type=HardwareType.BASE,
    joints=make_twist_base_joints(ALFRED_BASE_HARDWARE_ID),
    adapter_type="transport_lcm",
    auto_enable=True,
)
_BASE_VX_LIMIT, _BASE_VY_LIMIT, _BASE_WZ_LIMIT = ALFRED_PLANAR_BASE.velocity_limits


class AlfredNavCoordinator(PathFollowingCoordinator):
    """Carries the ``path`` and ``speed`` ports the follower binds to."""


def alfred_coordinator_tasks() -> list[TaskConfig]:
    """Arms and lift on one trajectory task, a hold on the arms under it, and the base's three claims.

    The lift is left out of the hold: it is a leadscrew under its own brake.
    """
    base_joints = list(_flowbase_hardware.joints)
    return [
        joint_trajectory_task(
            [*alfred_arm_joints(), PILLAR_LIFT_JOINT],
            velocity_limits={
                **dict.fromkeys(alfred_arm_joints(), ARM_VELOCITY_LIMIT_RAD_S),
                PILLAR_LIFT_JOINT: PILLAR_LIFT_VELOCITY_LIMIT_M_S,
            },
        ),
        # The Damiao motors are torque-free until their first command: held from the first
        # tick, the arms stay where they are through a restart instead of sagging.
        joint_hold_task(alfred_arm_joints(), base_joints, name="hold_arms", always=True),
        TaskConfig(
            name=BASE_VELOCITY_TASK_NAME,
            type="velocity",
            joint_names=base_joints,
            priority=TELEOP_PRIORITY,
            # The base coasts to a stop on its own ramp; zeroing on stream gaps fights it.
            params={"zero_on_timeout": False},
        ),
        TaskConfig(
            name=BASE_TRAJECTORY_TASK_NAME,
            type="planar_base_trajectory",
            joint_names=base_joints,
            priority=PLAN_PRIORITY,
            params={
                # The planner works in base_link: its base segment is a motion from where
                # the robot stands, laid down from the odometry pose when it starts.
                "relative": True,
                # The planner limits x and y separately, so a diagonal may exceed either.
                "max_linear": math.hypot(_BASE_VX_LIMIT, _BASE_VY_LIMIT),
                "max_angular": _BASE_WZ_LIMIT,
            },
        ),
        TaskConfig(
            name=NAV_FOLLOWER_TASK_NAME,
            type="holonomic_pose_follower",
            joint_names=base_joints,
            priority=NAV_PRIORITY,
            params={
                "speed": 0.4,
                "goal_tolerance": 0.20,
                "orientation_tolerance": 0.25,
                # Alfred's measured plant model, or the Go2's with a warning.
                "artifact_path": alfred_follower_artifact(),
            },
        ),
    ]


def _alfred_urdf_static(rr: Any) -> list[tuple[str, Any]]:
    factory = UrdfRobotStaticRerunFactory(
        urdf_path=alfred_rerun_urdf(), root_path=ALFRED_RERUN_ROOT
    )
    return [
        *factory(rr),
        (ALFRED_RERUN_ROOT, rr.Transform3D(parent_frame="tf#/base_link")),
    ]


class _AlfredJointStateVisual:
    """Lift and arm links follow coordinator_joint_state; the factory loads on the first message."""

    def __init__(self) -> None:
        self._factory: UrdfRobotJointStateRerunFactory | None = None

    def __call__(self, msg: Any) -> list[tuple[str, Any]]:
        if self._factory is None:
            self._factory = UrdfRobotJointStateRerunFactory(
                urdf_path=alfred_rerun_urdf(),
                root_path=ALFRED_RERUN_ROOT,
                joint_name_mapper=lambda name: name,
            )
        return self._factory(msg)


def _rerun_config() -> dict[str, Any]:
    """The shared Alfred nav view, drawn on the planner's model so the lift and arms move."""
    shared = _alfred_rerun_config()
    return {
        **shared,
        "static": {ALFRED_RERUN_ROOT: _alfred_urdf_static},
        "max_hz": {**shared["max_hz"], "world/coordinator_joint_state": 2.0},
        "visual_override": {
            **shared["visual_override"],
            "world/coordinator_joint_state": _AlfredJointStateVisual(),
        },
    }


# The planner's world is the robot itself. Anything that calls the planning frame "world"
# (the Controller's arm targets, the stock manipulation tooling) resolves through this
# identity edge under base_link, so a target is a pose on the robot: it rides along when the
# base drives and survives odometry jumps, and the planner's model root is the same base_link
# tf draws the robot at (odom's origin is wherever Point-LIO woke up, lidar height included).
# The whole-body model's planar base joints are the base's motion *from here*: zero whenever
# a plan starts, and executed as a relative base trajectory.
_WORLD_IS_BASE = Transform(
    translation=Vector3(0.0, 0.0, 0.0),
    rotation=Quaternion.from_euler(Vector3(0.0, 0.0, 0.0)),
    frame_id=BASE_FRAME,
    child_frame_id="world",
)

alfred_full = (
    autoconnect(
        _alfred_lidar_hardware,
        mid360_for_pointlio(lidar_ip=ALFRED_HARDWARE.mid360_ip),
        PointLio.blueprint(blind=LIO_BLIND_M),
        RayTracingVoxelMap.blueprint(
            voxel_size=VOXEL_SIZE_METERS,
            max_range=MID360_MAX_RANGE_METERS,
            global_emit_every=50,
        ),
        MLSPlannerNative.blueprint(
            voxel_size=VOXEL_SIZE_METERS,
            robot_height=ALFRED.height,
            wall_clearance_m=WALL_CLEARANCE_M,
            step_threshold_m=STEP_THRESHOLD_M,
            step_penalty_weight=1.0,
            viz_publish_hz=PLANNER_VIZ_HZ,
        ).remappings(
            [
                (MLSPlannerNative, "global_map", "global_map_unused"),
                (MLSPlannerNative, "path", "planner_path"),
                # A click is the goal: the Controller's "Go here" and the rerun web viewer both
                # publish clicked_point, and there is no MovementManager here to relay it.
                (MLSPlannerNative, "goal", "clicked_point"),
            ]
        ),
        # Its `path` is what the follower task binds to on the coordinator.
        LocalPlannerNative.blueprint(embodiment=ALFRED, step_speed=0.2),
        # The twist adapter reads the base pose off /flowbase/odom; the relay puts tf's
        # odom -> base_link there.
        StartRelay.blueprint(world_frame=ODOM_FRAME, base_frame=BASE_FRAME),
        LocalMapRelocalization.blueprint(world_frame=ODOM_FRAME, relocalize=MID360_NAV),
        # The rerun web viewer's clicks land on clicked_point like the Controller's, and its
        # keyboard on tele_cmd_vel. Teleop override is handled by priority.
        vis_module(global_config.viewer, rerun_config=_rerun_config()),
        PillarConnection.blueprint(),
        # The robot's URDF for Desktop's Controller: the XML on robot_description, meshes on
        # request. Joint tf stays off (Point-LIO owns mid360_link, AlfredMountTf the mounts;
        # the planner puts the moving links on tf), and the self filter is the perception
        # layer's business, so its cloud ports are parked.
        UrdfManager.blueprint(
            model=ALFRED_V1_MODEL, self_filter=False, publish_joint_tf=False
        ).remappings(
            [
                (UrdfManager, "pointcloud", "urdf_manager_pointcloud_unused"),
                (UrdfManager, "filtered_pointcloud", "urdf_manager_filtered_unused"),
            ]
        ),
        # Whole body in base_link: the planar base joints are not fed from odometry, they
        # are held at zero and a plan moves them from there. The obstacle map
        # alfred-full-perception adds arrives in base_link too.
        planner(
            model=alfred_planar_model_config(tf_extra_links=ALFRED_MOVING_LINKS),
            visualization={"backend": "none"},
            world_frame=BASE_FRAME,
            static_transforms=[_WORLD_IS_BASE],
            joint_state_defaults=dict.fromkeys(ALFRED_PLANAR_BASE.joint_names, 0.0),
            # Splits a whole-body plan; either side aborting cancels the other.
            trajectory_tasks={
                JOINT_TRAJECTORY_TASK_NAME: [*alfred_arm_joints(), PILLAR_LIFT_JOINT],
                BASE_TRAJECTORY_TASK_NAME: list(ALFRED_PLANAR_BASE.joint_names),
            },
        ),
        # Teleop arrives as a Twist onto the base's virtual joints; navigation as a Path on
        # the follower's port. Both are ordinary joint claims from here.
        AlfredNavCoordinator.blueprint(
            instance_name="ControlCoordinator",
            hardware=[pillar_hardware(), _openarm_hardware_from_env(), _flowbase_hardware],
            tasks=alfred_coordinator_tasks(),
        ).remappings([(AlfredNavCoordinator, "twist_command", "tele_cmd_vel")]),
    )
    .transports(
        {
            **pillar_motor_transports(),
            # The coordinator's one arbitrated base command; AlfredHighLevel turns it into wheels.
            ("cmd_vel", Twist): channel_spec(f"/{ALFRED_BASE_HARDWARE_ID}/cmd_vel", Twist),
            ("start_pose", PoseStamped): channel_spec(
                f"/{ALFRED_BASE_HARDWARE_ID}/odom", PoseStamped
            ),
        }
    )
    .global_config(n_workers=12, robot_model="alfred")
)
