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

"""JSON boundary for operator clients of the existing manipulation owner.

No transport, device access or execution state lives here. The module checks
the session before dispatch; native Python clients retain the typed RPCs.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.manipulation.manipulation_spec import PlanResult, PlanStatus
from dimos.manipulation.planning.planners.roboplan_config import RoboPlanCartesianPathConfig
from dimos.manipulation.visualization.operator import (
    CartesianTargetRequest,
    JointTargetRequest,
    ManipulationOperator,
    PoseTargetRequest,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryStatus

if TYPE_CHECKING:
    from dimos.manipulation.manipulation_module import ManipulationModule
    from dimos.manipulation.planning.monitor.world_monitor import WorldMonitor


def operator_json(value: Any) -> Any:
    """Encode the public manipulation values, retaining enum names and precision."""
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        return {str(key): operator_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [operator_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: operator_json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, JointState):
        return {"name": list(value.name), "position": list(value.position)}
    if isinstance(value, PoseStamped):
        return {
            "frame_id": value.frame_id,
            "position": dict(zip(("x", "y", "z"), value.position.to_list(), strict=True)),
            "orientation": dict(
                zip(("x", "y", "z", "w"), value.orientation.to_list(), strict=True)
            ),
        }
    if isinstance(value, JointTrajectory):
        return {
            "joint_names": value.joint_names,
            "duration": value.duration,
            "points": operator_json(value.points),
        }
    if isinstance(value, TrajectoryPoint):
        return {"time_from_start": value.time_from_start, "positions": value.positions}
    if isinstance(value, TrajectoryStatus):
        return {
            "timestamp": value.timestamp,
            "state": operator_json(value.state),
            "progress": value.progress,
            "time_elapsed": value.time_elapsed,
            "time_remaining": value.time_remaining,
            "error": value.error,
        }
    raise TypeError(f"No operator encoding for {type(value).__name__}")


def target_pose(value: dict[str, Any]) -> PoseStamped:
    position = value["position"]
    orientation = value["orientation"]
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError("A pose needs position [x,y,z] and quaternion [x,y,z,w]")
    if not all(math.isfinite(v) for v in (*position, *orientation)):
        raise ValueError("Pose coordinates must be finite")
    if not math.isclose(sum(v * v for v in orientation), 1.0, abs_tol=1e-3):
        raise ValueError("Pose quaternion must be normalized")
    return PoseStamped(
        frame_id=value.get("frame", "world"), position=position, orientation=orientation
    )


def operator_call(
    module: "ManipulationModule",
    world: "WorldMonitor",
    method: str,
    payload: dict[str, Any],
) -> Any:
    """Dispatch only the documented operator operations to their existing owner."""
    if method == "get_state":
        return operator_json(module.get_state())
    if method == "list_planning_groups":
        return operator_json(module.list_planning_groups())
    if method == "get_obstacles":
        return operator_json(module.get_obstacles())
    if method == "model":
        config = module.get_model_config()
        return {
            "xml": config.model.load().xml,
            "base_pose": operator_json(config.base_pose),
            "base_link": config.base_link,
            "joint_names": config.joint_names,
        }
    if method == "execute":
        plan_id = payload.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("Execute requires the inspected plan_id")
        return operator_json(module.execute(blocking=False, plan_id=plan_id))
    if method == "cancel":
        return operator_json(module.cancel())
    if method == "clear_planned_path":
        return operator_json(module.clear_planned_path())
    if method == "plan_to_joints":
        targets = {
            group: JointState(name=list(joints), position=list(joints.values()))
            for group, joints in payload["targets"].items()
        }
        return operator_json(module.plan_to_joints(targets, speed_scale=payload.get("speed")))
    if method == "plan_to_poses":
        poses = {group: target_pose(pose) for group, pose in payload["targets"].items()}
        return operator_json(module.plan_to_poses(poses, speed_scale=payload.get("speed")))
    if method not in {"evaluate", "plan_cartesian"}:
        raise ValueError(f"Unknown operator operation {method!r}")

    # Speed is local to this request, never borrowed from a live visualizer.
    operator = ManipulationOperator(module, world)
    if method == "plan_cartesian":
        speed = payload.get("speed")
        if speed is not None and not operator.set_motion_speed(speed):
            raise ValueError("Speed must be in (0, 1]")
        plan = operator.plan_cartesian(
            CartesianTargetRequest(
                {group: target_pose(pose) for group, pose in payload["poses"].items()},
                RoboPlanCartesianPathConfig(speed_mode="time_optimal", dt=0.05),
                tuple(payload.get("auxiliary", ())),
            )
        )
        return operator_json(
            PlanResult(
                PlanStatus.SUCCEEDED if plan else PlanStatus.FAILED,
                plan.message if plan else operator.status().error or "Cartesian planning failed",
                plan,
            )
        )

    group_ids = tuple(payload["groups"])
    groups = {group.id: group for group in module.list_planning_groups()}
    names = [name for group in group_ids for name in groups[group].joint_names]
    poses = payload.get("poses", {})
    if poses:
        seed = payload.get("seed", {})
        result = operator.evaluate_pose_target(
            PoseTargetRequest(
                {group: target_pose(pose) for group, pose in poses.items()},
                tuple(group for group in group_ids if group not in poses),
                JointState(name=names, position=[seed[name] for name in names])
                if all(name in seed for name in names)
                else None,
            )
        )
    else:
        joints = payload["joints"]
        result = operator.evaluate_joint_target(
            JointTargetRequest(
                group_ids, JointState(name=names, position=[joints[n] for n in names])
            )
        )
    return {
        "success": result.success,
        "collision_free": result.collision_free,
        "status": result.status,
        "message": "Target feasible in the configured model" if result.success else result.message,
        "joints": dict(zip(result.target_joints.name, result.target_joints.position, strict=True))
        if result.target_joints is not None
        else None,
        "poses": {
            group: {
                "position": pose.position.to_list(),
                "orientation": pose.orientation.to_list(),
                "frame": pose.frame_id,
            }
            for group, pose in result.group_poses.items()
            if pose is not None
        },
    }
