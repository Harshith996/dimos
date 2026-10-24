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

"""Pure NumPy boundary for privileged, simulation-only sparse YAM policies.

A policy artifact supplies values exported from its training contract. This module
imports neither the training package, Torch, MuJoCo nor a hardware driver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

SCHEMA = "dimos.openyam.sparse-policy.v1"
STATE_LAYOUT = (
    ("joint_position", 8),
    ("joint_velocity", 8),
    ("cube_position", 3),
    ("cube_quaternion_wxyz", 4),
    ("cube_linear_velocity_world", 3),
    ("cube_angular_velocity_world", 3),
    ("servo_targets", 7),
    ("previous_bounded_action", 7),
    ("goal_position", 3),
    ("elapsed_fraction", 1),
    ("dwell_fraction", 1),
    ("lifted", 1),
)
STATE_DIMENSION = sum(width for _, width in STATE_LAYOUT)
ACTION_DIMENSION = 7


def vector(value: ArrayLike, width: int, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly {width} finite coordinates")
    return result


def quaternion(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = vector(value, 4, name)
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{name} must be a unit WXYZ quaternion")
    return result


@dataclass(frozen=True)
class SparsePolicyContract:
    """Pure-data export; hashes bind weights, robot, task source and training pool.

    Cube/goal positions subtract the translated environment origin, not the robot
    base. Axes remain world-aligned. Cube quaternion and twist remain world WXYZ
    and world-axis velocity. Fixed base pose is relative to that same origin.
    Normalization belongs to the external policy, not this state adapter.
    """

    task_id: str
    policy_sha256: str
    robot_contract_sha256: str
    task_source_sha256: str
    reset_bank_sha256: str
    state_joint_names: tuple[str, ...]
    action_joint_names: tuple[str, ...]
    action_scales: tuple[float, ...]
    target_limits: tuple[tuple[float, float], ...]
    physics_dt: float
    control_decimation: int
    max_episode_steps: int
    required_hold_steps: int
    fixed_base_position: tuple[float, float, float]
    fixed_base_quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for value in (
            self.policy_sha256,
            self.robot_contract_sha256,
            self.task_source_sha256,
            self.reset_bank_sha256,
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Artifact identities must be lowercase SHA-256 digests")
        if (
            not self.task_id
            or len(self.state_joint_names) != 8
            or len(set(self.state_joint_names)) != 8
        ):
            raise ValueError("Expected a named task and eight unique scalar model joints")
        if self.action_joint_names != self.state_joint_names[:ACTION_DIMENSION]:
            raise ValueError("Action joints must be the first seven state joints in order")
        scales = vector(self.action_scales, ACTION_DIMENSION, "action scales")
        limits = np.asarray(self.target_limits, dtype=np.float64)
        if (
            np.any(scales <= 0)
            or limits.shape != (ACTION_DIMENSION, 2)
            or not np.isfinite(limits).all()
        ):
            raise ValueError("Expected seven positive scales and finite target ranges")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Target ranges must have positive width")
        for name in ("control_decimation", "max_episode_steps", "required_hold_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.required_hold_steps > self.max_episode_steps:
            raise ValueError("Dwell exceeds the episode horizon")
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("Physics period must be finite and positive")
        vector(self.fixed_base_position, 3, "fixed base position")
        quaternion(self.fixed_base_quaternion_wxyz, "fixed base quaternion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "simulation_only": True,
            "observation_privilege": "simulator_object_and_task_state",
            "position_frame": "world_axes_translated_environment_origin",
            "quaternion_order": "wxyz",
            "velocity_frame": "world",
            "state_layout": [list(item) for item in STATE_LAYOUT],
            "accumulator_dtype": "float32",
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SparsePolicyContract:
        data = dict(value)
        metadata = {
            "schema": SCHEMA,
            "simulation_only": True,
            "observation_privilege": "simulator_object_and_task_state",
            "position_frame": "world_axes_translated_environment_origin",
            "quaternion_order": "wxyz",
            "velocity_frame": "world",
            "state_layout": [list(item) for item in STATE_LAYOUT],
            "accumulator_dtype": "float32",
        }
        if any(data.pop(name, None) != expected for name, expected in metadata.items()):
            raise ValueError("Unsupported sparse policy layout, frame or privilege contract")
        for name in (
            "state_joint_names",
            "action_joint_names",
            "action_scales",
            "fixed_base_position",
            "fixed_base_quaternion_wxyz",
        ):
            data[name] = tuple(data[name])
        data["target_limits"] = tuple(tuple(v) for v in data["target_limits"])
        return cls(**data)


@dataclass(frozen=True)
class PolicyStamp:
    episode_id: int
    sequence: int
    physics_tick: int
    sim_time: float

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v < 0 for v in (self.episode_id, self.sequence, self.physics_tick)
        ):
            raise ValueError("Policy sample identifiers must be nonnegative integers")
        if not math.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("Policy sample time must be finite and nonnegative")


@dataclass(frozen=True)
class PrivilegedTaskState:
    """Actual same-tick simulator fields missing from a seven-joint robot stream.

    Passive finger state must be measured, not inferred from an equality. The
    caller owns contact predicates and task clocks; this is not a task evaluator.
    """

    stamp: PolicyStamp
    passive_position: float
    passive_velocity: float
    cube_position_world: tuple[float, float, float]
    cube_quaternion_wxyz: tuple[float, float, float, float]
    cube_linear_velocity_world: tuple[float, float, float]
    cube_angular_velocity_world: tuple[float, float, float]
    environment_origin: tuple[float, float, float]
    goal_position_world: tuple[float, float, float]
    base_position_world: tuple[float, float, float]
    base_quaternion_wxyz: tuple[float, float, float, float]
    elapsed_steps: int
    hold_steps: int
    lifted: bool
    terminal: bool


def sparse_state(
    contract: SparsePolicyContract,
    joint_positions: ArrayLike,
    joint_velocities: ArrayLike,
    task: PrivilegedTaskState,
    servo_targets: ArrayLike,
    previous_action: ArrayLike,
) -> NDArray[np.float32]:
    """Assemble the exact unnormalized sparse49 vector from native coordinates."""
    q = vector(joint_positions, ACTION_DIMENSION, "native joint positions")
    qd = vector(joint_velocities, ACTION_DIMENSION, "native joint velocities")
    passive = vector([task.passive_position, task.passive_velocity], 2, "passive finger state")
    origin = vector(task.environment_origin, 3, "environment origin")
    cube = vector(task.cube_position_world, 3, "cube position") - origin
    goal = vector(task.goal_position_world, 3, "goal position") - origin
    base = vector(task.base_position_world, 3, "base position") - origin
    bquat = quaternion(task.base_quaternion_wxyz, "base quaternion")
    expected_quat = np.asarray(contract.fixed_base_quaternion_wxyz)
    if (
        not np.allclose(base, contract.fixed_base_position, atol=2e-5, rtol=0)
        or min(np.linalg.norm(bquat - expected_quat), np.linalg.norm(bquat + expected_quat)) > 2e-5
    ):
        raise ValueError("Sparse policy requires its declared fixed base pose")
    for value, upper, name in (
        (task.elapsed_steps, contract.max_episode_steps, "clock"),
        (task.hold_steps, contract.required_hold_steps, "dwell"),
    ):
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f"Invalid task {name}")
    if type(task.lifted) is not bool or type(task.terminal) is not bool:
        raise ValueError("Task latch and termination must be Boolean")
    targets = vector(servo_targets, ACTION_DIMENSION, "servo preload")
    limits = np.asarray(contract.target_limits)
    # float32 endpoints are the training accumulator's declared representation.
    if np.any(targets < limits[:, 0].astype(np.float32)) or np.any(
        targets > limits[:, 1].astype(np.float32)
    ):
        raise ValueError("Servo preload escapes the policy target limits")
    prior = vector(previous_action, ACTION_DIMENSION, "previous bounded action")
    if np.any(np.abs(prior) > 1):
        raise ValueError("Previous action must be bounded")
    state = np.concatenate(
        (
            q,
            [passive[0]],
            qd,
            [passive[1]],
            cube,
            quaternion(task.cube_quaternion_wxyz, "cube quaternion"),
            vector(task.cube_linear_velocity_world, 3, "cube linear velocity"),
            vector(task.cube_angular_velocity_world, 3, "cube angular velocity"),
            targets,
            prior,
            goal,
            [
                task.elapsed_steps / contract.max_episode_steps,
                task.hold_steps / contract.required_hold_steps,
                float(task.lifted),
            ],
        )
    ).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("State cannot be represented as finite float32")
    state.flags.writeable = False
    return state


def increment_targets(
    contract: SparsePolicyContract,
    current: ArrayLike,
    action: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """One policy/control step, not one physics substep; zero means hold preload."""
    targets = vector(current, ACTION_DIMENSION, "current servo targets").astype(np.float32)
    bounded = vector(action, ACTION_DIMENSION, "policy action").astype(np.float32)
    if np.any(np.abs(vector(action, ACTION_DIMENSION, "policy action")) > 1):
        raise ValueError("Deterministic policy must emit bounded [-1,1] increments")
    limits = np.asarray(contract.target_limits, dtype=np.float32)
    if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
        raise ValueError("Current servo targets are outside the declared range")
    result = np.clip(
        targets + bounded * np.asarray(contract.action_scales, dtype=np.float32),
        limits[:, 0],
        limits[:, 1],
    )
    return result, bounded
