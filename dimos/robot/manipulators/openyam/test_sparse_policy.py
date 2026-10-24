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

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.robot.manipulators.openyam.simulation import (
    GRIPPER_STROKE,
    MODEL_JOINTS,
    PASSIVE_JOINT,
    PUBLIC_JOINTS,
    TARGET_LIMITS,
    YamArtifact,
    YamContract,
)
from dimos.robot.manipulators.openyam.sparse_policy import (
    PrivilegedTaskState,
    SparsePolicyContract,
    increment_targets,
    sparse_state,
)
from dimos.robot.manipulators.openyam.sparse_sim2 import SparseSimPolicyAdapter, policy_stamp
from dimos.sim2.messages import FrameMetadata, StateFrame, StreamDescriptor

if TYPE_CHECKING:
    from dimos.sim2.client import SimRobotClient


@pytest.fixture
def contract() -> SparsePolicyContract:
    # Static ABI fixture from frozen sparse49 TaskSpec/action configuration;
    # hashes identify fixture data, not an exported or successful policy.
    return SparsePolicyContract(
        task_id="openyam-lift-hold-sparse-v2",
        policy_sha256="1" * 64,
        robot_contract_sha256="2" * 64,
        task_source_sha256="3" * 64,
        reset_bank_sha256="4" * 64,
        state_joint_names=(*MODEL_JOINTS, PASSIVE_JOINT),
        action_joint_names=MODEL_JOINTS,
        action_scales=(0.02,) * 6 + (0.001,),
        target_limits=TARGET_LIMITS,
        physics_dt=0.005,
        control_decimation=4,
        max_episode_steps=400,
        required_hold_steps=25,
        fixed_base_position=(0, 0, 0.01),
        fixed_base_quaternion_wxyz=(1, 0, 0, 0),
    )


def sample(
    *, tick: int = 0, episode: int = 1, applied: int = 0
) -> tuple[StateFrame, PrivilegedTaskState]:
    frame = StateFrame(
        FrameMetadata(
            sequence=tick + 1,
            episode_id=episode,
            physics_tick=tick,
            control_tick=tick,
            sim_time=tick * 0.005,
            applied_action_sequence=applied,
        ),
        {
            "position": np.array([0.1, 1.2, 1.3, -0.8, 0.1, 0.2, 0.02 / GRIPPER_STROKE]),
            "velocity": np.array([0, 0, 0, 0, 0, 0, 0.004 / GRIPPER_STROKE]),
            "monotonic_time": np.array([time.monotonic()]),
        },
    )
    task = PrivilegedTaskState(
        stamp=policy_stamp(frame),
        passive_position=-0.021,
        passive_velocity=-0.003,
        cube_position_world=(3.28, 4, 0.52),
        cube_quaternion_wxyz=(1, 0, 0, 0),
        cube_linear_velocity_world=(0.1, 0.2, 0.3),
        cube_angular_velocity_world=(0.4, 0.5, 0.6),
        environment_origin=(3, 4, 0.5),
        goal_position_world=(3.28, 4, 0.62),
        base_position_world=(3, 4, 0.51),
        base_quaternion_wxyz=(1, 0, 0, 0),
        elapsed_steps=tick // 4,
        hold_steps=0,
        lifted=False,
        terminal=False,
    )
    return frame, task


class RecordingClient:
    def __init__(self) -> None:
        self.descriptor = StreamDescriptor(
            "unit-test",
            "yam",
            "fixture",
            physics_dt=0.005,
            joint_names=PUBLIC_JOINTS,
            actuator_modes=("position",) * 7,
        )
        self.feedback_timeout = 2.0
        self.commands: list[tuple[NDArray[np.float64], StateFrame]] = []
        self.failure: Exception | None = None

    def write_positions(self, positions: list[float], *, observed: StateFrame) -> int:
        if self.failure is not None:
            raise self.failure
        self.commands.append((np.asarray(positions), observed))
        return len(self.commands)


@pytest.fixture
def artifact(contract: SparsePolicyContract, tmp_path: Path) -> YamArtifact:
    model = YamContract(
        schema="humanoid-control-lab.yam-simulated.v1",
        model_joint_names=MODEL_JOINTS,
        public_joint_names=PUBLIC_JOINTS,
        passive_joint_names=(PASSIVE_JOINT,),
        root_body="mocap_base",
        grasp_site="grasp_site",
        initial_positions={},
        initial_root_position=contract.fixed_base_position,
        target_limits=dict(zip(MODEL_JOINTS, TARGET_LIMITS, strict=True)),
        gripper_public_to_model_scale=GRIPPER_STROKE,
        physics_dt=contract.physics_dt,
        control_decimation=contract.control_decimation,
        actuators=(),
        files={},
        mjlab_revision="fixture",
        dimos_revision="fixture",
        mujoco_version="fixture",
        hardware_calibration="unverified",
    )
    return YamArtifact(tmp_path, model, contract.robot_contract_sha256)


@pytest.fixture
def adapter(
    contract: SparsePolicyContract, artifact: YamArtifact
) -> tuple[SparseSimPolicyAdapter, RecordingClient]:
    client = RecordingClient()
    return SparseSimPolicyAdapter(cast("SimRobotClient", client), artifact, contract), client


def preload() -> NDArray[np.float64]:
    return np.array([0.1, 1.17, 1.32, -0.79, 0.1, 0.2, 0.019])


def test_exact_state_layout_passive_measurement_and_world_axes(
    contract: SparsePolicyContract,
) -> None:
    _, task = sample(tick=4)
    task = replace(task, hold_steps=5, lifted=True)
    state = sparse_state(
        contract,
        [0.1, 1.2, 1.3, -0.8, 0.1, 0.2, 0.02],
        [0, 0, 0, 0, 0, 0, 0.004],
        task,
        preload(),
        np.zeros(7),
    )
    expected = [
        0.1,
        1.2,
        1.3,
        -0.8,
        0.1,
        0.2,
        0.02,
        -0.021,
        0,
        0,
        0,
        0,
        0,
        0,
        0.004,
        -0.003,
        0.28,
        0,
        0.02,
        1,
        0,
        0,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.1,
        1.17,
        1.32,
        -0.79,
        0.1,
        0.2,
        0.019,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.28,
        0,
        0.12,
        1 / 400,
        5 / 25,
        1,
    ]
    np.testing.assert_allclose(state, np.asarray(expected, dtype=np.float32), atol=1e-7, rtol=0)
    assert state.dtype == np.float32
    assert not state.flags.writeable


def test_contract_roundtrip_rejects_privilege_or_frame_change(
    contract: SparsePolicyContract,
) -> None:
    exported = json.loads(json.dumps(contract.to_dict()))
    assert SparsePolicyContract.from_dict(exported) == contract
    exported["velocity_frame"] = "cube_local"
    with pytest.raises(ValueError, match="frame"):
        SparsePolicyContract.from_dict(exported)


def test_counter_arithmetic_matches_recorded_cuda_reciprocal(
    contract: SparsePolicyContract,
) -> None:
    _, task = sample()
    task = replace(task, elapsed_steps=171, hold_steps=13)
    state = sparse_state(contract, preload(), np.zeros(7), task, preload(), np.zeros(7))
    assert state[46] == np.float32(171) * np.float32(1 / 400)
    assert state[47] == np.float32(13) * np.float32(1 / 25)
    # The saved CUDA clock differs by one ULP from division in Python float64.
    assert state[46] != np.float32(171 / 400)


def test_v1_robot_instance_requires_identity_base_orientation(
    contract: SparsePolicyContract, artifact: YamArtifact
) -> None:
    client = cast("SimRobotClient", RecordingClient())
    rotated = replace(contract, fixed_base_quaternion_wxyz=(0, 0, 0, 1))
    with pytest.raises(ValueError, match="base orientation"):
        SparseSimPolicyAdapter(client, artifact, rotated)
    equivalent = replace(contract, fixed_base_quaternion_wxyz=(-1, 0, 0, 0))
    SparseSimPolicyAdapter(client, artifact, equivalent)


@pytest.mark.parametrize("mutation", ["base", "quaternion", "clock", "passive", "dwell"])
def test_malformed_privileged_state_rejected(contract: SparsePolicyContract, mutation: str) -> None:
    _, task = sample()
    changes: dict[str, dict[str, Any]] = {
        "base": {"base_position_world": (3, 4, 0.5)},
        "quaternion": {"cube_quaternion_wxyz": (2, 0, 0, 0)},
        "clock": {"elapsed_steps": 401},
        "passive": {"passive_position": float("nan")},
        "dwell": {"hold_steps": 26},
    }
    task = replace(task, **changes[mutation])
    with pytest.raises(ValueError):
        sparse_state(contract, preload(), np.zeros(7), task, preload(), np.zeros(7))


def test_increment_is_once_per_control_step_and_zero_holds_preload(
    contract: SparsePolicyContract,
) -> None:
    held, previous = increment_targets(contract, preload(), np.zeros(7))
    np.testing.assert_array_equal(held, preload().astype(np.float32))
    np.testing.assert_array_equal(previous, np.zeros(7))
    result, bounded = increment_targets(contract, held, [1, -1, 0, 0, 0, 0, -1])
    expected = held + np.array([0.02, -0.02, 0, 0, 0, 0, -0.001], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(bounded, [1, -1, 0, 0, 0, 0, -1])


@pytest.mark.parametrize("action", [[0] * 6, [0] * 6 + [1.001], [float("nan")] * 7])
def test_malformed_actions_rejected(contract: SparsePolicyContract, action: list[float]) -> None:
    with pytest.raises(ValueError):
        increment_targets(contract, preload(), action)


def test_actual_observation_forwarded_and_action_cannot_be_reused(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient],
) -> None:
    boundary, client = adapter
    frame, task = sample()
    boundary.reset(observed=frame, task=task, servo_targets=preload())
    state = boundary.observation(observed=frame, task=task)
    np.testing.assert_array_equal(state[29:36], preload().astype(np.float32))
    boundary.write_action(np.zeros(7), observed=frame, task=task)
    assert client.commands[0][1] is frame
    with pytest.raises(ValueError, match="unconsumed"):
        boundary.write_action(np.zeros(7), observed=frame, task=task)
    next_frame, next_task = sample(tick=4, applied=1)
    next_state = boundary.observation(observed=next_frame, task=next_task)
    np.testing.assert_array_equal(next_state[29:36], state[29:36])
    assert next_state[46] == np.float32(1 / 400)


def test_float32_limit_endpoints_have_valid_public_coordinates(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient],
) -> None:
    boundary, client = adapter
    frame, task = sample()
    limits = np.asarray(TARGET_LIMITS)
    boundary.reset(observed=frame, task=task, servo_targets=limits[:, 1])
    boundary.observation(observed=frame, task=task)
    boundary.write_action(np.ones(7), observed=frame, task=task)
    expected = limits[:, 1].copy()
    expected[-1] /= GRIPPER_STROKE
    np.testing.assert_allclose(client.commands[0][0], expected, atol=2e-7, rtol=0)
    assert client.commands[0][0][-1] <= 1


@pytest.mark.parametrize("tick,applied,episode", [(4, 0, 1), (8, 1, 1), (4, 1, 2)])
def test_skipped_unacknowledged_and_reset_frames_rejected(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient], tick: int, applied: int, episode: int
) -> None:
    boundary, _ = adapter
    frame, task = sample()
    boundary.reset(observed=frame, task=task, servo_targets=preload())
    boundary.observation(observed=frame, task=task)
    boundary.write_action(np.zeros(7), observed=frame, task=task)
    later, other = sample(tick=tick, applied=applied, episode=episode)
    with pytest.raises(ValueError):
        boundary.observation(observed=later, task=other)


def test_stale_and_mixed_tick_object_state_rejected(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient],
) -> None:
    boundary, _ = adapter
    frame, task = sample()
    _, other_task = sample(tick=4)
    with pytest.raises(ValueError, match="one sample stamp"):
        boundary.reset(observed=frame, task=other_task, servo_targets=preload())
    frame.values["monotonic_time"][0] -= 10
    with pytest.raises(TimeoutError):
        boundary.reset(observed=frame, task=task, servo_targets=preload())


@pytest.mark.parametrize("failure", [TimeoutError, OSError, ConnectionError])
def test_failed_transport_requires_explicit_new_generation(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient],
    failure: type[Exception],
) -> None:
    boundary, client = adapter
    frame, task = sample()
    boundary.reset(observed=frame, task=task, servo_targets=preload())
    boundary.observation(observed=frame, task=task)
    client.failure = failure("synthetic transport failure")
    with pytest.raises(failure):
        boundary.write_action(np.ones(7), observed=frame, task=task)
    with pytest.raises(RuntimeError, match="uncertain"):
        boundary.observation(observed=frame, task=task)
    with pytest.raises(RuntimeError, match="uncertain"):
        boundary.write_action(np.ones(7), observed=frame, task=task)
    assert client.commands == []
    frame2, task2 = sample(episode=2)
    boundary.reset(observed=frame2, task=task2, servo_targets=preload())
    np.testing.assert_array_equal(
        boundary.observation(observed=frame2, task=task2)[29:36], preload().astype(np.float32)
    )


def test_out_of_order_sequence_and_terminal_action_rejected(
    adapter: tuple[SparseSimPolicyAdapter, RecordingClient],
) -> None:
    boundary, _ = adapter
    frame, task = sample()
    boundary.reset(observed=frame, task=task, servo_targets=preload())
    boundary.observation(observed=frame, task=task)
    boundary.write_action(np.zeros(7), observed=frame, task=task)
    next_frame, next_task = sample(tick=4, applied=1)
    stale = replace(
        next_frame, metadata=replace(next_frame.metadata, sequence=frame.metadata.sequence)
    )
    with pytest.raises(ValueError, match="sequence did not advance"):
        boundary.observation(observed=stale, task=replace(next_task, stamp=policy_stamp(stale)))
    terminal = replace(next_task, terminal=True)
    boundary.observation(observed=next_frame, task=terminal)
    with pytest.raises(ValueError, match="terminal"):
        boundary.write_action(np.zeros(7), observed=next_frame, task=terminal)
