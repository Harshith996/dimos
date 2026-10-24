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

"""Simulation-only sparse49 policy binding; the caller owns inference and stepping.

This adapter never opens a client, starts a world or loads a policy. A caller must
supply a verified robot artifact, actual same-tick privileged simulator state and
an already connected SimRobotClient. It does not establish task contact parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dimos.robot.manipulators.openyam.simulation import (
    MODEL_JOINTS,
    PASSIVE_JOINT,
    TARGET_LIMITS,
    YamArtifact,
    YamSimAdapter,
    public_to_model,
)
from dimos.robot.manipulators.openyam.sparse_policy import (
    PolicyStamp,
    PrivilegedTaskState,
    SparsePolicyContract,
    increment_targets,
    sparse_state,
    vector,
)
from dimos.sim2.messages import StateFrame

if TYPE_CHECKING:
    from dimos.sim2.client import SimRobotClient


def policy_stamp(frame: StateFrame) -> PolicyStamp:
    metadata = frame.metadata
    return PolicyStamp(
        metadata.episode_id, metadata.sequence, metadata.physics_tick, metadata.sim_time
    )


@dataclass
class _Episode:
    reset_stamp: PolicyStamp
    targets: NDArray[np.float32]
    previous_action: NDArray[np.float32]
    expected_tick: int
    expected_elapsed: int
    expected_action_sequence: int
    inference_frame: StateFrame | None = None
    inference_task: PrivilegedTaskState | None = None
    faulted: bool = False
    last_consumed_stamp: PolicyStamp | None = None


class SparseSimPolicyAdapter:
    """One writer, fixed-period inference, exact float32 incremental memory.

    Reset preload is supplied explicitly from the simulator reset transaction;
    measured joint positions do not reconstruct gravity/contact holding targets.
    Runtime reset changes the generation. Each generation requires explicit reset.
    """

    def __init__(
        self, client: SimRobotClient, artifact: YamArtifact, contract: SparsePolicyContract
    ) -> None:
        if contract.robot_contract_sha256 != artifact.contract_sha256:
            raise ValueError("Policy robot artifact identity mismatch")
        if (
            contract.state_joint_names != (*MODEL_JOINTS, PASSIVE_JOINT)
            or contract.action_joint_names != MODEL_JOINTS
        ):
            raise ValueError("Policy native YAM joint order changed")
        if not np.array_equal(contract.target_limits, TARGET_LIMITS):
            raise ValueError("Policy and robot target envelopes differ")
        physical = artifact.contract
        if (
            not np.isclose(contract.physics_dt, physical.physics_dt, atol=1e-12, rtol=0)
            or contract.control_decimation != physical.control_decimation
        ):
            raise ValueError("Policy and robot control periods differ")
        if not np.isclose(client.descriptor.physics_dt, contract.physics_dt, atol=1e-12, rtol=0):
            raise ValueError("Client physics period differs from policy")
        if not np.array_equal(contract.fixed_base_position, physical.initial_root_position):
            raise ValueError("Policy fixed base differs from artifact")
        # The v1 YamArtifact.robot_instance() uses RobotInstance's identity rpy;
        # its physical contract does not expose a separate orientation field.
        base_quaternion = np.asarray(contract.fixed_base_quaternion_wxyz)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        if (
            min(
                np.linalg.norm(base_quaternion - identity),
                np.linalg.norm(base_quaternion + identity),
            )
            > 1e-12
        ):
            raise ValueError("Policy base orientation differs from the v1 robot instance")
        self._writer = YamSimAdapter(client)
        self._feedback_timeout = client.feedback_timeout
        self._contract = contract
        self._episode: _Episode | None = None
        self._lock = threading.RLock()

    def _validate_frame(self, frame: StateFrame, task: PrivilegedTaskState) -> PolicyStamp:
        if frame.lifecycle != "ready":
            raise ValueError("Simulator robot is not ready")
        stamp = policy_stamp(frame)
        if (
            type(frame.metadata.applied_action_sequence) is not int
            or frame.metadata.applied_action_sequence < 0
        ):
            raise ValueError("Applied command sequence must be a nonnegative integer")
        if stamp != task.stamp:
            raise ValueError("Robot and privileged object/task state must share one sample stamp")
        monotonic = vector(frame.values["monotonic_time"], 1, "sample monotonic time")[0]
        age = time.monotonic() - monotonic
        if not 0 <= age <= self._feedback_timeout:
            raise TimeoutError("Policy observation is stale")
        return stamp

    def _state(
        self, frame: StateFrame, task: PrivilegedTaskState, episode: _Episode
    ) -> NDArray[np.float32]:
        return sparse_state(
            self._contract,
            public_to_model(frame.values["position"], targets=False),
            public_to_model(frame.values["velocity"], targets=False),
            task,
            episode.targets,
            episode.previous_action,
        )

    def reset(
        self, *, observed: StateFrame, task: PrivilegedTaskState, servo_targets: ArrayLike
    ) -> None:
        """Bind freshly reset simulator state without sending any command."""
        with self._lock:
            stamp = self._validate_frame(observed, task)
            if (
                self._episode is not None
                and stamp.episode_id <= self._episode.reset_stamp.episode_id
            ):
                raise ValueError("A new simulator reset generation is required")
            if task.elapsed_steps != 0 or task.hold_steps != 0 or task.terminal:
                raise ValueError("Policy reset requires a fresh task clock and dwell")
            targets, previous = increment_targets(self._contract, servo_targets, np.zeros(7))
            episode = _Episode(
                stamp,
                targets,
                previous,
                stamp.physics_tick,
                0,
                observed.metadata.applied_action_sequence,
            )
            self._state(observed, task, episode)
            self._episode = episode

    def observation(
        self, *, observed: StateFrame, task: PrivilegedTaskState
    ) -> NDArray[np.float32]:
        """Return immutable unnormalized state49 for the external policy."""
        with self._lock:
            episode = self._require_episode()
            stamp = self._validate_frame(observed, task)
            if stamp.episode_id != episode.reset_stamp.episode_id:
                raise ValueError("Simulator generation changed; reset policy memory explicitly")
            if (
                stamp.physics_tick != episode.expected_tick
                or task.elapsed_steps != episode.expected_elapsed
            ):
                raise ValueError("Expected exactly one policy period since the preceding action")
            if (
                episode.last_consumed_stamp is not None
                and stamp.sequence <= episode.last_consumed_stamp.sequence
            ):
                raise ValueError("Simulator sample sequence did not advance")
            elapsed_physics = stamp.physics_tick - episode.reset_stamp.physics_tick
            expected_time = (
                episode.reset_stamp.sim_time + elapsed_physics * self._contract.physics_dt
            )
            if not np.isclose(stamp.sim_time, expected_time, atol=1e-9, rtol=0):
                raise ValueError("Simulator tick and time disagree")
            if observed.metadata.applied_action_sequence != episode.expected_action_sequence:
                raise ValueError("Previous policy command is not the acknowledged applied command")
            result = self._state(observed, task, episode)
            episode.inference_frame, episode.inference_task = observed, task
            return result

    def _require_episode(self) -> _Episode:
        if self._episode is None:
            raise RuntimeError("Explicit reset preload is required before inference")
        if self._episode.faulted:
            raise RuntimeError(
                "Command outcome is uncertain; reset the simulator before continuing"
            )
        return self._episode

    def write_action(
        self, action: ArrayLike, *, observed: StateFrame, task: PrivilegedTaskState
    ) -> int:
        """Consume this exact inference sample once, then retain its new targets."""
        with self._lock:
            episode = self._require_episode()
            if episode.inference_frame is not observed or episode.inference_task is not task:
                raise ValueError("Action must use the adapter's exact unconsumed inference sample")
            stamp = self._validate_frame(observed, task)
            if (
                stamp.physics_tick != episode.expected_tick
                or stamp.episode_id != episode.reset_stamp.episode_id
            ):
                raise ValueError("Action refers to a stale simulator tick or generation")
            if task.terminal or task.elapsed_steps >= self._contract.max_episode_steps:
                raise ValueError("A terminal task cannot consume another policy action")
            targets, bounded = increment_targets(self._contract, episode.targets, action)
            limits = np.asarray(self._contract.target_limits)
            transmitted = np.clip(targets.astype(np.float64), limits[:, 0], limits[:, 1])
            # The public API uses decimal float64 limits; clipping a float32 endpoint
            # to that representation must not alter the float32 training recurrence.
            if not np.array_equal(transmitted.astype(np.float32), targets):
                raise ValueError("Transport conversion would change the policy target recurrence")
            try:
                sequence = self._writer.write_model_targets(transmitted, observed=observed)
            except Exception:
                # publish_action advances its sequence before broadcasting. Any
                # transport failure may follow an applied command; never retry it.
                episode.faulted = True
                raise
            if type(sequence) is not int or sequence <= episode.expected_action_sequence:
                episode.faulted = True
                raise RuntimeError("Invalid command acknowledgement sequence")
            episode.targets, episode.previous_action = targets, bounded
            episode.expected_action_sequence = sequence
            episode.expected_tick += self._contract.control_decimation
            episode.expected_elapsed += 1
            episode.last_consumed_stamp = stamp
            episode.inference_frame = episode.inference_task = None
            return sequence
