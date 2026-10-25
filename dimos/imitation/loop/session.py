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

"""One command owner, explicit takeover, and provenance for corrective imitation."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
import time
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from dimos.imitation.collection.episode import EpisodeEvent, EpisodeStatus
from dimos.imitation.loop.act import PolicyRunner
from dimos.imitation.loop.device import Observation
from dimos.imitation.loop.recording import CONTRACT_KEYS, Episode, encode_image
from dimos.robot.manipulators.openyam.control import YamControl
from dimos.robot.manipulators.openyam.device import AuthorityMode


class LearningDevice(Protocol):
    def command(self, action: NDArray[np.float64], timestamp: float) -> None: ...


class Authority:
    """All transitions invalidate queued AND in-flight chunks, even on resume."""

    def __init__(self, initial: NDArray[np.float64]) -> None:
        self.generation = 0
        self.mode = "human"
        self.target = initial.copy()

    def switch(self, mode: str, measured: NDArray[np.float64]) -> None:
        if mode not in ("human", "correction", "policy", "hold"):
            raise ValueError(f"unknown authority: {mode}")
        self.generation += 1
        self.mode = mode
        self.target = measured.copy()


class LearningSession:
    def __init__(
        self,
        root: Path,
        joint_names: list[str],
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        initial: tuple[float, ...],
        *,
        fps: int = 20,
        task: str = "move the cube onto the green target",
        image_shape: tuple[int, int, int] = (240, 320, 3),
        observation_profile: str = "unspecified",
        artifact_contract_sha256: str = "",
        control: YamControl | None = None,
        source: str = "sim2",
    ) -> None:
        self.root, self.fps = root, fps
        self.contract: dict[str, Any] = dict(
            joint_names=joint_names,
            fps=fps,
            image_shape=list(image_shape),
            robot_type="openyam",
            action_semantics="absolute_public_joint_position",
            artifact_contract_sha256=artifact_contract_sha256,
            observation_profile=observation_profile,
        )
        self.control = control
        if control is not None:
            control.on_authority = self.switch
            control.on_record = self._record_button
        self.task, self.source = task, source
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.authority = Authority(np.asarray(initial, dtype=np.float64))
        self.lock = RLock()
        self.latest: Observation | None = None
        self.episode: Episode | None = None
        self.policy: PolicyRunner | None = None
        self._last_recorded_image = -1.0
        self.error: str | None = None
        self.chunks = 0
        self.run_record: str | None = None
        self._saved = self._discarded = 0
        self._report_recording("init")

    def _record_button(self, action: Literal["toggle", "discard"]) -> None:
        with self.lock:
            try:
                if self.episode is not None:
                    self.finish(keep=action == "toggle")
                elif action == "toggle":
                    self.record()
            except Exception as exc:
                self.error = str(exc)

    def _report_recording(self, event: EpisodeEvent) -> None:
        if self.control is not None:
            self.control.report_recording(
                EpisodeStatus(
                    ts=time.time(),
                    state="recording" if self.episode is not None else "idle",
                    episodes_saved=self._saved,
                    episodes_discarded=self._discarded,
                    last_event=event,
                    task_label=self.task,
                )
            )

    def load(self, worker: PolicyRunner) -> None:
        with self.lock:
            if self.policy is not None:
                raise RuntimeError("restart the session to load a different checkpoint")
            if any(worker.manifest.get(key) != self.contract[key] for key in CONTRACT_KEYS):
                raise ValueError(
                    "checkpoint joint order, image shape, FPS or action semantics do not match"
                )
            self.policy = worker

    def switch(self, mode: AuthorityMode) -> None:
        with self.lock:
            if self.latest is None and mode != "hold":
                raise RuntimeError("wait for fresh observations")
            if mode == "policy" and self.policy is None:
                raise RuntimeError("load a checkpoint before starting rollout")
            if mode == "human" and self.policy is not None:
                mode = "correction"
            measured = self.latest.state if self.latest is not None else self.authority.target
            previous = self.authority.mode
            self.authority.switch(mode, measured)
            if self.control is not None:
                self.authority.generation = self.control.transition(mode)
            if self.policy is not None:
                if previous == "policy":
                    self.policy.stop_rollout()
                if mode == "policy":
                    try:
                        self.policy.start_rollout()
                    except BaseException:
                        self.authority.switch("hold", measured)
                        if self.control is not None:
                            self.authority.generation = self.control.transition("hold")
                        raise
            self.error = None

    def set_target(self, index: int, value: float) -> None:
        with self.lock:
            if not np.isfinite(value) or not self.lower[index] <= value <= self.upper[index]:
                raise ValueError("target outside joint limits")
            if self.authority.mode in ("policy", "hold"):
                self.switch("human")
            self.authority.target[index] = value

    def record(self, *, split: str = "train", demonstrator: str = "operator") -> Path:
        with self.lock:
            if split not in ("train", "eval") or self.episode is not None:
                raise ValueError("choose train/eval and finish the current episode first")
            self.episode = Episode(
                self.root,
                {
                    **self.contract,
                    "task": self.task,
                    "split": split,
                    "policy_sha256": self.policy.identity if self.policy else None,
                    "source": self.source,
                    "command_acknowledgment": "driver-accepted"
                    if self.source == "hardware"
                    else "sim2-applied-sequence",
                    "demonstrator": demonstrator,
                    "sim2_run_record": self.run_record,
                },
            )
            self._last_recorded_image = -1.0
            self._report_recording("start")
            return self.episode.directory

    def finish(self, *, keep: bool, outcome: str = "ungraded") -> Path:
        with self.lock:
            if self.episode is None:
                raise RuntimeError("no episode is recording")
            episode = self.episode
            self.episode = None
            try:
                path = episode.finish(keep=keep, outcome=outcome)
            except Exception:
                self._discarded += 1
                self._report_recording("discard")
                raise
            self._saved += int(keep)
            self._discarded += int(not keep)
            self._report_recording("save" if keep else "discard")
            return path

    def tick(self, device: LearningDevice, observation: Observation) -> None:
        with self.lock:
            if self.latest is None and self.authority.mode == "hold":
                self.authority.target = observation.state.copy()
            if self.latest is not None and observation.generation != self.latest.generation:
                self.latest = observation
                self.switch("hold")
            self.latest = observation
            self._record_observation(observation)
            authority = self.authority
            if self.policy is not None:
                self.policy.observe(observation)
                status = self.policy.status
                if status is not None:
                    self.chunks = status["chunks_accepted"]
                    if authority.mode == "policy" and not status["active"]:
                        self.switch("hold")
                        self.error = status["last_error"] or "Policy rollout stopped"
            if authority.mode in ("hold", "policy"):
                return
            if self.control is not None and self.control.teleop_engaged:
                authority.target = observation.state.copy()
                return
            desired = authority.target
            # The coordinator's persistent target task owns velocity limits.
            action = np.clip(desired, self.lower, self.upper)
            device.command(action, time.time())

    def _record_observation(self, observation: Observation) -> None:
        if self.episode is None or observation.image_ts == self._last_recorded_image:
            return
        self._last_recorded_image = observation.image_ts
        accepted = observation.accepted
        self.episode.append(
            dict(
                state=observation.state.tolist(),
                state_ts=observation.state_ts,
                image_png=encode_image(observation.image),
                image_ts=observation.image_ts,
                accepted_action=list(accepted.positions) if accepted else [],
                command_ts=accepted.timestamp if accepted else observation.state_ts,
                # Sensor timestamps use the simulation clock; dispatch uses the host clock.
                # Sample acknowledged targets on the sensor clock and retain both clocks.
                applied_ts=observation.state_ts,
                command_age=(
                    (
                        observation.received_at
                        if observation.received_at is not None
                        else observation.state_ts
                    )
                    - accepted.timestamp
                    if accepted
                    else 0.0
                ),
                command_sequence=accepted.sequence if accepted else -1,
                applied=accepted is not None,
                authority=accepted.authority if accepted else "hold",
                segment=accepted.segment if accepted else self.authority.generation,
                sim_generation=observation.generation,
                policy_sha256=self.policy.identity if self.policy else None,
            )
        )

    def close(self) -> None:
        if self.control is not None:
            self.control.on_authority = None
            self.control.on_record = None
            self.control.transition("hold")
        with self.lock:
            if self.episode is not None:
                self.finish(keep=False, outcome="interrupted")
        if self.policy is not None:
            self.policy.close()
