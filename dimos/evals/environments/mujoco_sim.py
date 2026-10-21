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

"""MuJoCo launch, readiness and settling for live manipulation evals."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Thread
import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field

from dimos.evals.environments.runtime.recorded_poses import last_body_transform
from dimos.evals.environments.sim import Sim, SimConfig
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.porcelain.dimos import Dimos
from dimos.sim2.world.scene_types import SceneControlSpec

if TYPE_CHECKING:
    from dimos.e2e_tests.dimos_cli_call import DimosCliCall
    from dimos.memory.store.base import Store
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


class MujocoEnvironmentConfig(SimConfig):
    tracked_bodies: tuple[str, ...] = ()
    at_rest_rad_s: float = 0.02
    module_env: dict[str, str] = Field(default_factory=dict)
    ready_streams: tuple[str, ...] = ("color_image", "coordinator_joint_state")
    recorded_topics: tuple[str, ...] = (
        "color_image",
        "camera_info",
        "coordinator_joint_state",
        "tf",
        "odom",
    )
    # Robot-free MJCF with a scene sidecar defining entities and the robot mount.
    scene: Path | None = None


class MujocoEnvironment(Sim):
    """Run sim2 with sensor recordings for agents and separate physical truth for grading."""

    config: MujocoEnvironmentConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scene: SceneControlSpec | None = None
        self._truth_error: Exception | None = None
        self._truth_stop = Event()
        self._truth_thread: Thread | None = None
        self._truth: SqliteStore | None = None

    def configure_launch(self, proc: DimosCliCall) -> None:
        proc.simulator = "mujoco"
        proc.global_args = ["--record-topics", ",".join(self.config.recorded_topics)]
        if self.config.scene is not None:
            proc.global_args.extend(["--scene-package", str(self.config.scene.resolve())])
        proc.extra_env.update(self.config.module_env)
        proc.extra_env.setdefault(
            "SIMULATIONMODULE__VIEWER", os.environ.get("SIMULATIONMODULE__VIEWER", "false")
        )

    def setup_scene(self) -> None:
        app = Dimos.connect()
        self._resources.callback(app.stop)
        self.scene = app.find_module_by_spec(SceneControlSpec)
        missing = set(self.config.tracked_bodies) - self.scene.describe_scene().entities.keys()
        if missing:
            raise ValueError(f"scene lacks tracked entities: {sorted(missing)}")

    def prepare_recording(self, recording: Store, path: Path, deadline: float) -> dict[str, Path]:
        truth_path = path.with_name("truth.db")
        self._truth = SqliteStore(path=str(truth_path))
        self._resources.callback(self._truth.stop)
        self._truth_stop.clear()
        self._truth_error = None
        self._record_truth()
        self._truth_thread = Thread(target=self._record_truth_loop, daemon=True)
        self._truth_thread.start()
        self._resources.callback(self._stop_truth)
        self.wait_ready(recording, deadline=deadline)
        return {"truth": truth_path}

    def _record_truth(self) -> None:
        assert self.scene is not None and self._truth is not None
        state = self.scene.scene_state()
        self._truth.stream("tf", TFMessage).append(
            TFMessage(
                *[
                    Transform(
                        translation=state.entities[name].pose.position,
                        rotation=state.entities[name].pose.orientation,
                        frame_id="world",
                        child_frame_id=name,
                        ts=state.ts,
                    )
                    for name in self.config.tracked_bodies
                ]
            )
        )

    def _record_truth_loop(self) -> None:
        try:
            while not self._truth_stop.wait(0.1):
                self._record_truth()
        except Exception as error:
            self._truth_error = error

    def _stop_truth(self) -> None:
        self._truth_stop.set()
        if self._truth_thread is not None:
            self._truth_thread.join(timeout=10)
            if self._truth_thread.is_alive():
                raise RuntimeError("sim2 truth recorder did not stop")
        if self._truth_error is not None:
            raise RuntimeError("sim2 truth recording failed") from self._truth_error
        self._record_truth()

    def wait_ready(self, recording: Store, *, deadline: float) -> None:
        """Wait for fresh samples on every ready stream and a pose for every tracked body."""
        while time.monotonic() < deadline:
            if self._truth_error is not None:
                raise RuntimeError("sim2 truth recording failed") from self._truth_error
            try:
                ages = [
                    time.time() - getattr(recording.streams, name).last().data.ts
                    for name in self.config.ready_streams
                    if name in recording.streams
                ]
                for body in self.config.tracked_bodies:
                    if self._truth is None:
                        raise LookupError("truth recorder has not started")
                    last_body_transform(self._truth, body)
            except (LookupError, AttributeError):
                ages = []
            if len(ages) == len(self.config.ready_streams) and all(age < 10.0 for age in ages):
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"MuJoCo did not publish fresh {', '.join(self.config.ready_streams)}"
            + (
                f" and poses for {', '.join(self.config.tracked_bodies)}"
                if self.config.tracked_bodies
                else ""
            )
        )

    def latest_pose(self, recording: Store) -> PoseStamped:
        """Base pose from ``odom`` for floating-base robots; a fixed-base arm has none."""
        if "odom" not in recording.streams:
            raise LookupError("No MuJoCo odometry recorded")
        return cast("PoseStamped", recording.streams.odom.last().data)

    def settle(self, budget_s: float) -> None:
        """At rest once every joint is slower than ``at_rest_rad_s`` for ``at_rest_s``."""
        if self._truth_error is not None:
            raise RuntimeError("sim2 truth recording failed") from self._truth_error
        recording = self._recording
        if recording is None or "coordinator_joint_state" not in recording.streams:
            super().settle(budget_s)
            return
        rest_since: float | None = None
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            try:
                state = recording.streams.coordinator_joint_state.last().data
            except LookupError:
                return
            now = time.monotonic()
            if any(abs(v) > self.config.at_rest_rad_s for v in state.velocity):
                rest_since = None
            elif rest_since is None:
                rest_since = now
            elif now - rest_since >= self.config.at_rest_s:
                return
            time.sleep(self.config.settle_poll_s)
