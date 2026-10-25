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

"""Process boundary to the repository's pinned LeRobot environment."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Any

from dimos.core.transport import ZenohTransport
from dimos.experimental.isolated_python.module import isolated_python_run_command
from dimos.imitation.policy.lerobot.module import LeRobotPolicyModule, RolloutStatus
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState

if TYPE_CHECKING:
    from dimos.imitation.loop.device import Observation
    from dimos.robot.manipulators.openyam.control import YamControl
from dimos.imitation.loop.checkpoint import read_checkpoint

PROJECT = Path(__file__).resolve().parents[3] / "native" / "python" / "lerobot"


def command(*args: str) -> list[str]:
    python = PROJECT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            f"Install the isolated trainer first: uv sync --locked --project {PROJECT}"
        )
    return isolated_python_run_command(
        PROJECT, "--project", str(PROJECT), "python", "-m", "dimos_lerobot.act_loop", *args
    )


def train(
    snapshot: Path,
    output: Path,
    *,
    steps: int = 2000,
    device: str = "cpu",
    smoke: bool = False,
    initial: Path | None = None,
) -> None:
    args = [
        "train",
        str(snapshot.resolve()),
        str(output.resolve()),
        "--steps",
        str(steps),
        "--device",
        device,
    ]
    if smoke:
        args.append("--smoke")
    if initial is not None:
        args += ["--initial", str(initial.resolve())]
    subprocess.run(command(*args), check=True)


class PolicyRunner:
    """Desktop lifecycle for the shared isolated LeRobot policy module."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.checkpoint = checkpoint.resolve()
        self.manifest = read_checkpoint(checkpoint)
        self.identity: str = self.manifest["model_sha256"]
        self._stack = ExitStack()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-preflight")
        self._preflight: Future[RolloutStatus] | None = None
        self.status: RolloutStatus | None = None
        self._next_status = 0.0
        self._inputs: dict[str, ZenohTransport[Any]] = {}
        height, width, _ = self.manifest["image_shape"]
        self.module = LeRobotPolicyModule(
            policy_path=str(self.checkpoint),
            device=device,
            task="OpenYAM task",
            joint_names=self.manifest["joint_names"],
            fps=self.manifest["fps"],
            image_height=height,
            image_width=width,
            robot_type=self.manifest["robot_type"],
            startup_timeout=120.0,
            extra_env={"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        )

    def start(self, control: YamControl, task: str) -> None:
        self.module.config.task = task
        proxy = control.policy_control()
        self._stack.callback(proxy.stop_rpc_client)
        self.module.set_module_ref("_control", proxy)
        for name, kind in (("color_image", Image), ("coordinator_joint_state", JointState)):
            transport: ZenohTransport[Any] = ZenohTransport(
                f"{control.coordinator.config.rpc_name}/policy/{name}", kind
            )
            self._stack.callback(transport.stop)
            self.module.set_transport(name, transport)
            self._inputs[name] = transport
        self._stack.callback(self.module.stop)
        try:
            self.module.build()
            self.module.start()
        except BaseException:
            self.close()
            raise

    def observe(self, observation: Observation) -> None:
        self._inputs["coordinator_joint_state"].broadcast(
            None,
            JointState(
                name=self.manifest["joint_names"],
                position=observation.state.tolist(),
                ts=observation.state_ts,
            ),
        )
        self._inputs["color_image"].broadcast(
            None,
            Image.from_numpy(observation.image, format=ImageFormat.RGB, ts=observation.image_ts),
        )
        if self._preflight is None:
            self._preflight = self._executor.submit(self.module.preflight_rollout)
        elif self._preflight.done() and time.monotonic() >= self._next_status:
            if self.status is None:
                self.status = self._preflight.result()
            else:
                self.status = self.module.rollout_status()
            self._next_status = time.monotonic() + 0.2

    def start_rollout(self) -> None:
        if self.status is None:
            raise RuntimeError("Checkpoint preflight is still running")
        if not self.status["policy_ready"]:
            raise RuntimeError(self.status["last_error"] or "Checkpoint preflight failed")
        self.status = self.module.start_rollout()
        if not self.status["active"]:
            raise RuntimeError(self.status["last_error"] or "Policy did not start")

    def stop_rollout(self) -> None:
        self.status = self.module.stop_rollout()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._stack.close()
