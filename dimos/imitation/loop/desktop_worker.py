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

"""Desktop-owned experiment worker. JSON RPC on stdio; renderer on the main thread.

Opening Desktop only starts this idle process. Simulation, commands and training
require explicit requests. Operator scene geometry is never passed to ACT.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import json
from pathlib import Path
from queue import Empty, Queue
import re
import signal
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from typing import Any, TextIO
from uuid import uuid4

import cv2
import mujoco
import numpy as np
from numpy.typing import NDArray

from dimos.hardware.sensors.camera.webcam import Webcam
from dimos.imitation.loop.act import PolicyRunner, command
from dimos.imitation.loop.attached import AttachedBlueprint
from dimos.imitation.loop.checkpoint import export_policy, import_policy
from dimos.imitation.loop.device import OpenYamSimulation
from dimos.imitation.loop.hardware import OpenYamHardware, StationConfig, discover_station
from dimos.imitation.loop.recording import export_snapshot
from dimos.imitation.loop.session import LearningSession
from dimos.imitation.loop.transport import learning_bus
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.manipulators.openyam.device import AuthorityMode
from dimos.robot.manipulators.openyam.joints import OPENYAM_JOINTS
from dimos.robot.manipulators.openyam.simulation import YamArtifact, load_yam_artifact
from dimos.robot.manipulators.openyam.workcell import FPS, HEIGHT, OBSERVATION_PROFILE, WIDTH
from dimos.teleop.webxr.module import WebXRTeleopConfig


def jpeg(image: NDArray[np.uint8]) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if not ok:
        raise ValueError("could not encode camera")
    return base64.b64encode(encoded).decode("ascii")


def preview_camera(camera_id: str) -> dict[str, str]:
    """Open only the selected camera, return a frame and release it."""
    if not camera_id.startswith(("avfoundation:", "v4l2:")):
        raise ValueError("Choose a discovered USB camera")
    ready = Event()
    frames: list[Image] = []
    errors: list[str] = []

    def receive(image: Image) -> None:
        if not frames:
            frames.append(image)
            ready.set()

    def failed(error: Exception) -> None:
        errors.append(str(error))
        ready.set()

    camera = Webcam(camera_index=camera_id, width=640, height=480, fps=30)
    subscription = camera.image_stream().subscribe(receive, on_error=failed)
    try:
        if not ready.wait(5):
            raise RuntimeError("No camera frame. Check its USB connection and macOS camera access.")
        if errors:
            raise RuntimeError(errors[0])
        return {"camera_id": camera_id, "jpeg": jpeg(frames[0].to_rgb().data)}
    finally:
        subscription.dispose()


class DesktopExperiment:
    def __init__(
        self,
        root: Path,
        artifact_directory: Path | None = None,
        station: StationConfig | None = None,
    ) -> None:
        self.root = root
        self.artifact_directory = artifact_directory
        self.station = station
        self.artifact: YamArtifact | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.session: LearningSession | None = None
        self.robot: OpenYamSimulation | OpenYamHardware | None = None
        self.run_id: str | None = None
        self.model_id: str | None = None
        self.task = "Move the cube onto the green target"
        self.last_error: str | None = None
        self.last_heartbeat = time.monotonic()
        self.job: dict[str, Any] | None = None
        self.training: subprocess.Popen[bytes] | None = None
        self.log_stream: Any = None
        self.frames: dict[str, str] = {}
        self.poses: list[list[float]] = []
        self.geometry: dict[str, Any] | None = None
        self._next_frame = 0.0
        self._last_observation = time.monotonic()

    def _catalog(self) -> dict[str, Any]:
        episodes = []
        for path in sorted((self.root / "episodes").glob("*/metadata.json")):
            meta = json.loads(path.read_text())
            if meta.get("kept"):
                episodes.append(
                    {
                        "id": path.parent.name,
                        "compatible": meta.get("observation_profile") == self.observation_profile,
                        "observation_profile": meta.get("observation_profile"),
                        **{
                            k: meta.get(k)
                            for k in (
                                "task",
                                "split",
                                "outcome",
                                "frames",
                                "training_frames",
                                "policy_sha256",
                            )
                        },
                    }
                )
        models = []
        for path in sorted((self.root / "models").glob("*/loop.json")):
            try:
                meta = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                # The trainer may be publishing this manifest during a catalog poll.
                continue
            models.append(
                {
                    "id": path.parent.name,
                    "compatible": meta.get("observation_profile") == self.observation_profile,
                    "steps": meta["training_job"]["steps"],
                    "smoke": meta["smoke"],
                    "loss": meta["training_job"]["last_loss"],
                    "sha256": meta["model_sha256"],
                }
            )
        return {"episodes": episodes, "models": models}

    @property
    def observation_profile(self) -> str:
        return self.station.observation_profile if self.station else OBSERVATION_PROFILE

    def state(self) -> dict[str, Any]:
        session = self.session
        return {
            "running": self.robot is not None,
            "environment": "hardware" if self.station else "simulation",
            "controller_hand": self.station.controller_hand if self.station else "right",
            "station": self.robot.preflight() if isinstance(self.robot, OpenYamHardware) else None,
            "run_id": self.run_id,
            "task": self.task,
            "mode": session.authority.mode if session else "idle",
            "model_id": self.model_id,
            "recording": session.episode.directory.name if session and session.episode else None,
            "recorded_frames": session.episode.frames if session and session.episode else 0,
            "chunks": session.chunks if session else 0,
            "policy": session.policy.status if session and session.policy else None,
            "joints": session.latest.state.tolist() if session and session.latest else [],
            "targets": session.authority.target.tolist() if session else [],
            "joint_names": list(OPENYAM_JOINTS),
            "lower": session.lower.tolist() if session else [],
            "upper": session.upper.tolist() if session else [],
            "artifact_directory": str(self.artifact_directory) if self.artifact_directory else "",
            "artifact_sha256": self.artifact.contract_sha256 if self.artifact else None,
            "teleop_port": self.robot.control.teleop_port
            if self.robot and self.robot.control
            else None,
            "teleop_default_port": WebXRTeleopConfig.model_fields["server_port"].default,
            "error": self.last_error or (session.error if session else None),
            "job": self.job,
            "data_root": str(self.root),
            **self._catalog(),
        }

    def _require_session(self) -> LearningSession:
        if self.session is None or self.robot is None:
            raise RuntimeError("Connect the station or start the simulation first")
        return self.session

    def _require_finished_episode(self) -> None:
        if self.session is not None and self.session.episode is not None:
            raise RuntimeError("Save or discard the recording first")

    def stop(self) -> None:
        self.stack.close()
        self.stack = ExitStack()
        self.robot = None
        self.session = None
        self.geometry = None
        self.frames.clear()
        self.poses.clear()
        self.model_id = None
        self.run_id = None

    def start(self, model_id: str | None = None, device: str = "cpu") -> None:
        self._require_finished_episode()
        if self.training is not None:
            raise RuntimeError("Wait for training to finish or cancel it")
        checkpoint = self._checkpoint(model_id) if model_id else None
        if self.artifact_directory is None:
            raise ValueError("Choose the exported YAM artifact directory in Setup first")
        artifact = load_yam_artifact(self.artifact_directory)
        joints = artifact.robot_instance().config.joints
        home = tuple(artifact.robot_instance().initial_positions[name] for name in OPENYAM_JOINTS)
        self.stop()
        try:
            self.artifact = artifact
            if self.station:
                self.robot = self.stack.enter_context(OpenYamHardware(artifact, self.station))
            else:
                self.robot = self.stack.enter_context(
                    OpenYamSimulation(artifact, self.root / "sim2")
                )
            physical = isinstance(self.robot, OpenYamHardware)
            session = LearningSession(
                self.root,
                list(OPENYAM_JOINTS),
                tuple(j.lower for j in joints),
                tuple(j.upper for j in joints),
                home,
                task=self.task,
                fps=FPS,
                image_shape=(HEIGHT, WIDTH, 3),
                observation_profile=self.robot.observation_profile
                if isinstance(self.robot, OpenYamHardware)
                else OBSERVATION_PROFILE,
                artifact_contract_sha256=self.robot.contract_sha256
                if isinstance(self.robot, OpenYamHardware)
                else artifact.contract_sha256,
                source="hardware" if physical else "sim2",
                control=self.robot.control,
            )
            self.stack.callback(session.close)
            if checkpoint:
                worker = PolicyRunner(checkpoint, device=device)
                try:
                    assert self.robot.control is not None
                    worker.start(self.robot.control, self.task)
                    session.load(worker)
                except BaseException:
                    worker.close()
                    raise
            self.session = session
            self.model_id = model_id
            self.run_id = uuid4().hex
            session.run_record = (
                self.robot.module.describe()["run_record"]
                if isinstance(self.robot, OpenYamSimulation)
                else None
            )
            session.authority.switch("hold", np.asarray(home))
            self._last_observation = self.last_heartbeat = time.monotonic()
            self.last_error = None
            self.geometry = self._scene()
        except BaseException:
            self.stop()
            raise

    def _checkpoint(self, model_id: str, *, require_compatible: bool = True) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
            raise ValueError("Invalid checkpoint ID")
        path = self.root / "models" / model_id
        if not (path / "loop.json").is_file():
            raise ValueError("Checkpoint has not finished training")
        manifest = json.loads((path / "loop.json").read_text())
        if require_compatible and manifest.get("observation_profile") != self.observation_profile:
            raise ValueError("This checkpoint belongs to a different station or simulation setup")
        return path

    def _scene(self) -> dict[str, Any]:
        if not isinstance(self.robot, OpenYamSimulation):
            return {"meshes": {}, "geometries": [], "run_id": self.run_id}
        model = self.robot.cameras[0].reader.model
        meshes: dict[str, Any] = {}
        geometries = []
        for i in range(model.ngeom):
            # Visual robot geoms (group 2), plus authored scene objects (group 0).
            if int(model.geom_group[i]) not in (0, 2):
                continue
            kind = int(model.geom_type[i])
            mesh_id = int(model.geom_dataid[i])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH) and str(mesh_id) not in meshes:
                start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
                face, nface = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
                meshes[str(mesh_id)] = {
                    "vertices": model.mesh_vert[start : start + count].ravel().tolist(),
                    "faces": model.mesh_face[face : face + nface].ravel().tolist(),
                }
            material = int(model.geom_matid[i])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[i]
            geometries.append(
                {
                    "id": i,
                    "type": kind,
                    "mesh": str(mesh_id),
                    "size": model.geom_size[i].tolist(),
                    "rgba": rgba.tolist(),
                }
            )
        return {
            "meshes": meshes,
            "geometries": geometries,
            "run_id": self.run_id,
            "purpose": "operator visualization only",
        }

    def dispatch(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        self.last_heartbeat = time.monotonic()
        if action == "heartbeat":
            return {"ok": True}
        if action == "devices":
            return discover_station()
        if action == "preview_camera":
            if self.robot is not None:
                raise RuntimeError("Stop the session before previewing a different camera")
            return preview_camera(str(params["camera_id"]))
        if action == "start":
            task = str(params.get("task", self.task)).strip()
            if not task:
                raise ValueError("Enter the task you will demonstrate")
            self.task = task
            self.start()
        elif action == "stop":
            self._require_finished_episode()
            self.stop()
        elif action == "enable_arm":
            if not isinstance(self.robot, OpenYamHardware):
                raise RuntimeError("Connect a physical station first")
            self.robot.enable()
        elif action == "target":
            index = params.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(OPENYAM_JOINTS)
            ):
                raise ValueError("Unknown joint")
            self._require_session().set_target(index, float(params["value"]))
        elif action == "record":
            session = self._require_session()
            if session.authority.mode == "hold":
                session.switch("human")
            session.record(split=str(params.get("split", "train")))
        elif action in ("save", "discard"):
            outcome = str(params.get("outcome", "ungraded"))
            if outcome not in ("ungraded", "success", "failure"):
                raise ValueError("Unknown outcome")
            self._require_session().finish(keep=action == "save", outcome=outcome)
        elif action in ("takeover", "hold", "rollout"):
            modes: dict[str, AuthorityMode] = {
                "takeover": "human",
                "hold": "hold",
                "rollout": "policy",
            }
            session = self._require_session()
            if action == "rollout" and params.get("record") and session.episode is None:
                session.record(split=str(params.get("split", "train")))
            session.switch(modes[action])
        elif action == "teleop":
            self._require_session()
            assert self.robot is not None and self.robot.control is not None
            self.robot.control.start_teleop(int(params["port"]))
        elif action == "reset":
            self._require_finished_episode()
            session = self._require_session()
            assert self.robot is not None
            session.switch("hold")
            session.latest = None
            if not isinstance(self.robot, OpenYamSimulation):
                raise RuntimeError("Physical tasks must be reset by the operator")
            self.robot.module.reset()
        elif action == "load":
            self.start(str(params["model_id"]), str(params.get("device", "cpu")))
        elif action == "export_policy":
            model_id = str(params["model_id"])
            path = export_policy(
                self._checkpoint(model_id, require_compatible=False),
                self.root / "bundles" / f"{model_id}.zip",
            )
            return {"path": str(path), "name": path.name}
        elif action == "import_policy":
            import_policy(Path(str(params["path"])), self.root / "models")
        elif action == "train":
            self._train(params)
        elif action == "cancel_training":
            self._cancel_training()
        elif action == "scene":
            return self.geometry or {"meshes": {}, "geometries": [], "run_id": None}
        else:
            raise ValueError(f"Unknown action: {action}")
        self.last_error = None
        return self.state()

    def _train(self, params: dict[str, Any]) -> None:
        self._require_finished_episode()
        if self.training is not None:
            raise RuntimeError("Training is already running")
        steps = params.get("steps", 2000)
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 200000:
            raise ValueError("Training steps must be between 1 and 200000")
        device = str(params.get("device", "cpu"))
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("Choose cpu, mps or cuda")
        initial = self._checkpoint(str(params["initial"])) if params.get("initial") else None
        self.stop()
        job_id = f"act-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:5]}"
        snapshot = export_snapshot(
            self.root,
            self.root / "snapshots" / job_id,
            observation_profile=self.observation_profile,
        )
        output = self.root / "models" / job_id
        output.parent.mkdir(parents=True, exist_ok=True)
        log = self.root / "jobs" / f"{job_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        args = ["train", str(snapshot), str(output), "--steps", str(steps), "--device", device]
        if params.get("smoke"):
            args.append("--smoke")
        if initial:
            args += ["--initial", str(initial)]
        self.log_stream = log.open("wb")
        try:
            self.training = subprocess.Popen(
                command(*args), stdout=self.log_stream, stderr=subprocess.STDOUT
            )
        except BaseException:
            self.log_stream.close()
            self.log_stream = None
            raise
        self.job = {
            "id": job_id,
            "status": "running",
            "steps": steps,
            "step": 0,
            "device": device,
            "log": str(log),
            "output": str(output),
            "started_at": time.time(),
        }

    def _finish_job(self) -> None:
        assert self.job is not None
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        self.training = None
        self.job["finished_at"] = time.time()
        (self.root / "jobs" / f"{self.job['id']}.json").write_text(json.dumps(self.job, indent=2))

    def update(self) -> None:
        if self.training is not None:
            assert self.job is not None
            with Path(self.job["log"]).open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 8000))
                text = log.read().decode(errors="replace")
            self.job["tail"] = text
            matches = re.findall(r"ACT step (\d+)/(\d+): loss=([\d.]+)", text)
            if matches:
                self.job.update(step=int(matches[-1][0]), loss=float(matches[-1][2]))
            code = self.training.poll()
            if code is not None:
                self.job["status"] = "complete" if code == 0 else "failed"
                self._finish_job()
        if self.robot is None or self.session is None:
            return
        if isinstance(self.robot, OpenYamHardware):
            preview = self.robot.preview
            if preview is not None:
                self.frames = {"wrist": jpeg(preview.data)}
        observation = self.robot.observe()
        now = time.monotonic()
        if observation is None:
            if now - self._last_observation > 0.5 and self.session.latest is not None:
                self.session.switch("hold")
                self.last_error = "Camera or joint feedback is stale; control is held"
            return
        self._last_observation = now
        assert self.robot.control is not None
        self.robot.control.check_input()
        if now - self.last_heartbeat > 2 and not self.robot.control.teleop_engaged:
            if self.session.authority.mode != "hold":
                self.session.switch("hold")
        self.session.tick(self.robot, observation)
        if now >= self._next_frame:
            self.frames = {"wrist": jpeg(observation.image)}
            overview = self.robot.overview
            if overview is not None:
                self.frames["overview"] = jpeg(overview)
            if isinstance(self.robot, OpenYamSimulation):
                data = self.robot.cameras[0].reader.data
                self.poses = [
                    np.concatenate((data.geom_xpos[i], data.geom_xmat[i])).tolist()
                    for i in range(len(data.geom_xpos))
                ]
            self._next_frame = now + 0.1

    def close(self) -> None:
        self.stop()
        self._cancel_training()

    def _cancel_training(self) -> None:
        if self.training is not None:
            self.training.terminate()
            try:
                self.training.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training.kill()
                self.training.wait(timeout=5)
            assert self.job is not None
            self.job["status"] = "cancelled"
            self._finish_job()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--station", help="JSON station configuration; omitted for simulation")
    parser.add_argument("--follow-blueprint", action="store_true")
    args = parser.parse_args()
    if args.follow_blueprint:
        serve(args, AttachedBlueprint())
    else:
        with learning_bus():
            serve(args)


def serve(args: argparse.Namespace, attached: AttachedBlueprint | None = None) -> None:
    wire: TextIO = sys.stdout
    sys.stdout = sys.stderr  # Imported modules/loggers must not corrupt the JSON protocol.
    stopped = Event()
    output_lock = Lock()
    requests: Queue[dict[str, Any]] = Queue(maxsize=64)

    def emit(value: dict[str, Any]) -> None:
        with output_lock:
            wire.write(json.dumps(value, allow_nan=False) + "\n")
            wire.flush()

    def receive() -> None:
        try:
            for line in sys.stdin:
                requests.put(json.loads(line), timeout=2)
        finally:
            stopped.set()

    def stop_signal(*_: Any) -> None:
        stopped.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_signal)
    Thread(target=receive, daemon=True, name="desktop-requests").start()
    experiment = attached or DesktopExperiment(
        args.root.expanduser().resolve(),
        args.artifact,
        StationConfig.model_validate_json(args.station) if args.station else None,
    )
    next_status = 0.0
    try:
        if attached is not None:
            attached.start()
        emit({"event": "ready", "data": experiment.state()})
        while not stopped.is_set():
            started = time.monotonic()
            for _ in range(8):
                try:
                    request = requests.get_nowait()
                except Empty:
                    break
                try:
                    result = experiment.dispatch(request["action"], request.get("params", {}))
                    emit({"id": request["id"], "result": result})
                except Exception as exc:
                    # Operator errors belong to the request/job. A stale browser
                    # session must not latch a fault on the connected robot.
                    if not request["action"].startswith("operator_"):
                        experiment.last_error = str(exc)
                    emit({"id": request["id"], "error": str(exc)})
            try:
                experiment.update()
                if attached is not None:
                    attached.frames = {
                        key: jpeg(image.to_rgb().data) for key, image in attached.images().items()
                    }
            except Exception as exc:
                experiment.last_error = str(exc)
                if experiment.session and experiment.session.latest is not None:
                    experiment.session.switch("hold")
            if started >= next_status:
                emit(
                    {
                        "event": "state",
                        "data": experiment.state(),
                        "frames": experiment.frames,
                        "poses": experiment.poses,
                    }
                )
                next_status = started + 0.1
            stopped.wait(max(0, 0.05 - (time.monotonic() - started)))
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
