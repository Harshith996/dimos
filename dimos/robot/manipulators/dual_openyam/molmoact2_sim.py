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

"""ManiSkill Bimanual YAM simulation bridge for DimOS WebXR teleoperation."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol, cast

import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.imitation.collection.episode_monitor import EpisodeControlSpec
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.spec.utils import Spec
from dimos.teleop.webxr.controller_types import BUTTON_ALIASES, Buttons
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DIMOS_DOF = len(DUAL_OPENYAM_JOINTS)
_MANISKILL_QPOS_DOF = 16
_FINGER_OPEN_RANGE_M = 0.0475


def dimos_positions_to_maniskill_action(
    positions: list[float] | np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray:
    """Convert DimOS joint order into ManiSkill's 14-D controller action."""
    values = np.asarray(positions, dtype=np.float32)
    if values.shape != (_DIMOS_DOF,):
        raise ValueError(f"Expected {_DIMOS_DOF} DimOS joint positions, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("DimOS joint positions must all be finite")

    action = np.empty(_DIMOS_DOF, dtype=np.float32)
    action[:6] = values[:6]
    # The ManiSkill MJCF mirrors wrist roll relative to the DimOS URDF.
    action[5] = -values[5]
    action[6] = 1.0 - 2.0 * np.clip(values[12], 0.0, 1.0)
    action[7:13] = values[6:12]
    action[12] = -values[11]
    action[13] = 1.0 - 2.0 * np.clip(values[13], 0.0, 1.0)
    return action


def maniskill_qpos_to_dimos(
    qpos: list[float] | np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray:
    """Convert ManiSkill's interleaved 16-D qpos into DimOS joint order."""
    values = np.asarray(qpos, dtype=np.float32)
    if values.shape != (_MANISKILL_QPOS_DOF,):
        raise ValueError(
            f"Expected {_MANISKILL_QPOS_DOF} ManiSkill joint positions, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("ManiSkill joint positions must all be finite")

    result = np.empty(_DIMOS_DOF, dtype=np.float32)
    result[:6] = values[[0, 2, 4, 6, 8, 10]]
    result[6:12] = values[[1, 3, 5, 7, 9, 11]]
    # Convert mirrored simulator wrist roll back into the DimOS convention.
    result[5] = -result[5]
    result[11] = -result[11]
    result[12] = np.clip(-values[12] / _FINGER_OPEN_RANGE_M, 0.0, 1.0)
    result[13] = np.clip(-values[14] / _FINGER_OPEN_RANGE_M, 0.0, 1.0)
    return result


def maniskill_qvel_to_dimos(qvel: list[float] | np.ndarray[Any, np.dtype[Any]]) -> np.ndarray:
    """Convert ManiSkill qvel into DimOS order and normalized gripper velocity."""
    values = np.asarray(qvel, dtype=np.float32)
    if values.shape != (_MANISKILL_QPOS_DOF,):
        raise ValueError(
            f"Expected {_MANISKILL_QPOS_DOF} ManiSkill joint velocities, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("ManiSkill joint velocities must all be finite")

    result = np.empty(_DIMOS_DOF, dtype=np.float32)
    result[:6] = values[[0, 2, 4, 6, 8, 10]]
    result[6:12] = values[[1, 3, 5, 7, 9, 11]]
    # Velocity uses the same coordinate convention as position.
    result[5] = -result[5]
    result[11] = -result[11]
    result[12] = -values[12] / _FINGER_OPEN_RANGE_M
    result[13] = -values[14] / _FINGER_OPEN_RANGE_M
    return result


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _extract_rgb(obs: dict[str, Any], camera: str) -> np.ndarray:
    sensors = obs.get("sensor_data") or {}
    data = sensors.get(camera)
    raw = data.get("rgb") if isinstance(data, dict) else data
    if raw is None:
        raise KeyError(f"Camera {camera!r} unavailable; found {sorted(sensors)}")
    image = _as_numpy(raw)
    if image.dtype != np.uint8:
        scale = 255.0 if image.size and float(image.max()) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Camera {camera!r} returned unexpected shape {image.shape}")
    return np.ascontiguousarray(image)


class MolmoAct2YamSimConfig(ModuleConfig):
    """Runtime settings for the MolmoAct2 ManiSkill YAM environment."""

    molmoact2_root: str = "/workspace/molmoact2"
    env_id: str = "BimanualYAMEasyPickCubes-v1"
    task_module: str | None = "dimos.robot.manipulators.dual_openyam.maniskill_easy_task"
    seed: int = 42
    control_hz: int = Field(default=30, gt=0)
    sim_hz: int = Field(default=150, gt=0)
    camera_hz: float = Field(default=15.0, gt=0.0)
    shader_pack: str = "rt-fast"
    startup_timeout_s: float = Field(default=90.0, gt=0.0)
    reset_timeout_s: float = Field(default=15.0, gt=0.0)
    reset_on_termination: bool = False


class MolmoAct2YamSimSpec(Spec, Protocol):
    def reset_scene(self) -> str: ...


class MolmoAct2YamSimModule(Module):
    """Run ManiSkill YAM and bridge motor commands, feedback, and three RGB cameras."""

    dedicated_worker = True
    config: MolmoAct2YamSimConfig

    motor_command: In[MotorCommandArray]
    motor_states: Out[JointState]
    egocentric_image: Out[Image]
    left_wrist_image: Out[Image]
    right_wrist_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._command_lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._reset_requested = threading.Event()
        self._reset_complete = threading.Event()
        self._reset_rpc_lock = threading.Lock()
        self._reset_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.motor_command.subscribe(self._on_motor_command)))
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_simulation,
            name="MolmoAct2YamSim",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=self.config.startup_timeout_s):
            raise TimeoutError(
                f"ManiSkill environment did not start within {self.config.startup_timeout_s}s"
            )
        if self._startup_error is not None:
            raise RuntimeError(
                "Failed to start MolmoAct2 ManiSkill environment"
            ) from self._startup_error

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self._reset_complete.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        super().stop()

    def _on_motor_command(self, command: MotorCommandArray) -> None:
        try:
            action = dimos_positions_to_maniskill_action(command.q)
        except ValueError as exc:
            logger.warning("Ignoring invalid simulator motor command", error=str(exc))
            return
        with self._command_lock:
            self._latest_action = action

    @rpc
    def reset_scene(self) -> str:
        """Reset arms and objects on the simulator thread and hold the new home pose."""
        with self._reset_rpc_lock:
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError("ManiSkill simulation is not running")
            self._reset_error = None
            self._reset_complete.clear()
            self._reset_requested.set()
            if not self._reset_complete.wait(timeout=self.config.reset_timeout_s):
                raise TimeoutError(
                    f"ManiSkill scene reset did not finish within {self.config.reset_timeout_s}s"
                )
            if self._reset_error is not None:
                raise RuntimeError("ManiSkill scene reset failed") from self._reset_error
        return "ManiSkill scene reset to its initial state"

    def _run_simulation(self) -> None:
        env: Any = None
        try:
            env = self._create_environment()
            obs = self._reset_environment(env)
            self._ready_event.set()
            logger.info(
                "MolmoAct2 ManiSkill bridge ready",
                env_id=self.config.env_id,
                control_hz=self.config.control_hz,
                camera_hz=self.config.camera_hz,
            )

            period = 1.0 / self.config.control_hz
            camera_period = 1.0 / self.config.camera_hz
            next_step = time.monotonic()
            next_camera = next_step + camera_period
            while not self._stop_event.is_set():
                if self._reset_requested.is_set():
                    try:
                        obs = self._reset_environment(env)
                        self._reset_error = None
                    except BaseException as exc:
                        self._reset_error = exc
                        logger.exception("ManiSkill scene reset failed")
                    finally:
                        self._reset_requested.clear()
                        self._reset_complete.set()
                    next_camera = time.monotonic() + camera_period

                with self._command_lock:
                    assert self._latest_action is not None
                    action = self._latest_action.copy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, _, terminated, truncated, _ = env.step(action)
                self._publish_state(obs)

                now = time.monotonic()
                if now >= next_camera:
                    self._publish_cameras(obs)
                    next_camera += camera_period
                    if next_camera <= now:
                        next_camera = now + camera_period
                episode_ended = bool(_as_numpy(terminated).any()) or bool(
                    _as_numpy(truncated).any()
                )
                if episode_ended and self.config.reset_on_termination:
                    obs, _ = env.reset(seed=self.config.seed)
                    self._publish_state(obs)
                    self._publish_cameras(obs)

                next_step += period
                delay = next_step - time.monotonic()
                if delay > 0.0:
                    self._stop_event.wait(delay)
                else:
                    next_step = time.monotonic()
        except BaseException as exc:
            self._startup_error = exc
            self._reset_error = exc
            logger.exception("MolmoAct2 ManiSkill bridge failed")
            self._ready_event.set()
            self._reset_complete.set()
        finally:
            if env is not None:
                env.close()

    def _create_environment(self) -> Any:
        root = Path(self.config.molmoact2_root).expanduser().resolve()
        if not (root / "sim_eval").is_dir():
            raise FileNotFoundError(f"MolmoAct2 sim_eval not found under {root}")
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            gym = importlib.import_module("gymnasium")
            importlib.import_module("mani_skill.envs")
            importlib.import_module("sim_eval.tasks")
            if self.config.task_module is not None:
                importlib.import_module(self.config.task_module)
        except ImportError as exc:
            raise RuntimeError(
                "MolmoAct2 simulation requires gymnasium, mani-skill, and sapien in the "
                "DimOS environment"
            ) from exc

        return gym.make(
            self.config.env_id,
            obs_mode="rgb",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            max_episode_steps=1_000_000,
            reward_mode="none",
            sensor_configs={"shader_pack": self.config.shader_pack},
            sim_config={"sim_freq": self.config.sim_hz, "control_freq": self.config.control_hz},
        )

    def _reset_environment(self, env: Any) -> dict[str, Any]:
        obs, _ = env.reset(seed=self.config.seed)
        initial_positions = self._publish_state(obs)
        with self._command_lock:
            self._latest_action = dimos_positions_to_maniskill_action(initial_positions)
        self._publish_cameras(obs)
        return cast("dict[str, Any]", obs)

    def _publish_state(self, obs: dict[str, Any]) -> np.ndarray:
        agent = obs.get("agent") or {}
        qpos = _as_numpy(agent.get("qpos"))
        positions = maniskill_qpos_to_dimos(qpos)

        qvel_raw = agent.get("qvel")
        velocities = (
            np.zeros(_DIMOS_DOF, dtype=np.float32)
            if qvel_raw is None
            else maniskill_qvel_to_dimos(_as_numpy(qvel_raw))
        )
        self.motor_states.publish(
            JointState(
                frame_id="dual_openyam_sim",
                name=list(DUAL_OPENYAM_JOINTS),
                position=positions.tolist(),
                velocity=velocities.tolist(),
                effort=[0.0] * _DIMOS_DOF,
            )
        )
        return positions

    def _publish_cameras(self, obs: dict[str, Any]) -> None:
        timestamp = time.time()
        outputs = (
            ("top_cam", self.egocentric_image),
            ("left_cam", self.left_wrist_image),
            ("right_cam", self.right_wrist_image),
        )
        for camera, output in outputs:
            output.publish(
                Image(
                    data=_extract_rgb(obs, camera),
                    format=ImageFormat.RGB,
                    frame_id=camera,
                    ts=timestamp,
                )
            )


class MolmoAct2SceneResetConfig(ModuleConfig):
    """Quest button used for a coordinated episode discard and scene reset."""

    reset_button: str = "X"


class MolmoAct2SceneResetController(Module):
    """Discard an active take, then reset the ManiSkill scene from a Quest button."""

    config: MolmoAct2SceneResetConfig
    button_pressed: In[Buttons]

    _episode_control: EpisodeControlSpec
    _simulator: MolmoAct2YamSimSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reset_lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.button_pressed.subscribe(self._on_button_pressed)))

    def _on_button_pressed(self, buttons: Buttons) -> None:
        attribute = BUTTON_ALIASES.get(self.config.reset_button, self.config.reset_button)
        if getattr(buttons, attribute):
            threading.Thread(
                target=self._perform_reset, name="ManiSkillSceneReset", daemon=True
            ).start()

    def _perform_reset(self) -> None:
        with self._reset_lock:
            status = self._episode_control.get_status()
            if status.state == "recording":
                self._episode_control.command("discard")
            result = self._simulator.reset_scene()
            logger.info("Full ManiSkill scene reset", result=result)


molmo_act2_yam_sim = MolmoAct2YamSimModule.blueprint
molmo_act2_scene_reset = MolmoAct2SceneResetController.blueprint
