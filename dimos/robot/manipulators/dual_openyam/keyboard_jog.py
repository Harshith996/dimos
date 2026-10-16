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

"""TEMP: a terminal remote for one dual OpenYAM arm, for calibration touches.

Acts as a virtual Quest controller for ``keyboard-teleop-dual-openyam``: it
streams a controller pose for each hand and holds the selected arm's grip, and
keys nudge that pose. Tilts pivot about the fingertip, so it stays put.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import pinocchio as pin

from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.robot.manipulators.dual_openyam.model import DUAL_OPENYAM_MODEL_PATH
from dimos.teleop.webxr.controller_types import Buttons

SIDES = ("left", "right")
# Closed fingertip end in the grasp frame (gripper link z -0.1): housing face
# at 63.5 mm plus 82.5 mm of finger, calipers 2026-10-14.
FINGERTIP_IN_GRASP = np.array([0.0, 0.0, -0.046])
# A direction key starts continuous motion at the selected speed; the same
# key or space stops it. A segment auto-stops after MAX_SEGMENT_MM / _DEG.
SPEEDS_MM_S = (2.0, 5.0, 10.0, 20.0, 40.0)
SPEEDS_DEG_S = (2.0, 5.0, 10.0, 20.0, 40.0)
MAX_SEGMENT_MM = 150.0
MAX_SEGMENT_DEG = 90.0
RATE_HZ = 50.0
# Terminal key auto-repeat: a repeat arriving within REPEAT_S means the key is
# still held; no repeat for RELEASE_S means it was released. A lone tap gives
# one TAP step.
REPEAT_S = 0.6
RELEASE_S = 0.15
TAP_MM = 1.0
TAP_DEG = 1.0
HOME_JOINTS = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
REST_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # folded on the supports (URDF zero)
HOME_SPEED_M_S = 0.05
HOME_SPEED_RAD_S = np.radians(30.0)

HELP = """\
1/2 left/right arm   w/s forward/back   a/d left/right   r/f up/down
i/k tip fingers forward/back   j/l tip fingers left/right   u/o twist
tap = 1 mm / 1 deg, hold = continuous at the speed, release = stop, space = stop
-/= linear slower/faster   [/] turn slower/faster   c/v close/open gripper   h home   P park   p status   q quit"""


class Kinematics:
    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(DUAL_OPENYAM_MODEL_PATH))
        self.data = self.model.createData()

    def grasp_frame(self, side: str, positions: dict[str, float]) -> pin.SE3:
        q = pin.neutral(self.model)
        for index in range(1, 7):
            joint = self.model.joints[self.model.getJointId(f"{side}_joint{index}")]
            q[joint.idx_q] = positions[f"{side}_joint{index}"]
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.model.getFrameId(f"{side}_grasp_frame")].copy()


def _rotation(axis: str, degrees: float) -> np.ndarray:
    vector = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    return pin.exp3(np.radians(degrees) * np.array(vector))


class Remote:
    def __init__(self) -> None:
        self.kinematics = Kinematics()
        self.lock = threading.Lock()
        self.side = "left"
        self.speed_index = 1  # 5 mm/s; = for faster
        self.turn_index = 1  # 5 deg/s; ] for faster, [ for slower
        # Active motion: (key, linear direction unit vector or None, rotation axis or None, sign)
        self.motion: tuple[str, np.ndarray | None, str | None, float] | None = None
        self.segment_mm = 0.0
        self.segment_deg = 0.0
        self.last_direction: tuple[str, np.ndarray | None, str | None, float] | None = None
        # Virtual controller pose per hand; the task follows changes relative to
        # the pose it saw when the grip went down.
        self.position = {side: np.zeros(3) for side in SIDES}
        self.rotation = {side: np.eye(3) for side in SIDES}
        self.reference = pin.SE3.Identity()
        self.homing = threading.Event()
        self.homing_was_stopped = False
        self.gripper: dict[str, float | None] = {side: None for side in SIDES}
        self.joints: dict[str, float] = {}
        self.pose_out = {
            side: make_transport(f"{side}_cartesian_command", PoseStamped) for side in SIDES
        }
        self.gripper_out = {
            side: make_transport(f"{side}_gripper_command", Float32) for side in SIDES
        }
        self.buttons_out = make_transport("teleop_buttons", Buttons)
        self.joints_in = make_transport("coordinator_joint_state", JointState)
        self.joints_in.subscribe(self._on_joints)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._stream, daemon=True)

    def _on_joints(self, message: JointState) -> None:
        with self.lock:
            self.joints = dict(zip(message.name, message.position, strict=False))

    def _stream(self) -> None:
        last = time.time()
        while not self.stop.is_set():
            now = time.time()
            self._apply_motion(now - last)
            last = now
            with self.lock:
                positions = {side: self.position[side].copy() for side in SIDES}
                rotations = {side: self.rotation[side].copy() for side in SIDES}
                grippers = dict(self.gripper)
                selected = self.side
            for side in SIDES:
                quaternion = pin.Quaternion(rotations[side])
                self.pose_out[side].broadcast(
                    None,
                    PoseStamped(
                        ts=time.time(),
                        frame_id="world",
                        position=Vector3(*positions[side]),
                        orientation=Quaternion(
                            quaternion.x, quaternion.y, quaternion.z, quaternion.w
                        ),
                    ),
                )
                opening = grippers[side]
                if opening is not None:
                    self.gripper_out[side].broadcast(None, Float32(opening))
            buttons = Buttons()
            buttons.left_grip = selected == "left"
            buttons.right_grip = selected == "right"
            self.buttons_out.broadcast(None, buttons)
            time.sleep(1.0 / RATE_HZ)

    def start(self) -> str:
        deadline = time.time() + 5.0
        while not self.joints and time.time() < deadline:
            time.sleep(0.05)
        if not self.joints:
            raise RuntimeError("no joint states; is keyboard-teleop-dual-openyam running?")
        self._select(self.side)
        self.thread.start()
        return self.status()

    def _select(self, side: str) -> None:
        # Switching grips makes the task re-reference both arms, so restart
        # both virtual poses from identity and note where this arm points now.
        with self.lock:
            self.side = side
            for each in SIDES:
                self.position[each] = np.zeros(3)
                self.rotation[each] = np.eye(3)
            self.reference = self.kinematics.grasp_frame(side, self.joints)

    def status(self) -> str:
        with self.lock:
            grasp = self.kinematics.grasp_frame(self.side, self.joints)
        tip = grasp.act(FINGERTIP_IN_GRASP) * 1000.0
        axis = grasp.rotation @ np.array([0.0, 0.0, -1.0])
        tilt = np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0)))
        return (
            f"[{self.side}] speed {SPEEDS_MM_S[self.speed_index]:g} mm/s / "
            f"{SPEEDS_DEG_S[self.turn_index]:g} deg/s   tip x {tip[0]:+.1f} y {tip[1]:+.1f} "
            f"z {tip[2]:+.1f} mm   tilt {tilt:.1f} deg"
        )

    def _apply_motion(self, dt: float) -> None:
        """Advance the active motion segment by one tick, stopping at its cap."""
        motion = self.motion
        if motion is None or dt <= 0.0:
            return
        key, direction, axis, sign = motion
        if direction is not None:
            step = SPEEDS_MM_S[self.speed_index] * dt
            self.segment_mm += step
            self.nudge(direction * step, np.eye(3))
            if self.segment_mm >= MAX_SEGMENT_MM:
                self.motion = None
        else:
            assert axis is not None
            step = SPEEDS_DEG_S[self.turn_index] * dt
            self.segment_deg += step
            self.nudge(np.zeros(3), _rotation(axis, sign * step))
            if self.segment_deg >= MAX_SEGMENT_DEG:
                self.motion = None

    def nudge(self, translation_mm: np.ndarray, turn: np.ndarray) -> None:
        with self.lock:
            side = self.side
            # The task turns the grasp frame about its own origin; shift it so
            # the fingertip stays where it is.
            lever = self.rotation[side] @ self.reference.rotation @ FINGERTIP_IN_GRASP
            self.position[side] = (
                self.position[side] + translation_mm / 1000.0 + lever - turn @ lever
            )
            self.rotation[side] = turn @ self.rotation[side]

    def home(self) -> None:
        """Glide the selected arm's grasp frame to the home pose; any key stops it."""
        self.glide_to(HOME_JOINTS)

    def park(self) -> None:
        """Glide home, then down onto the rest supports (all joints zero); any key stops it."""
        self.glide_to(HOME_JOINTS)
        if self.homing_was_stopped:
            return
        self.glide_to(REST_JOINTS)

    def glide_to(self, joints: tuple[float, ...]) -> None:
        self.homing_was_stopped = False
        with self.lock:
            side = self.side
            home = self.kinematics.grasp_frame(
                side, {f"{side}_joint{i + 1}": v for i, v in enumerate(joints)}
            )
            start_position = self.position[side].copy()
            start_rotation = self.rotation[side].copy()
        goal_position = home.translation - self.reference.translation
        goal_rotation = home.rotation @ self.reference.rotation.T
        turn = pin.log3(start_rotation.T @ goal_rotation)
        duration = max(
            np.linalg.norm(goal_position - start_position) / HOME_SPEED_M_S,
            np.linalg.norm(turn) / HOME_SPEED_RAD_S,
            0.5,
        )
        self.homing.set()
        start = time.time()
        while self.homing.is_set():
            fraction = min((time.time() - start) / duration, 1.0)
            with self.lock:
                self.position[side] = start_position + fraction * (goal_position - start_position)
                self.rotation[side] = start_rotation @ pin.exp3(fraction * turn)
            if fraction >= 1.0:
                break
            time.sleep(1.0 / RATE_HZ)
        else:
            self.homing_was_stopped = True
        self.homing.clear()

    def hold(self, key: str) -> str | None:
        """A repeat of the last direction key arrived: the key is held, move continuously."""
        if self.last_direction is None or self.last_direction[0] != key:
            return None
        if self.motion is None:
            self.segment_mm = 0.0
            self.segment_deg = 0.0
            self.motion = self.last_direction
            direction = self.last_direction[1]
            rate = (
                f"{SPEEDS_MM_S[self.speed_index]:g} mm/s"
                if direction is not None
                else f"{SPEEDS_DEG_S[self.turn_index]:g} deg/s"
            )
            return f"[{self.side}] holding {key}: moving at {rate}"
        return None

    def release(self) -> str | None:
        if self.motion is None:
            return None
        self.motion = None
        return self.status()

    def handle(self, key: str) -> str | None:
        if self.homing.is_set():
            self.homing.clear()
            return f"[{self.side}] homing stopped"
        if key == "h":
            self.motion = None
            threading.Thread(target=self.home, daemon=True).start()
            return f"[{self.side}] homing (any key stops)"
        if key == "P":
            self.motion = None
            threading.Thread(target=self.park, daemon=True).start()
            return f"[{self.side}] parking: home, then onto the supports (any key stops)"
        moves: dict[str, tuple[np.ndarray | None, str | None, float]] = {
            "w": (np.array([1.0, 0.0, 0.0]), None, 1.0),
            "s": (np.array([-1.0, 0.0, 0.0]), None, 1.0),
            "a": (np.array([0.0, 1.0, 0.0]), None, 1.0),
            "d": (np.array([0.0, -1.0, 0.0]), None, 1.0),
            "r": (np.array([0.0, 0.0, 1.0]), None, 1.0),
            "f": (np.array([0.0, 0.0, -1.0]), None, 1.0),
            # Turning about +y swings the fingertips toward -x (back).
            "i": (None, "y", -1.0),
            "k": (None, "y", 1.0),
            "j": (None, "x", 1.0),
            "l": (None, "x", -1.0),
            "u": (None, "z", 1.0),
            "o": (None, "z", -1.0),
        }
        if key == " ":
            self.motion = None
            return self.status()
        if key in moves:
            direction, axis, sign = moves[key]
            if direction is not None:
                self.nudge(direction * TAP_MM, np.eye(3))
            else:
                assert axis is not None
                self.nudge(np.zeros(3), _rotation(axis, sign * TAP_DEG))
            self.last_direction = (key, direction, axis, sign)
            return None
        if key in ("1", "2"):
            self.motion = None
            self._select("left" if key == "1" else "right")
            return self.status()
        if key == "-":
            self.speed_index = max(0, self.speed_index - 1)
            return self.status()
        if key == "=":
            self.speed_index = min(len(SPEEDS_MM_S) - 1, self.speed_index + 1)
            return self.status()
        if key == "[":
            self.turn_index = max(0, self.turn_index - 1)
            return self.status()
        if key == "]":
            self.turn_index = min(len(SPEEDS_DEG_S) - 1, self.turn_index + 1)
            return self.status()
        if key in ("c", "v"):
            with self.lock:
                self.gripper[self.side] = 0.0 if key == "c" else 1.0
            return f"[{self.side}] gripper {'closing' if key == 'c' else 'opening'}"
        if key == "p":
            return self.status()
        return None


def _read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    return sys.stdin.read(1) if ready else None


def main() -> None:
    remote = Remote()
    print(HELP)
    print(remote.start(), flush=True)
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        last_key, last_time = "", 0.0
        while True:
            key = _read_key(RELEASE_S / 3)
            now = time.time()
            if key is None:
                if remote.motion is not None and now - last_time > RELEASE_S:
                    message = remote.release()
                    if message:
                        print(message, flush=True)
                continue
            if key == "q":
                break
            try:
                if key == last_key and now - last_time < REPEAT_S:
                    message = remote.hold(key)
                else:
                    message = remote.handle(key)
            except Exception as exc:
                message = f"error: {exc}"
            last_key, last_time = key, now
            if message:
                print(message, flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        remote.stop.set()


if __name__ == "__main__":
    main()
