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

"""ChArUco boards of the dual OpenYAM workcell as the overhead D405 sees them.

Two boards (see ``~/calibration/make_boards.py``): the table board, 7 x 5
squares of 30 mm with 22 mm markers (ids 0..16), and the gripper board on the
jig, 5 x 5 squares of 25 mm with 19 mm markers (ids 100..111). Both use
``DICT_5X5_1000``. Positions are in OpenCV's board frame: origin at the
board's first corner, x along the first square row, y along the columns, z out
of the printed face.

The D405 reports its colour intrinsics with the ``inverse_brown_conrady``
model, so detected pixels are undistorted with the *forward* Brown-Conrady
formula and the pose is then solved as a pinhole (zero distortion).

Frames come either from the ``color_image`` stream of a running camera module
(:class:`StreamFrames`) or from PNGs saved earlier (:class:`SavedFrames`), so
the detection path runs identically on the rig and offline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol

import cv2
import numpy as np
import pinocchio as pin

DICTIONARY = cv2.aruco.DICT_5X5_1000
MIN_CORNERS = 6
DEFAULT_FRAME_SIZE = (1280, 720)


@dataclass(frozen=True)
class BoardSpec:
    """One printed ChArUco board."""

    name: str
    squares: tuple[int, int]  # across, down
    square_m: float
    marker_m: float
    first_id: int

    @property
    def marker_count(self) -> int:
        return (self.squares[0] * self.squares[1]) // 2

    @property
    def corner_count(self) -> int:
        return (self.squares[0] - 1) * (self.squares[1] - 1)

    @property
    def size_m(self) -> tuple[float, float]:
        return (self.squares[0] * self.square_m, self.squares[1] * self.square_m)

    def centre(self) -> np.ndarray:
        return np.array([self.size_m[0] / 2.0, self.size_m[1] / 2.0, 0.0])

    def outer_corners(self) -> dict[str, np.ndarray]:
        """The four outer corners of the checker pattern, counter-clockwise from the origin."""
        w, h = self.size_m
        return {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([w, 0.0, 0.0]),
            "C": np.array([w, h, 0.0]),
            "D": np.array([0.0, h, 0.0]),
        }

    def opencv(self) -> Any:
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        ids = np.arange(self.first_id, self.first_id + self.marker_count, dtype=np.int32)
        return cv2.aruco.CharucoBoard(self.squares, self.square_m, self.marker_m, dictionary, ids)


TABLE_BOARD = BoardSpec("table", (7, 5), 0.030, 0.022, 0)
GRIPPER_BOARD = BoardSpec("gripper", (5, 5), 0.025, 0.019, 100)
# Named points of the table board that the arms visit: the centre and the four outer corners.
TABLE_POINT_NAMES = ("centre", "A", "B", "C", "D")


def table_points(board: BoardSpec = TABLE_BOARD) -> dict[str, np.ndarray]:
    return {"centre": board.centre(), **board.outer_corners()}


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    model: str = "distortion.inverse_brown_conrady"
    width: int = DEFAULT_FRAME_SIZE[0]
    height: int = DEFAULT_FRAME_SIZE[1]

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Intrinsics:
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            coeffs=tuple(float(c) for c in data.get("coeffs", (0.0,) * 5)),
            model=str(data.get("model", "distortion.inverse_brown_conrady")),
            width=int(data.get("width", DEFAULT_FRAME_SIZE[0])),
            height=int(data.get("height", DEFAULT_FRAME_SIZE[1])),
        )

    @classmethod
    def from_calibration(cls, calibration: Mapping[str, Any]) -> Intrinsics:
        """The ``overhead_camera.intrinsics_1280x720`` block of the workcell calibration file."""
        camera = calibration["overhead_camera"]
        key = next(k for k in camera if k.startswith("intrinsics_"))
        width, height = (int(v) for v in key.removeprefix("intrinsics_").split("x"))
        return cls.from_dict({**camera[key], "width": width, "height": height})

    @classmethod
    def from_intrinsics_json(cls, path: Path) -> Intrinsics:
        """``intrinsics.json`` as ``snap_overhead.py`` / ``camera_server.py`` wrote it."""
        return cls.from_dict(json.loads(path.read_text())["color"])

    def undistort_points(self, pixels: np.ndarray) -> np.ndarray:
        """Ideal pinhole pixels for detected pixels, per the camera's distortion model."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if not any(self.coeffs) or self.model.endswith("none"):
            return pixels
        if self.model.endswith("inverse_brown_conrady"):
            k1, k2, p1, p2, k3 = (list(self.coeffs) + [0.0] * 5)[:5]
            x = (pixels[:, 0] - self.cx) / self.fx
            y = (pixels[:, 1] - self.cy) / self.fy
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
            xu = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yu = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
            return np.stack([xu * self.fx + self.cx, yu * self.fy + self.cy], axis=1)
        # Plain Brown-Conrady: OpenCV's iterative inverse.
        ideal = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), self.matrix, np.asarray(self.coeffs), P=self.matrix
        )
        return np.asarray(ideal).reshape(-1, 2)

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Ideal pinhole pixels of points in the camera frame (no distortion)."""
        p = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        return np.stack(
            [p[:, 0] / p[:, 2] * self.fx + self.cx, p[:, 1] / p[:, 2] * self.fy + self.cy], axis=1
        )


@dataclass(frozen=True)
class BoardDetection:
    board: BoardSpec
    corner_ids: np.ndarray  # (n,)
    object_points: np.ndarray  # (n, 3) board frame
    image_points: np.ndarray  # (n, 2) as detected
    image_points_ideal: np.ndarray  # (n, 2) undistorted

    @property
    def count(self) -> int:
        return len(self.corner_ids)

    def to_json(self) -> dict[str, Any]:
        return {
            "board": self.board.name,
            "corner_ids": [int(i) for i in self.corner_ids],
            "object_points": self.object_points.tolist(),
            "image_points": self.image_points.tolist(),
            "image_points_ideal": self.image_points_ideal.tolist(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], board: BoardSpec) -> BoardDetection:
        ideal = np.asarray(data["image_points_ideal"], dtype=np.float64)
        return cls(
            board=board,
            corner_ids=np.asarray(data["corner_ids"], dtype=np.int32),
            object_points=np.asarray(data["object_points"], dtype=np.float64),
            image_points=np.asarray(data.get("image_points", ideal), dtype=np.float64),
            image_points_ideal=ideal,
        )


@dataclass(frozen=True)
class BoardPose:
    board_in_camera: pin.SE3
    rms_px: float
    corners: int

    @property
    def facing_deg(self) -> float:
        """Angle between the board normal and the camera axis (0 = square on)."""
        normal = self.board_in_camera.rotation[:, 2]
        return float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.board_in_camera.translation))


def to_gray(image: np.ndarray, rgb: bool = False) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if rgb else cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)


def detect_board(
    image: np.ndarray,
    board: BoardSpec,
    intrinsics: Intrinsics,
    *,
    min_corners: int = MIN_CORNERS,
) -> BoardDetection | None:
    """ChArUco corners of ``board`` in a BGR or grey image, or None under ``min_corners``."""
    gray = to_gray(image)
    if gray.shape[1] != intrinsics.width or gray.shape[0] != intrinsics.height:
        raise ValueError(
            f"frame is {gray.shape[1]}x{gray.shape[0]} but the intrinsics are for "
            f"{intrinsics.width}x{intrinsics.height}"
        )
    cv_board = board.opencv()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(cv_board).detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj, img = cv_board.matchImagePoints(corners, ids)
    image_points = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    return BoardDetection(
        board=board,
        corner_ids=np.asarray(ids, dtype=np.int32).ravel(),
        object_points=np.asarray(obj, dtype=np.float64).reshape(-1, 3),
        image_points=image_points,
        image_points_ideal=intrinsics.undistort_points(image_points),
    )


def solve_board_pose(detection: BoardDetection, intrinsics: Intrinsics) -> BoardPose:
    """Pinhole PnP on the undistorted corners: the board frame in the camera frame."""
    ok, rvec, tvec = cv2.solvePnP(
        detection.object_points.reshape(-1, 1, 3),
        detection.image_points_ideal.reshape(-1, 1, 2),
        intrinsics.matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed on the board corners")
    rotation = cv2.Rodrigues(rvec)[0]
    pose = pin.SE3(rotation, np.asarray(tvec, dtype=np.float64).ravel())
    projected = intrinsics.project(detection.object_points @ rotation.T + pose.translation)
    rms = float(np.sqrt(np.mean(np.sum((projected - detection.image_points_ideal) ** 2, axis=1))))
    return BoardPose(board_in_camera=pose, rms_px=rms, corners=detection.count)


def best_detection(
    frames: Sequence[np.ndarray],
    board: BoardSpec,
    intrinsics: Intrinsics,
    *,
    min_corners: int = MIN_CORNERS,
) -> tuple[BoardDetection | None, int]:
    """The detection with the most corners over ``frames`` and its frame index (-1 when none)."""
    best: BoardDetection | None = None
    index = -1
    for i, frame in enumerate(frames):
        detection = detect_board(frame, board, intrinsics, min_corners=min_corners)
        if detection is not None and (best is None or detection.count > best.count):
            best, index = detection, i
    return best, index


def se3_from_pose(pose: Mapping[str, Any]) -> pin.SE3:
    """``{"xyz_m": [...], "quat_xyzw": [...]}`` as the calibration file stores poses."""
    q = pose["quat_xyzw"]
    return pin.SE3(
        pin.Quaternion(float(q[3]), float(q[0]), float(q[1]), float(q[2])).toRotationMatrix(),
        np.asarray(pose["xyz_m"], dtype=np.float64),
    )


def pose_to_dict(pose: pin.SE3) -> dict[str, Any]:
    quat = pin.Quaternion(pose.rotation).coeffs()  # x y z w
    rpy = pin.rpy.matrixToRpy(pose.rotation)
    return {
        "xyz_m": [round(float(v), 6) for v in pose.translation],
        "quat_xyzw": [round(float(v), 6) for v in quat],
        "rpy_deg": [round(float(np.degrees(v)), 3) for v in rpy],
    }


def points_in_rig(
    board_in_camera: pin.SE3,
    camera_in_rig: pin.SE3,
    points_board: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    board_in_rig = camera_in_rig * board_in_camera
    return {
        name: board_in_rig.act(np.asarray(p, dtype=np.float64)) for name, p in points_board.items()
    }


# Frame sources


class FrameSource(Protocol):
    def grab(self, count: int = 3, timeout_s: float = 5.0) -> list[np.ndarray]:
        """``count`` fresh BGR frames taken after the call."""

    def close(self) -> None: ...


class SavedFrames:
    """PNG frames from disk, served in a cycle, for offline and mock runs."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("no saved frames")
        self._next = 0

    @classmethod
    def from_dir(cls, folder: Path, pattern: str = "*.png") -> SavedFrames:
        folder = Path(folder)
        if folder.is_file():
            return cls([folder])
        paths = sorted(folder.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no {pattern} frames in {folder}")
        return cls(paths)

    def grab(self, count: int = 3, timeout_s: float = 5.0) -> list[np.ndarray]:
        frames = []
        for _ in range(count):
            path = self.paths[self._next % len(self.paths)]
            self._next += 1
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot read {path}")
            frames.append(image)
        return frames

    def close(self) -> None:
        return None


class StreamFrames:
    """Latest frames of a running camera module's ``color_image`` stream (BGR)."""

    def __init__(self, topic: str = "color_image") -> None:
        from dimos.core.transport_factory import make_transport
        from dimos.msgs.sensor_msgs.Image import Image

        self._lock = threading.Lock()
        self._frames: list[tuple[float, np.ndarray]] = []
        self._transport = make_transport(topic, Image)
        self._unsubscribe = self._transport.subscribe(self._on_image)

    def _on_image(self, image: Any) -> None:
        data = np.asarray(image.data)
        fmt = str(getattr(getattr(image, "format", None), "value", "bgr")).lower()
        if data.ndim == 3 and data.shape[2] == 3 and fmt.startswith("rgb"):
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frames.append((time.monotonic(), data))
            del self._frames[:-10]

    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def grab(self, count: int = 3, timeout_s: float = 5.0) -> list[np.ndarray]:
        start = time.monotonic()
        deadline = start + timeout_s
        while True:
            with self._lock:
                fresh = [frame for stamp, frame in self._frames if stamp >= start]
            if len(fresh) >= count:
                return fresh[:count]
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"only {len(fresh)} of {count} frames arrived on the camera stream in "
                    f"{timeout_s:g} s; is the camera blueprint running?"
                )
            time.sleep(0.02)

    def close(self) -> None:
        try:
            self._unsubscribe()
        finally:
            self._transport.stop()


@dataclass
class BoardSighting:
    """What one grab saw: the best detection, its pose and which frame it came from."""

    detection: BoardDetection | None
    pose: BoardPose | None
    frame_index: int
    frames: list[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def seen(self) -> bool:
        return self.pose is not None


def sight_board(
    source: FrameSource,
    board: BoardSpec,
    intrinsics: Intrinsics,
    *,
    count: int = 3,
    min_corners: int = MIN_CORNERS,
    timeout_s: float = 5.0,
) -> BoardSighting:
    frames = source.grab(count, timeout_s)
    detection, index = best_detection(frames, board, intrinsics, min_corners=min_corners)
    pose = solve_board_pose(detection, intrinsics) if detection is not None else None
    return BoardSighting(detection=detection, pose=pose, frame_index=index, frames=frames)
