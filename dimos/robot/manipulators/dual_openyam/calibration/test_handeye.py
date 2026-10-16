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

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import pinocchio as pin
import pytest

from dimos.robot.manipulators.dual_openyam.calibration.board_vision import (
    GRIPPER_BOARD,
    TABLE_BOARD,
    BoardDetection,
    Intrinsics,
    SavedFrames,
    detect_board,
    points_in_rig,
    se3_from_pose,
    sight_board,
    solve_board_pose,
    table_points,
)
from dimos.robot.manipulators.dual_openyam.calibration.handeye_poses import (
    CameraView,
    JigBoard,
    generate_handeye_poses,
)
from dimos.robot.manipulators.dual_openyam.calibration.handeye_solve import (
    DEFAULT_CALIBRATION,
    HandEyeRecord,
    fit_handeye,
    load_records,
    save_records,
    write_outputs,
)
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import ArmKinematics

SESSION = Path.home() / ".local/share/dimos/calibration/dual_openyam/2026-10-14"
needs_frames = pytest.mark.skipif(
    not (SESSION / "handeye/frames").is_dir(), reason="saved overhead frames not on this machine"
)


@pytest.fixture(scope="module")
def calibration() -> dict:
    return json.loads(DEFAULT_CALIBRATION.read_text())


@pytest.fixture(scope="module")
def kinematics() -> ArmKinematics:
    return ArmKinematics()


def test_intrinsics_undistort_is_identity_without_coefficients() -> None:
    intrinsics = Intrinsics(600.0, 600.0, 640.0, 360.0, coeffs=(0.0,) * 5)
    pixels = np.array([[100.0, 50.0], [900.0, 700.0]])
    assert np.allclose(intrinsics.undistort_points(pixels), pixels)


def test_inverse_brown_conrady_moves_edge_pixels_more_than_centre(calibration: dict) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    centre = intrinsics.undistort_points(np.array([[intrinsics.cx, intrinsics.cy]]))
    edge = intrinsics.undistort_points(np.array([[50.0, 50.0]]))
    assert np.allclose(centre, [[intrinsics.cx, intrinsics.cy]], atol=1e-9)
    assert np.linalg.norm(edge - [[50.0, 50.0]]) > 1.0


def _render(board, intrinsics: Intrinsics, board_in_camera: pin.SE3) -> np.ndarray:
    """Project every board corner into a synthetic 1280x720 detection (no image)."""
    cv_board = board.opencv()
    obj = np.asarray(cv_board.getChessboardCorners(), dtype=np.float64)
    ids = np.arange(len(obj), dtype=np.int32)
    points = obj @ board_in_camera.rotation.T + board_in_camera.translation
    return BoardDetection(board, ids, obj, intrinsics.project(points), intrinsics.project(points))


def test_solve_board_pose_recovers_a_synthetic_pose(calibration: dict) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    truth = pin.SE3(pin.rpy.rpyToMatrix(0.2, -0.1, 0.4), np.array([0.05, -0.02, 0.5]))
    pose = solve_board_pose(_render(GRIPPER_BOARD, intrinsics, truth), intrinsics)
    assert pose.rms_px < 1e-6
    assert np.allclose(pose.board_in_camera.translation, truth.translation, atol=1e-6)
    assert np.allclose(pose.board_in_camera.rotation, truth.rotation, atol=1e-6)


@needs_frames
def test_gripper_board_is_detected_on_the_saved_right_frames(calibration: dict) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    frames = sorted((SESSION / "handeye/frames").glob("right_*_0.png"))[:6]
    seen = 0
    for path in frames:
        detection = detect_board(cv2.imread(str(path)), GRIPPER_BOARD, intrinsics)
        if detection is None:
            continue
        seen += 1
        assert set(detection.corner_ids) <= set(range(GRIPPER_BOARD.corner_count))
        pose = solve_board_pose(detection, intrinsics)
        assert pose.rms_px < 1.0
        assert 0.2 < pose.distance_m < 0.7
    assert seen >= 3


@needs_frames
def test_table_board_corners_land_on_the_table(calibration: dict) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    camera_in_rig = se3_from_pose(calibration["overhead_camera"]["camera_in_rig"])
    sighting = sight_board(
        SavedFrames.from_dir(SESSION / "overhead_board", "color_*.png"), TABLE_BOARD, intrinsics
    )
    assert sighting.pose is not None and sighting.pose.rms_px < 1.0
    points = points_in_rig(sighting.pose.board_in_camera, camera_in_rig, table_points())
    table_z = calibration["table"]["top_z_in_rig_m"]
    for point in points.values():
        assert abs(point[2] - table_z) < 0.006
    assert np.linalg.norm(points["A"] - points["B"]) == pytest.approx(0.21, abs=0.002)
    assert np.linalg.norm(points["B"] - points["C"]) == pytest.approx(0.15, abs=0.002)


def test_handeye_poses_face_the_camera_and_clear_the_table(calibration: dict, kinematics) -> None:
    jig = JigBoard.from_options("+z", (0.0, 0.0, -0.146))
    pose_set = generate_handeye_poses(
        "right", calibration=calibration, jig=jig, count=6, kinematics=kinematics
    )
    view = CameraView.from_calibration(calibration)
    assert len(pose_set.poses) == 6
    for pose, facing in zip(pose_set.poses, pose_set.facing_deg, strict=True):
        assert facing <= 45.0
        gripper = view.gripper_in_rig(kinematics, "right", pose.joints_rad)
        assert view.board_visible(gripper, jig)
    # Spread: no two poses coincide.
    joints = np.array([p.joints_rad for p in pose_set.poses])
    distances = np.linalg.norm(joints[:, None] - joints[None], axis=2)
    assert distances[~np.eye(len(joints), dtype=bool)].min() > math.radians(10.0)


def test_board_normal_option_is_validated() -> None:
    with pytest.raises(ValueError, match="board normal"):
        JigBoard.from_options("+w")


def _synthetic_records(
    side: str,
    kinematics: ArmKinematics,
    intrinsics: Intrinsics,
    camera_in_base: pin.SE3,
    board_in_gripper: pin.SE3,
    true_offsets: dict[int, float],
    poses_deg: list[list[float]],
    noise_px: float = 0.0,
) -> list[HandEyeRecord]:
    rng = np.random.default_rng(1)
    records = []
    for i, deg in enumerate(poses_deg):
        encoder = np.radians(deg)
        true = encoder.copy()
        for joint, offset in true_offsets.items():
            true[joint - 1] += offset
        board_in_camera = (
            camera_in_base.inverse()
            * kinematics.gripper_in_base(side, list(true))
            * board_in_gripper
        )
        detection = _render(GRIPPER_BOARD, intrinsics, board_in_camera)
        ideal = detection.image_points_ideal + rng.normal(
            0.0, noise_px, detection.image_points_ideal.shape
        )
        records.append(
            HandEyeRecord(
                joints_rad=[float(v) for v in encoder],
                detection=BoardDetection(
                    GRIPPER_BOARD, detection.corner_ids, detection.object_points, ideal, ideal
                ),
                name=f"pose_{i:02d}",
            )
        )
    return records


SYNTHETIC_POSES_DEG = [
    [40.0, 75.0, 52.0, -75.0, 75.0, -60.0],
    [40.0, 100.0, 79.0, -75.0, 37.5, 30.0],
    [-40.0, 25.0, 52.0, -75.0, -37.5, 0.0],
    [40.0, 75.0, 25.0, 37.5, 0.0, 60.0],
    [20.0, 25.0, 25.0, -37.5, 0.0, -60.0],
    [40.0, 25.0, 25.0, -37.5, 37.5, 30.0],
    [40.0, 75.0, 79.0, -75.0, 0.0, -60.0],
    [20.0, 50.0, 52.0, -75.0, 0.0, 0.0],
    [0.0, 75.0, 52.0, -37.5, 37.5, 60.0],
    [-20.0, 25.0, 25.0, -37.5, 0.0, 30.0],
    [40.0, 75.0, 52.0, -37.5, 37.5, -30.0],
    [20.0, 100.0, 52.0, -75.0, 75.0, 0.0],
    [0.0, 50.0, 25.0, 0.0, 0.0, 60.0],
    [40.0, 50.0, 25.0, -37.5, 75.0, -30.0],
    [20.0, 25.0, 25.0, -37.5, 37.5, -30.0],
]


def test_fit_recovers_injected_camera_pose_and_joint_offsets(calibration: dict, kinematics) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    base_in_rig = se3_from_pose(calibration["arms"]["right"]["base_in_rig"])
    camera_in_base = base_in_rig.inverse() * se3_from_pose(
        calibration["overhead_camera"]["camera_in_rig"]
    )
    # Nudge the camera so the fit cannot just echo the calibration it never sees anyway.
    camera_in_base = camera_in_base * pin.SE3(
        pin.rpy.rpyToMatrix(0.03, -0.02, 0.05), np.array([0.01, -0.02, 0.015])
    )
    board_in_gripper = JigBoard.from_options("+z", (0.0, 0.0, -0.146)).nominal_in_gripper()
    true_offsets = {
        2: math.radians(1.5),
        3: math.radians(-2.0),
        4: math.radians(0.8),
        5: math.radians(-0.5),
    }
    records = _synthetic_records(
        "right",
        kinematics,
        intrinsics,
        camera_in_base,
        board_in_gripper,
        true_offsets,
        SYNTHETIC_POSES_DEG,
        noise_px=0.2,
    )
    fit = fit_handeye("right", records, kinematics=kinematics, intrinsics=intrinsics)
    assert fit.rms_px < 0.4
    assert fit.outliers == []
    assert np.allclose(fit.camera_in_base.translation, camera_in_base.translation, atol=0.003)
    assert (
        math.degrees(
            np.linalg.norm(pin.log3(fit.camera_in_base.rotation.T @ camera_in_base.rotation))
        )
        < 0.3
    )
    for joint, offset, sigma in zip(
        fit.offset_joints, fit.offsets_rad, fit.offset_sigma_rad, strict=True
    ):
        assert math.degrees(abs(offset - true_offsets[joint])) < 0.3, joint
        assert math.degrees(sigma) < 0.5
    assert np.allclose(fit.board_in_gripper.translation, board_in_gripper.translation, atol=0.003)


def test_fit_flags_a_corrupted_pose_as_an_outlier(calibration: dict, kinematics) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    base_in_rig = se3_from_pose(calibration["arms"]["right"]["base_in_rig"])
    camera_in_base = base_in_rig.inverse() * se3_from_pose(
        calibration["overhead_camera"]["camera_in_rig"]
    )
    board_in_gripper = JigBoard.from_options().nominal_in_gripper()
    records = _synthetic_records(
        "right",
        kinematics,
        intrinsics,
        camera_in_base,
        board_in_gripper,
        {},
        SYNTHETIC_POSES_DEG[:10],
        noise_px=0.1,
    )
    # Pose 4's encoder reading is wrong by 3 degrees on joint 2: a slipped reading.
    bad = records[3]
    bad.joints_rad[1] += math.radians(3.0)
    fit = fit_handeye("right", records, kinematics=kinematics, intrinsics=intrinsics)
    assert fit.outliers == [3]
    assert 3 not in fit.used
    assert fit.rms_px < 0.3
    assert all(math.degrees(abs(o)) < 0.3 for o in fit.offsets_rad)


def test_records_round_trip_and_outputs_are_written(
    tmp_path: Path, calibration: dict, kinematics
) -> None:
    intrinsics = Intrinsics.from_calibration(calibration)
    base_in_rig = se3_from_pose(calibration["arms"]["left"]["base_in_rig"])
    camera_in_base = base_in_rig.inverse() * se3_from_pose(
        calibration["overhead_camera"]["camera_in_rig"]
    )
    records = _synthetic_records(
        "left",
        kinematics,
        intrinsics,
        camera_in_base,
        JigBoard.from_options().nominal_in_gripper(),
        {},
        [[-p[0], *p[1:]] for p in SYNTHETIC_POSES_DEG[:8]],
    )
    path = tmp_path / "captures_left.json"
    save_records(path, records)
    loaded = load_records(path)
    assert len(loaded) == 8 and loaded[0].name == "pose_00"
    assert np.allclose(
        loaded[2].detection.image_points_ideal, records[2].detection.image_points_ideal
    )
    fit = fit_handeye("left", loaded, kinematics=kinematics, intrinsics=intrinsics)
    json_path, md_path = write_outputs(
        fit, tmp_path, source=str(path), calibration=calibration, check=None, intrinsics=intrinsics
    )
    result = json.loads(json_path.read_text())
    assert result["side"] == "left" and "camera_in_rig" in result
    assert set(result["fit"]["joint_zero_offsets_rad"]) == {"joint2", "joint3", "joint4", "joint5"}
    assert "hand-eye (left arm)" in md_path.read_text()


def test_capture_marker_round_trips_through_a_recording(
    tmp_path: Path, calibration: dict, kinematics
) -> None:
    """In-process stand-in for the driver: mock module, go_to a pose, mark the capture, read it back."""
    from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
    from dimos.memory.store.sqlite import SqliteStore
    from dimos.msgs.std_msgs.String import String
    from dimos.robot.manipulators.dual_openyam.calibration.handeye_solve import CAPTURE_LABEL
    from dimos.robot.manipulators.dual_openyam.calibration.step_test import JointStepTestModule
    from dimos.robot.manipulators.dual_openyam.config import DUAL_OPENYAM_HOME_JOINTS
    from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS

    intrinsics = Intrinsics.from_calibration(calibration)
    base_in_rig = se3_from_pose(calibration["arms"]["right"]["base_in_rig"])
    camera_in_base = base_in_rig.inverse() * se3_from_pose(
        calibration["overhead_camera"]["camera_in_rig"]
    )
    pose_set = generate_handeye_poses(
        "right", calibration=calibration, count=2, kinematics=kinematics
    )
    recording = tmp_path / "run"
    store = SqliteStore(path=str(recording / "memory.db"))
    store.start()
    adapter = MockWholeBodyAdapter(
        dof=len(DUAL_OPENYAM_JOINTS), initial_positions=[*DUAL_OPENYAM_HOME_JOINTS, 0.0, 0.0]
    )
    module = JointStepTestModule(adapter=adapter, loop_hz=200.0)
    events = store.stream("gravity_events", String)
    module.gravity_events.subscribe(lambda msg: events.append(msg))
    try:
        module.start()
        for index, pose in enumerate(pose_set.poses, 1):
            module.go_to(pose.as_targets(), 30.0)
            time.sleep(0.05)  # the measured state is one control tick behind the target
            status = module.status()
            measured = [
                math.radians(status["joints"][n]["measured_deg"]) for n in pose.as_targets()
            ]
            assert np.allclose(measured, pose.joints_rad, atol=1e-6)  # the mock tracks the target
            board_in_camera = (
                camera_in_base.inverse()
                * kinematics.gripper_in_base("right", measured)
                * JigBoard.from_options().nominal_in_gripper()
            )
            record = HandEyeRecord(
                measured, _render(GRIPPER_BOARD, intrinsics, board_in_camera), pose.name
            )
            module.mark(
                CAPTURE_LABEL, {"side": "right", "index": index, "record": record.to_json()}
            )
        module.mark("handeye_end", {"side": "right"})
    finally:
        module.stop()
        store.stop()
    records = load_records(recording, "right")
    assert [r.name for r in records] == ["handeye_01", "handeye_02"]
    assert load_records(recording, "left") == []
    assert records[1].detection.count == GRIPPER_BOARD.corner_count
