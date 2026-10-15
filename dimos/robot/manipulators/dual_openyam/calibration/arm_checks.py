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

"""Dual OpenYAM arm checks, section A of the calibration procedure.

    python -m dimos.robot.manipulators.dual_openyam.calibration.arm_checks joints
    python -m dimos.robot.manipulators.dual_openyam.calibration.arm_checks zero --cycle 1
    python -m dimos.robot.manipulators.dual_openyam.calibration.arm_checks touch --side left
    python -m dimos.robot.manipulators.dual_openyam.calibration.arm_checks gripper --side left --gap-mm 47

None of these commands can move an arm. ``joints`` and ``zero`` query the
encoders with the motors off; ``touch`` and ``gripper`` listen to the joint
states of a running teleop session, which is what moves the arms.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import typer

from dimos.robot.manipulators.dual_openyam.calibration.encoders import (
    DEFAULT_CAN_PORTS,
    SIDES,
    DualOpenYamEncoders,
    EncoderReadError,
    EncoderReading,
)
from dimos.robot.manipulators.dual_openyam.calibration.joint_stream import (
    JOINT_STATE_TOPIC,
    JointStream,
    JointStreamError,
)
from dimos.robot.manipulators.dual_openyam.calibration.kinematics import (
    ArmKinematics,
    RigGeometry,
)
from dimos.robot.manipulators.dual_openyam.calibration.touch_check import (
    TouchResult,
    format_fingertip,
    format_touch,
    load_touches,
    save_touch,
    touch_record,
)
from dimos.robot.manipulators.dual_openyam.calibration.touch_grid import (
    GRID_POINT_NAMES,
    finger_tilt_deg,
    fit_arm,
    format_arm_fit,
    format_bases,
    grid_touch,
    load_grid,
    right_base_in_left_base,
    save_grid,
)
from dimos.robot.manipulators.dual_openyam.calibration.zero_check import (
    ZERO_HOLD_CYCLES,
    evaluate_zero_hold,
    format_zero_hold,
    load_snapshots,
    save_snapshot,
    snapshot_record,
)

DEFAULT_SESSION_ROOT = Path.home() / ".local/share/dimos/calibration/dual_openyam"

app = typer.Typer(
    no_args_is_help=True,
    help="Dual OpenYAM arm checks (procedure section A). No command here moves an arm.",
)
# Fully open pad gap, calipers 2026-10-14; the model's is 94 mm.
OPEN_GAP_M = 0.095
GRIPPER_TOLERANCE_M = 0.002

_LEFT_PORT = typer.Option(
    DEFAULT_CAN_PORTS["left"], "--left-can-port", help="Left arm CAN interface"
)
_RIGHT_PORT = typer.Option(
    DEFAULT_CAN_PORTS["right"], "--right-can-port", help="Right arm CAN interface"
)
_SESSION_DIR = typer.Option(
    None,
    "--session-dir",
    help=f"Where captures go; default {DEFAULT_SESSION_ROOT}/<today>",
)


def _session_dir(session_dir: Path | None) -> Path:
    return session_dir or DEFAULT_SESSION_ROOT / datetime.now().date().isoformat()


def _connect(left_can_port: str, right_can_port: str) -> DualOpenYamEncoders:
    encoders = DualOpenYamEncoders({"left": left_can_port, "right": right_can_port})
    try:
        encoders.connect()
    except EncoderReadError as exc:
        raise typer.Exit(_fail(str(exc))) from exc
    return encoders


def _fail(message: str) -> int:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return 1


def _format_joints(reading: EncoderReading) -> str:
    lines = []
    for side in SIDES:
        arm = reading.arms[side]
        joints = "  ".join(
            f"j{index + 1} {np.degrees(value):+8.2f}" for index, value in enumerate(arm.joints_rad)
        )
        lines.append(f"{side:<5} {joints}   gripper motor {arm.gripper_motor_rad:+.3f} rad")
    return "\n".join(lines)


@app.command()
def joints(
    left_can_port: str = _LEFT_PORT,
    right_can_port: str = _RIGHT_PORT,
    watch: bool = typer.Option(False, "--watch", help="Keep printing until Ctrl-C"),
) -> None:
    """Print every joint of both arms in degrees."""
    encoders = _connect(left_can_port, right_can_port)
    while True:
        try:
            typer.echo(_format_joints(encoders.read()))
        except EncoderReadError as exc:
            if not watch:
                raise typer.Exit(_fail(str(exc))) from exc
            typer.secho(str(exc), fg=typer.colors.YELLOW)
        if not watch:
            return
        typer.echo("")
        time.sleep(0.5)


@app.command()
def zero(
    cycle: int = typer.Option(
        ..., "--cycle", min=1, max=ZERO_HOLD_CYCLES, help="Power cycle number, 1 to 3"
    ),
    left_can_port: str = _LEFT_PORT,
    right_can_port: str = _RIGHT_PORT,
    session_dir: Path | None = _SESSION_DIR,
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace this cycle's capture"),
) -> None:
    """Step 1: capture the parked arms after a power cycle and compare the cycles."""
    directory = _session_dir(session_dir)
    existing = load_snapshots(directory)
    if cycle in existing and not overwrite:
        raise typer.Exit(
            _fail(f"cycle {cycle} is already captured in {directory}; pass --overwrite to redo it")
        )
    encoders = _connect(left_can_port, right_can_port)
    try:
        reading = encoders.read()
    except EncoderReadError as exc:
        raise typer.Exit(_fail(str(exc))) from exc
    path = save_snapshot(directory, snapshot_record(reading, cycle, datetime.now().astimezone()))
    typer.echo(f"Saved {path}\n")
    result = evaluate_zero_hold(load_snapshots(directory))
    typer.echo(format_zero_hold(result))
    if result.complete and not result.passed:
        raise typer.Exit(1)


_SIDE = typer.Option(..., "--side", help="left or right")
_TOPIC = typer.Option(JOINT_STATE_TOPIC, "--topic", help="Joint state stream of the teleop session")


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise typer.BadParameter("--side must be left or right")


def _capture(stream: JointStream, names: list[str]) -> np.ndarray:
    try:
        return stream.capture(names)
    except JointStreamError as exc:
        raise typer.Exit(_fail(str(exc))) from exc


@app.command()
def touch(
    side: str = _SIDE,
    topic: str = _TOPIC,
    session_dir: Path | None = _SESSION_DIR,
    base_spacing_mm: float = typer.Option(430.0, help="Arm base centre to centre"),
    finger_length_mm: float = typer.Option(
        82.5, help="Gripper housing face to the closed fingertip end"
    ),
    table_z_mm: float = typer.Option(-30.0, help="Table top relative to the base blocks"),
) -> None:
    """Step 5: capture one arm's fingertip on the cross; compare once both arms are in."""
    _check_side(side)
    geometry = RigGeometry(
        base_spacing_m=base_spacing_mm / 1000.0,
        finger_length_m={s: finger_length_mm / 1000.0 for s in SIDES},
        table_z_m=table_z_mm / 1000.0,
    )
    stream = JointStream(topic)
    try:
        joints_rad = _capture(stream, [f"{side}_joint{index}" for index in range(1, 7)])
    finally:
        stream.close()
    fingertip = ArmKinematics().fingertip_in_rig(side, joints_rad, geometry)
    directory = _session_dir(session_dir)
    path = save_touch(
        directory,
        touch_record(side, joints_rad, fingertip, geometry, datetime.now().astimezone()),
    )
    typer.echo(f"Saved {path}\n")
    typer.echo(format_fingertip(side, fingertip, geometry.table_z_m))
    touches = load_touches(directory)
    if set(touches) != set(SIDES):
        other = next(s for s in SIDES if s not in touches)
        typer.echo(f"\nNow touch the same cross with the {other} arm.")
        return
    result = TouchResult(
        fingertips_m={s: np.array(touches[s]["fingertip_rig_m"]) for s in SIDES},
        table_z_m=geometry.table_z_m,
    )
    typer.echo("\n" + format_touch(result))
    if not result.passed:
        raise typer.Exit(1)


@app.command()
def gripper(
    side: str = _SIDE,
    topic: str = _TOPIC,
    gap_mm: float = typer.Option(..., "--gap-mm", help="Pad gap you measured with calipers"),
) -> None:
    """Step 3: compare a calipered pad gap with the opening the gripper reports."""
    _check_side(side)
    stream = JointStream(topic)
    try:
        (opening,) = _capture(stream, [f"{side}_arm/gripper"])
    finally:
        stream.close()
    expected_mm = opening * OPEN_GAP_M * 1000.0
    error_mm = gap_mm - expected_mm
    passed = abs(error_mm) <= GRIPPER_TOLERANCE_M * 1000.0
    typer.echo(
        f"{side} gripper reports {opening:.1%} open -> {expected_mm:.1f} mm by the model; "
        f"you measured {gap_mm:.1f} mm ({error_mm:+.1f} mm). "
        + ("ok" if passed else f"FAIL: over {GRIPPER_TOLERANCE_M * 1000:.0f} mm")
    )
    if not passed:
        raise typer.Exit(1)


@app.command()
def grid(
    side: str = _SIDE,
    point: str = typer.Option("", "--point", help="near-left, near-right, far-left or far-right"),
    undo: bool = typer.Option(False, "--undo", help="Drop this arm's last touch instead"),
    topic: str = _TOPIC,
    session_dir: Path | None = _SESSION_DIR,
) -> None:
    """Touch grid: capture one arm's fingertip on a table-board corner."""
    _check_side(side)
    directory = _session_dir(session_dir)
    touches = load_grid(directory, side)
    if undo:
        if not touches:
            raise typer.Exit(_fail(f"no {side} touches to undo"))
        dropped = touches.pop()
        save_grid(directory, side, touches)
        typer.echo(f"Dropped {side} {dropped['point']} ({dropped['taken_at']})")
        return
    if point not in GRID_POINT_NAMES:
        raise typer.BadParameter(f"--point must be one of {', '.join(GRID_POINT_NAMES)}")
    stream = JointStream(topic)
    try:
        joints_rad = _capture(stream, [f"{side}_joint{index}" for index in range(1, 7)])
    finally:
        stream.close()
    touches.append(grid_touch(point, joints_rad, datetime.now().astimezone()))
    path = save_grid(directory, side, touches)
    tilt = finger_tilt_deg(ArmKinematics(), side, joints_rad)
    counts = ", ".join(
        f"{name} {sum(t['point'] == name for t in touches)}" for name in GRID_POINT_NAMES
    )
    typer.echo(f"Saved {side} {point}, tilt {tilt:.0f} deg -> {path}\n{side} touches: {counts}")


@app.command("grid-solve")
def grid_solve(
    session_dir: Path | None = _SESSION_DIR,
    square_mm: float = typer.Option(30.0, help="Measured table-board square size"),
    finger_length_mm: float = typer.Option(
        82.5, help="Gripper housing face to the closed fingertip end"
    ),
    offset_joints: str = typer.Option("2,3", help="Joints whose zero error to fit"),
) -> None:
    """Touch grid: fit each arm's joint zero errors and where the bases really sit."""
    directory = _session_dir(session_dir)
    geometry = RigGeometry(finger_length_m={s: finger_length_mm / 1000.0 for s in SIDES})
    joints_to_fit = tuple(int(j) for j in offset_joints.split(",") if j.strip())
    kinematics = ArmKinematics()
    fits = {}
    for side in SIDES:
        touches = load_grid(directory, side)
        if not touches:
            typer.echo(f"{side}: no touches yet")
            continue
        try:
            fits[side] = fit_arm(
                side,
                touches,
                kinematics,
                geometry,
                square_m=square_mm / 1000.0,
                offset_joints=joints_to_fit,
            )
        except ValueError as exc:
            typer.echo(f"{side}: {exc}")
            continue
        typer.echo(format_arm_fit(fits[side]) + "\n")
    if set(fits) == set(SIDES):
        typer.echo(format_bases(right_base_in_left_base(fits["left"], fits["right"]), geometry))


if __name__ == "__main__":
    app()
