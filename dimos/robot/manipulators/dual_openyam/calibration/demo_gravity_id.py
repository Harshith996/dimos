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

"""Drive the gravity-identification pose sweep on a running ``dual_openyam_step_test``.

    dimos run dual-openyam-step-test --left-can-port can0 --right-can-port can1   # terminal 1
    python -m dimos.robot.manipulators.dual_openyam.calibration.demo_gravity_id [--side right]
    python -m dimos.robot.manipulators.dual_openyam.calibration.gravity_fit recordings/<run-id>

For every pose of ``gravity_poses.generate_poses`` the driver marks the pose
(``gravity_events``), ramps the chosen arm there at 10 deg/s, holds, then
dithers joints 2-5 by +d and -d degrees, holding after each, so the offline
fit can cancel friction. Every wait shows a countdown and the expected total
time is printed first. Each pose waits for a prompt unless ``--no-confirm``:
Enter continues, ``s`` skips the pose, ``p`` parks the arms and ends the run
(Ctrl-C does the same). The other arm stays where the module holds it. The run
always ends with the end marker, a park on the supports and a clean disconnect.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import math
import time
from typing import Any

import typer

from dimos.porcelain.dimos import Dimos
from dimos.robot.manipulators.dual_openyam.calibration.gravity_poses import (
    ArmModel,
    Pose,
    format_pose_set,
    generate_poses,
    home_pose,
)
from dimos.robot.manipulators.dual_openyam.calibration.progress import (
    ParkRequestedError,
    confirm,
    countdown,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import MAX_STEP_DEG

RPC_OVERHEAD_S = 0.2
END_LABEL = "end"

app = typer.Typer(add_completion=False)


def travel_s(a: Sequence[float], b: Sequence[float], speed_deg_s: float) -> float:
    return math.degrees(max(abs(x - y) for x, y in zip(a, b, strict=True))) / speed_deg_s


def plan_time_s(
    start: Sequence[float],
    poses: Sequence[Pose],
    *,
    speed_deg_s: float,
    hold_s: float,
    dither_hold_s: float,
    joints: int,
    side: str = "left",
) -> float:
    """Ramps between consecutive poses (and back home), holds and dithers, plus RPC overhead."""
    total = 0.0
    previous = tuple(start)
    for pose in poses:
        total += travel_s(previous, pose.joints_rad, speed_deg_s) + RPC_OVERHEAD_S
        total += hold_s + joints * 2 * (dither_hold_s + RPC_OVERHEAD_S)
        previous = pose.joints_rad
    return total + travel_s(previous, home_pose(side).joints_rad, speed_deg_s)


def dither_plan(dithers: dict[str, float]) -> list[tuple[str, float]]:
    """``(joint, delta_deg)`` in order: +d then -d per joint, so each joint ends where it started."""
    return [(joint, sign * size) for joint, size in dithers.items() for sign in (1.0, -1.0)]


def _measured(status: dict[str, Any], joint_names: Sequence[str]) -> tuple[float, ...]:
    return tuple(math.radians(status["joints"][name]["measured_deg"]) for name in joint_names)


def _path(arm: ArmModel, current: Sequence[float], target: Pose) -> list[Pose] | None:
    """Direct ramp when it clears the table, otherwise via home, otherwise None."""
    if arm.transition_ok(current, target.joints_rad):
        return [target]
    home = home_pose(arm.side)
    if arm.transition_ok(current, home.joints_rad) and arm.transition_ok(
        home.joints_rad, target.joints_rad
    ):
        return [home, target]
    return None


@app.command()
def run(
    side: str = typer.Option("left", help="Which arm to identify: left or right"),
    poses: int = typer.Option(15, help="Number of identification poses"),
    dither_deg: float = typer.Option(1.0, help="Dither size for joints 2 and 3, degrees"),
    wrist_dither_deg: float = typer.Option(
        1.0, help="Dither size for joints 4 and 5, degrees (3 helps break stiction at kp=10)"
    ),
    hold_s: float = typer.Option(2.0, help="Hold after reaching a pose, seconds"),
    dither_hold_s: float = typer.Option(1.5, help="Hold after each dither, seconds"),
    speed_deg_s: float = typer.Option(10.0, help="Ramp speed between poses"),
    gravity_free: bool = typer.Option(
        False, "--gravity-free", help="Also visit the gravity-free poses after the sweep"
    ),
    confirm_each: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ask before each pose (Enter / s skip / p park)"
    ),
) -> None:
    """Visit the identification poses and dither joints 2-5 of one arm at each one."""
    if not 0.0 < dither_deg <= MAX_STEP_DEG or not 0.0 < wrist_dither_deg <= MAX_STEP_DEG:
        raise typer.BadParameter(f"dithers must be in (0, {MAX_STEP_DEG:g}] deg")
    try:
        arm = ArmModel(side=side)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(f"generating {side} arm poses ...")
    pose_set = generate_poses(arm, count=poses)
    typer.echo(format_pose_set(arm, pose_set))
    plan = list(pose_set.identification)
    if gravity_free:
        plan.extend(g.pose for g in pose_set.gravity_free.values())
    dithers = {
        joint: (dither_deg if joint.endswith(("joint2", "joint3")) else wrist_dither_deg)
        for joint in arm.measured_joints
    }
    home = home_pose(side)

    dimos = Dimos.connect()
    module: Any = dimos.JointStepTestModule
    # Park = ramp to home, then fold onto the supports: at most ~120 deg of travel.
    park_s = 120.0 / speed_deg_s + 2.0
    sweep_started = False
    try:
        status: dict[str, Any] = module.status()
        typer.echo(
            f"connected: adapter={status['adapter']} loop={status['measured_hz']} Hz "
            f"ramping={status['ramping']}"
        )
        current = _measured(status, arm.joint_names)
        expected = travel_s(current, home.joints_rad, speed_deg_s) + plan_time_s(
            home.joints_rad,
            plan,
            speed_deg_s=speed_deg_s,
            hold_s=hold_s,
            dither_hold_s=dither_hold_s,
            joints=len(arm.measured_joints),
            side=side,
        )
        typer.echo(
            f"plan: {side} arm, {len(plan)} pose(s), {len(arm.measured_joints)} joints x 2 "
            f"dithers each, about {expected / 60:.1f} min of motion"
            + (" plus confirmations" if confirm_each else "")
            + f"; parking takes up to {park_s:.0f} s"
        )
        if not confirm("Ramp both arms to the test pose (home)?", confirm_each):
            raise ParkRequestedError("home ramp skipped")
        typer.echo(
            countdown(
                "go home",
                travel_s(current, home.joints_rad, speed_deg_s),
                lambda: module.go_to_test_pose(speed_deg_s),
            )
        )
        current = home.joints_rad
        started = time.monotonic()
        for index, pose in enumerate(plan, 1):
            angles = " ".join(f"{v:.1f}" for v in pose.joints_deg)
            label = f"pose {index}/{len(plan)} {pose.name} [{angles}] deg"
            if not confirm(label, confirm_each):
                continue
            path = _path(arm, current, pose)
            if path is None:
                typer.echo(f"{label}: no table-clearing ramp from the current pose; skipped")
                continue
            sweep_started = True
            module.mark(
                pose.name, {"index": index, "side": side, "targets_deg": list(pose.joints_deg)}
            )
            for waypoint in path:
                typer.echo(
                    countdown(
                        f"{label}: ramp to {waypoint.name}",
                        travel_s(current, waypoint.joints_rad, speed_deg_s),
                        partial(module.go_to, waypoint.as_targets(), speed_deg_s),
                    )
                )
                current = waypoint.joints_rad
            countdown(f"{label}: settle", hold_s, partial(time.sleep, hold_s))
            for joint, delta in dither_plan(dithers):
                try:
                    typer.echo(
                        countdown(
                            f"{label}: {joint} {delta:+g} deg",
                            dither_hold_s,
                            partial(module.step, joint, delta, dither_hold_s),
                        )
                    )
                except ValueError as error:
                    # A refusal means the arm is not tracking; the pose is unusable.
                    typer.echo(f"refused: {error}; skipping the rest of this pose", err=True)
                    break
            elapsed = (time.monotonic() - started) / 60
            typer.echo(f"{label}: done, {elapsed:.1f} min elapsed")
        typer.echo(
            "sweep done; fit with: python -m "
            f"dimos.robot.manipulators.dual_openyam.calibration.gravity_fit recordings/<run-id> "
            f"--side {side}"
        )
    except ParkRequestedError as request:
        typer.echo(f"\n{request}: ending the run")
    except KeyboardInterrupt:
        typer.echo("\ninterrupted: ending the run")
    finally:
        if sweep_started:
            try:
                module.mark(END_LABEL, {"side": side})
            except Exception as error:
                typer.echo(f"end marker failed: {error}", err=True)
        try:
            typer.echo(countdown("park on the supports", park_s, lambda: module.park(speed_deg_s)))
        except Exception as error:
            typer.echo(f"park failed: {error}; the arm holds its last target", err=True)
        dimos.stop()


if __name__ == "__main__":
    app()
