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

"""Drive a kp/kd step sweep on a running ``dual_openyam_step_test`` blueprint.

    dimos run dual_openyam_step_test --left-can-port can0 --right-can-port can1   # terminal 1
    python -m dimos.robot.manipulators.dual_openyam.calibration.demo_step_sweep   # terminal 2

For each joint and each gain scale the driver sets the gains and commands the
steps +1, -1, +3, -3 deg, holding after each with a countdown. The expected
total time is printed first. Every step waits for a prompt unless
``--no-confirm``: Enter runs it, ``s`` skips it, ``p`` parks the arms and ends
the run (Ctrl-C does the same). The module refuses steps while a joint is not
tracking its target; pass ``--go-home`` to ramp to the test pose first. The
run always ends with the baseline gains restored, a park and a clean disconnect.
"""

from __future__ import annotations

from functools import partial
import json
import math
from pathlib import Path
from typing import Any

import typer

from dimos.porcelain.dimos import Dimos
from dimos.robot.manipulators.dual_openyam.calibration.progress import (
    ParkRequestedError,
    confirm,
    countdown,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import MAX_KD
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_ARM_JOINTS

DEFAULT_KP_SCALES = "1,1.5,2"
DEFAULT_DELTAS_DEG = "1,-1,3,-3"
RPC_OVERHEAD_S = 0.2  # round trip plus the gain change between steps

app = typer.Typer(add_completion=False)


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


@app.command()
def sweep(
    joints: str = typer.Option(
        ",".join(DUAL_OPENYAM_ARM_JOINTS), help="Comma-separated arm joints to sweep"
    ),
    kp_scales: str = typer.Option(
        DEFAULT_KP_SCALES, help="kp multipliers on the baseline gains; kd scales by sqrt"
    ),
    gains_file: Path | None = typer.Option(
        None,
        help="Calibration JSON with a control_gains section: use its per-joint kp/kd "
        "as the single gain set (ignores --kp-scales and --kd)",
    ),
    fixed_kd: float | None = typer.Option(
        None,
        "--kd",
        help="Hold kd at this value for every gain set instead of scaling it "
        "(the wrist DM4310s buzz above about kd 2)",
    ),
    deltas_deg: str = typer.Option(DEFAULT_DELTAS_DEG, help="Steps per gain set, degrees"),
    hold_s: float = typer.Option(2.0, help="Hold after each step, seconds"),
    confirm_each: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ask before each step (Enter / s skip / p park)"
    ),
    go_home: bool = typer.Option(
        True, help="Ramp to the test pose before sweeping (runs start parked on the supports)"
    ),
    speed_deg_s: float = typer.Option(10.0, help="Ramp speed for --go-home"),
) -> None:
    """Run the sweep against the running step test module."""
    joint_list = [j.strip() for j in joints.split(",") if j.strip()]
    unknown = [j for j in joint_list if j not in DUAL_OPENYAM_ARM_JOINTS]
    if unknown:
        raise typer.BadParameter(f"unknown joints {unknown}")
    scales = _floats(kp_scales)
    deltas = _floats(deltas_deg)

    dimos = Dimos.connect()
    module: Any = dimos.JointStepTestModule
    # Park = ramp to home, then fold onto the supports: at most ~120 deg of travel.
    park_s = 120.0 / speed_deg_s + 2.0
    try:
        status: dict[str, Any] = module.status()
        typer.echo(
            f"connected: adapter={status['adapter']} loop={status['measured_hz']} Hz "
            f"ramping={status['ramping']}"
        )
        n_steps = len(joint_list) * len(scales) * len(deltas)
        expected_s = n_steps * (hold_s + RPC_OVERHEAD_S)
        home_s = 0.0
        if go_home:
            home_s = (
                max(abs(info["pose_error_deg"] or 0.0) for info in status["joints"].values())
                / speed_deg_s
            )
            expected_s += home_s
        typer.echo(
            f"plan: {len(joint_list)} joint(s) x {len(scales)} gain set(s) x {len(deltas)} "
            f"step(s) = {n_steps} steps, about {expected_s / 60:.1f} min of motion"
            + (" plus your confirmations" if confirm_each else "")
        )
        if go_home and confirm("Ramp both arms to the test pose?", confirm_each):
            typer.echo(
                countdown("go_to_test_pose", home_s, lambda: module.go_to_test_pose(speed_deg_s))
            )
        baseline = {name: (info["kp"], info["kd"]) for name, info in status["joints"].items()}
        chosen: dict[str, tuple[float, float]] = {}
        if gains_file is not None:
            sides = json.loads(gains_file.read_text())["control_gains"]
            for side in ("left", "right"):
                for name, value in sides.get(side, {}).items():
                    if isinstance(value, list):
                        chosen[name] = (float(value[0]), float(value[1]))
            missing = [j for j in joint_list if j not in chosen]
            if missing:
                raise typer.BadParameter(f"{gains_file} has no gains for {missing}")
            scales = [1.0]
        done = 0
        for j, joint in enumerate(joint_list, 1):
            kp0, kd0 = baseline[joint]
            try:
                for g, scale in enumerate(scales, 1):
                    if chosen:
                        kp, kd = chosen[joint]
                    else:
                        kp = kp0 * scale
                        # kd scales with sqrt(kp) to keep the damping ratio, capped at the
                        # MIT frame's maximum (5.0); the baseline arm joints already sit there.
                        kd = min(
                            fixed_kd if fixed_kd is not None else kd0 * math.sqrt(scale), MAX_KD
                        )
                    typer.echo(module.set_gains(joint, kp, kd))
                    for d, delta in enumerate(deltas, 1):
                        done += 1
                        label = (
                            f"joint {j}/{len(joint_list)}, gains {g}/{len(scales)}, "
                            f"step {d}/{len(deltas)} ({done}/{n_steps}): "
                            f"{joint} {delta:+g} deg at kp={kp:g} kd={kd:g}"
                        )
                        if not confirm(label, confirm_each):
                            continue
                        try:
                            result = countdown(
                                label, hold_s, partial(module.step, joint, delta, hold_s)
                            )
                        except ValueError as error:
                            typer.echo(f"refused: {error}", err=True)
                        else:
                            typer.echo(result)
            finally:
                module.set_gains(joint, kp0, kd0)
        typer.echo("sweep done; baseline gains restored")
    except ParkRequestedError as request:
        typer.echo(f"\n{request}: ending the run")
    except KeyboardInterrupt:
        typer.echo("\ninterrupted: ending the run")
    finally:
        try:
            typer.echo(countdown("park on the supports", park_s, lambda: module.park(speed_deg_s)))
        except Exception as error:
            typer.echo(f"park failed: {error}; the arm holds its last target", err=True)
        dimos.stop()


if __name__ == "__main__":
    app()
