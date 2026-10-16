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

"""Hand-teaching ("float") test on a running ``dual_openyam_step_test``.

    dimos run dual-openyam-step-test --left-can-port can0 --right-can-port can1   # terminal 1
    python -m dimos.robot.manipulators.dual_openyam.calibration.demo_float_test --side left
    python -m dimos.robot.manipulators.dual_openyam.calibration.float_report recordings/<run-id>

One arm goes home, then for each pose: the arm floats (kp=0, a little kd, the
adapter's gravity feed-forward does the holding), you move it by hand to a new
pose and let go, the driver waits a second, marks a ``float_events`` window and
records 10 s of joint state, then holds the arm again at the pose it measures.
The fingertip drift over each window is printed live from the joint change
through the arm model; the pass bar is 2 mm. Every prompt takes Enter (go),
``s`` (skip) or ``p`` (hold, park, quit); Ctrl-C does the same as ``p``. The
run always ends with the arm held, both arms parked on the supports and a clean
disconnect.
"""

from __future__ import annotations

from functools import partial
import math
import threading
import time
from typing import Any

import numpy as np
import typer

from dimos.porcelain.dimos import Dimos
from dimos.robot.manipulators.dual_openyam.calibration.float_analysis import (
    DRIFT_LIMIT_MM,
    FingertipModel,
    FloatEvent,
    analyze_window,
    format_metrics,
)
from dimos.robot.manipulators.dual_openyam.calibration.gravity_poses import check_side
from dimos.robot.manipulators.dual_openyam.calibration.progress import (
    ParkRequestedError,
    confirm,
    countdown,
)
from dimos.robot.manipulators.dual_openyam.calibration.step_test import DEFAULT_FLOAT_KD, MAX_KD

RPC_OVERHEAD_S = 0.2
JOINT_STATE_STREAM = "joint_state"

app = typer.Typer(add_completion=False)


class WindowRecorder:
    """Collects one arm's joint angles from the module's ``joint_state`` stream."""

    def __init__(self, joint_names: list[str]) -> None:
        self._names = joint_names
        self._lock = threading.Lock()
        self._t: list[float] = []
        self._q: list[list[float]] = []

    def __call__(self, message: Any) -> None:
        try:
            index = {name: i for i, name in enumerate(message.name)}
            row = [float(message.position[index[name]]) for name in self._names]
        except (KeyError, AttributeError, IndexError):
            return
        stamp = float(getattr(message, "ts", 0.0) or time.time())
        with self._lock:
            self._t.append(stamp)
            self._q.append(row)

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            t = np.asarray(self._t)
            q = np.asarray(self._q).reshape(-1, len(self._names))
        order = np.argsort(t, kind="stable")
        return t[order], q[order]


def _measured_rad(status: dict[str, Any], joint_names: list[str]) -> list[float]:
    return [math.radians(status["joints"][name]["measured_deg"]) for name in joint_names]


def _subscribe(module: Any, recorder: WindowRecorder) -> Any:
    """Subscribe to the module's joint_state; None when the stream is not reachable."""
    try:
        return module.get_stream(JOINT_STATE_STREAM).subscribe(recorder)
    except Exception as error:  # the status() fallback still gives a drift number
        typer.echo(f"joint_state stream unavailable ({error}); using status() snapshots", err=True)
        return None


@app.command()
def run(
    side: str = typer.Option("left", help="Which arm floats: left or right"),
    poses: int = typer.Option(5, help="How many hand-taught poses to record"),
    kd: float = typer.Option(DEFAULT_FLOAT_KD, help="Damping kept on while floating (0..5)"),
    settle_s: float = typer.Option(1.0, help="Wait after letting go before the window starts"),
    window_s: float = typer.Option(10.0, help="Recorded window per pose, seconds"),
    speed_deg_s: float = typer.Option(10.0, help="Ramp speed for the home and park moves"),
    confirm_each: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Prompt at each pose (Enter / s skip / p park)"
    ),
    go_home: bool = typer.Option(True, help="Ramp both arms to the test pose first"),
) -> None:
    """Float one arm, record N hand-taught poses, report the fingertip drift of each."""
    try:
        check_side(side)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    if poses < 1:
        raise typer.BadParameter("poses must be at least 1")
    if not 0.0 <= kd <= MAX_KD:
        raise typer.BadParameter(f"kd must be in 0..{MAX_KD:g}")
    if window_s <= 0.0 or settle_s < 0.0:
        raise typer.BadParameter("window_s must be positive and settle_s non-negative")

    typer.echo(f"loading the {side} arm model ...")
    model = FingertipModel(side)
    joint_names = model.joint_names

    dimos = Dimos.connect()
    module: Any = dimos.JointStepTestModule
    # Park = ramp to home, then fold onto the supports: at most ~120 deg of travel.
    park_s = 120.0 / speed_deg_s + 2.0
    results: list[str] = []
    try:
        status: dict[str, Any] = module.status()
        typer.echo(
            f"connected: adapter={status['adapter']} loop={status['measured_hz']} Hz "
            f"ramping={status['ramping']} floating={status.get('floating')}"
        )
        home_s = 0.0
        if go_home:
            home_s = (
                max(abs(info["pose_error_deg"] or 0.0) for info in status["joints"].values())
                / speed_deg_s
            )
        expected_s = home_s + poses * (settle_s + window_s + 3 * RPC_OVERHEAD_S)
        typer.echo(
            f"plan: {side} arm, {poses} pose(s) x {window_s:g} s windows at float kd={kd:g}, "
            f"about {expected_s / 60:.1f} min of motion plus your hand moves"
            + (" and confirmations" if confirm_each else "")
            + f"; parking takes up to {park_s:.0f} s"
        )
        if go_home and confirm("Ramp both arms to the test pose (home)?", confirm_each):
            typer.echo(countdown("go home", home_s, lambda: module.go_to_test_pose(speed_deg_s)))
        for index in range(1, poses + 1):
            label = f"pose {index}/{poses}"
            if not confirm(f"{label}: float the {side} arm?", confirm_each):
                continue
            typer.echo(module.float_arm(side, kd))
            if not confirm(
                f"{label}: move the {side} arm by hand to a new pose and let go, then Enter",
                confirm_each,
            ):
                typer.echo(module.hold_arm(side))
                continue
            countdown(f"{label}: settle", settle_s, partial(time.sleep, settle_s))
            recorder = WindowRecorder(joint_names)
            unsubscribe = _subscribe(module, recorder)
            q_first = _measured_rad(module.status(), joint_names)
            t_first = time.time()
            typer.echo(module.mark_float(side, index, window_s, {"kd": kd}))
            countdown(f"{label}: record {window_s:g} s", window_s, partial(time.sleep, window_s))
            q_last = _measured_rad(module.status(), joint_names)
            t_last = time.time()
            typer.echo(module.hold_arm(side))
            if unsubscribe is not None:
                unsubscribe()
            t, q = recorder.samples()
            if t.size < 2:
                t, q = np.asarray([t_first, t_last]), np.asarray([q_first, q_last])
            event = FloatEvent(side=side, index=index, t0=t_first, duration_s=window_s, kd=kd)
            line = format_metrics(analyze_window(model, event, t, q))
            results.append(line)
            typer.echo(line)
        typer.echo(
            f"float test done ({len(results)} window(s), pass < {DRIFT_LIMIT_MM:g} mm); report with: "
            "python -m dimos.robot.manipulators.dual_openyam.calibration.float_report "
            "recordings/<run-id>"
        )
    except ParkRequestedError as request:
        typer.echo(f"\n{request}: ending the run")
    except KeyboardInterrupt:
        typer.echo("\ninterrupted: ending the run")
    finally:
        try:
            typer.echo(module.hold_arm(side))
        except Exception as error:
            typer.echo(f"hold failed: {error}", err=True)
        try:
            typer.echo(countdown("park on the supports", park_s, lambda: module.park(speed_deg_s)))
        except Exception as error:
            typer.echo(f"park failed: {error}; the arm holds its last target", err=True)
        dimos.stop()
        if results:
            typer.echo("\n".join(["summary:", *results]))


if __name__ == "__main__":
    app()
