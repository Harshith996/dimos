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

"""TEMP: hand-teaching switch for the running dual-openyam-step-test.

python -m ...demo_teach float left    # kp 0: move the arm by hand, it stays where you leave it
python -m ...demo_teach hold left     # lock it where it is
python -m ...demo_teach park          # hold both arms, then fold them onto the supports
"""

from __future__ import annotations

import typer

from dimos.porcelain.dimos import Dimos

app = typer.Typer(add_completion=False)


def _module():  # type: ignore[no-untyped-def]
    return Dimos.connect().JointStepTestModule


@app.command("float")
def float_(side: str, kd: float = 0.5) -> None:
    typer.echo(_module().float_arm(side, kd))


@app.command()
def hold(side: str) -> None:
    typer.echo(_module().hold_arm(side))


@app.command()
def park() -> None:
    module = _module()
    for side in ("left", "right"):
        module.hold_arm(side)
    typer.echo("parking both arms (about 15 s)...")
    typer.echo(module.park())


if __name__ == "__main__":
    app()
