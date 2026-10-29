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

"""One simulated world, built from every module in the blueprint that describes itself."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import tempfile
import time
from typing import Any, Protocol, cast

import mujoco
from pydantic import Field

from dimos.core.coordination.blueprint_config.fields import module_config_cls
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.sim2.spec import Simulated, SimulatedRobot, SimulatedSensor
from dimos.sim2.state import SharedState
from dimos.sim2.world import compile_world
from dimos.spec.utils import Spec

# Physics catches up to the wall clock in bursts, publishing after each, since macOS oversleeps short
# waits. Lag past one burst is dropped: a controller's command must not land a whole control period late.
_MAX_BURST_SECONDS = 0.01


class SimulationSpec(Spec, Protocol):
    def describe(self) -> dict[str, Any]: ...


class SimulationConfig(ModuleConfig):
    scene: Path | None = Field(default_factory=lambda m: m["g"].scene_package)
    # describe_simulated() results keyed by instance name; filled in by Simulation.autoconnect.
    simulated: dict[str, Simulated] = Field(default_factory=dict)
    # Sim seconds per wall second.
    speed: float = 1.0


class Simulation(Module):
    config: SimulationConfig

    @classmethod
    def autoconnect(cls, *blueprints: Blueprint, **config: Any) -> Blueprint:
        """autoconnect(*blueprints) plus a Simulation of every module with describe_simulated.

        MuJoCo compiles every robot together, so descriptions are read from the blueprint's
        atoms before anything is deployed. They see the blueprint's kwargs, not CLI overrides.
        """
        composed = autoconnect(*blueprints)
        simulated = {
            atom.name: cast("Any", atom.module).describe_simulated(
                module_config_cls(atom)(**atom.kwargs)
            )
            for atom in composed.active_blueprints
            if callable(getattr(atom.module, "describe_simulated", None))
        }
        return autoconnect(composed, cls.blueprint(simulated=simulated, **config))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model: mujoco.MjModel | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._state: SharedState | None = None

    @rpc
    def build(self) -> None:
        super().build()
        if self.config.scene is None:
            raise ValueError("Simulation needs a scene; pass scene= or --scene-package")
        self._model = compile_world(self.config.scene, self.config.simulated.values())
        self._directory = tempfile.TemporaryDirectory(prefix="dimos-sim2-")
        mujoco.mj_saveModel(self._model, str(self.model_path), None)
        self._state = SharedState(
            self._model, Path(self._directory.name) / "state.bin", create=True
        )

    async def main(self) -> AsyncIterator[None]:
        self._stepping = asyncio.create_task(self._step_in_real_time())
        yield

    async def _step_in_real_time(self) -> None:
        assert self._model is not None and self._state is not None
        model, state = self._model, self._state
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state.publish(data)
        started = time.monotonic()
        max_steps = max(1, int(_MAX_BURST_SECONDS / model.opt.timestep))
        while True:
            steps = int(
                ((time.monotonic() - started) * self.config.speed - data.time) / model.opt.timestep
            )
            if steps > max_steps:
                started += (steps - max_steps) * model.opt.timestep / self.config.speed
                steps = max_steps
            for _ in range(steps):
                data.ctrl[:] = state.ctrl
                mujoco.mj_step(model, data)
            state.publish(data)
            await asyncio.sleep(0.001)

    @property
    def model_path(self) -> Path:
        if self._directory is None:
            raise RuntimeError("Simulation.build() has not run")
        return Path(self._directory.name) / "world.mjb"

    @rpc
    def describe(self) -> dict[str, Any]:
        """The compiled model's path and what went into it."""
        if self._model is None:
            raise RuntimeError("Simulation.build() has not run")
        values = self.config.simulated.values()
        return {
            "model": str(self.model_path),
            "robots": sorted(d.robot_id for d in values if isinstance(d, SimulatedRobot)),
            "sensors": sorted(
                f"{d.robot_id or 'world'}/{d.name}"
                for d in values
                if isinstance(d, SimulatedSensor)
            ),
            "bodies": self._model.nbody,
            "state": str(self._state.path) if self._state is not None else None,
        }

    @rpc
    def stop(self) -> None:
        super().stop()
        self._state = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
