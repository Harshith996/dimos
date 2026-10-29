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

from pathlib import Path
import tempfile
from typing import Any, cast

import mujoco
from pydantic import Field

from dimos.core.coordination.blueprint_config.fields import module_config_cls
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.sim2.spec import Simulated, SimulatedRobot, SimulatedSensor
from dimos.sim2.world import compile_world


class SimulationConfig(ModuleConfig):
    scene: Path | None = Field(default_factory=lambda m: m["g"].scene_package)
    # describe_simulated() results keyed by instance name; filled in by Simulation.autoconnect.
    simulated: dict[str, Simulated] = Field(default_factory=dict)


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

    @rpc
    def build(self) -> None:
        super().build()
        if self.config.scene is None:
            raise ValueError("Simulation needs a scene; pass scene= or --scene-package")
        self._model = compile_world(self.config.scene, self.config.simulated.values())
        self._directory = tempfile.TemporaryDirectory(prefix="dimos-sim2-")
        mujoco.mj_saveModel(self._model, str(self.model_path), None)

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
        }

    @rpc
    def stop(self) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        super().stop()
