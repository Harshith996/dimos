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

"""Exercise the full Go2 blueprint through cmd_vel; no direct motor or physics writes."""

from collections import deque
from dataclasses import replace
import json
import time
from typing import Any

import numpy as np

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.robot.get_all_blueprints import get_blueprint_by_name
from dimos.robot.unitree.go2.freewalk_connection import Go2FreewalkConnection
from dimos.sim2.module import SimulationModule


def main() -> None:
    global_config.update(simulation="mujoco", viewer="none", transport="zenoh")
    blueprint = get_blueprint_by_name("unitree-go2-freewalk")
    blueprint = replace(
        blueprint,
        blueprints=tuple(
            replace(atom, kwargs={**atom.kwargs, "viewer": False})
            if atom.module is SimulationModule
            else atom
            for atom in blueprint.blueprints
        ),
    )
    parsed = BlueprintConfigParser(blueprint).parse(
        global_overrides=global_config.model_dump(mode="python")
    )
    poses: deque[PoseStamped] = deque(maxlen=10000)
    odom = make_transport("odom", PoseStamped)
    command = make_transport("cmd_vel", Twist)
    unsubscribe = odom.subscribe(poses.append)
    coordinator = None
    try:
        before = time.monotonic()
        coordinator = ModuleCoordinator.build(blueprint, parsed_config=parsed)
        startup = time.monotonic() - before
        sim = coordinator.get_instance(SimulationModule)
        policy = coordinator.get_instance(Go2FreewalkConnection)
        deadline = time.monotonic() + 15
        while not poses or not policy.status()["armed"]:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Go2 did not arm: {policy.status()}")
            time.sleep(0.05)
        command.start()
        initial_sim = sim.status()
        initial_ticks = policy.status()["policy_ticks"]
        start = time.monotonic()
        phases: list[dict[str, Any]] = []
        for name, seconds, vx, vy, yaw in (
            ("stand", 2, 0, 0, 0),
            ("forward", 4, 0.3, 0, 0),
            ("lateral", 3, 0, 0.2, 0),
            ("turn", 3, 0, 0, 0.3),
            ("stop", 2, 0, 0, 0),
        ):
            first = poses[-1]
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                command.broadcast(None, Twist(linear=(vx, vy, 0), angular=(0, 0, yaw)))
                time.sleep(0.04)
            last = poses[-1]
            c, s = np.cos(first.yaw), np.sin(first.yaw)
            dx, dy = last.x - first.x, last.y - first.y
            phases.append(
                {
                    "name": name,
                    "body_dx_m": float(c * dx + s * dy),
                    "body_dy_m": float(-s * dx + c * dy),
                    "yaw_change_rad": float(np.angle(np.exp(1j * (last.yaw - first.yaw)))),
                    "height_m": last.z,
                }
            )
        elapsed = time.monotonic() - start
        status = policy.status()
        result = {
            "startup_seconds": startup,
            "elapsed_seconds": elapsed,
            "realtime_factor": (sim.status()["sim_time"] - initial_sim["sim_time"]) / elapsed,
            "policy_hz": (status["policy_ticks"] - initial_ticks) / elapsed,
            "phases": phases,
            "policy": status,
        }
        print(json.dumps(result, indent=2), flush=True)
        assert status["fault"] is None and status["armed"], result
        assert min(p["height_m"] for p in phases) > 0.2, "Go2 fell"
        assert phases[1]["body_dx_m"] > 0.4, "forward command did not reach the motors"
        assert phases[2]["body_dy_m"] > 0.15, "lateral command did not reach the motors"
        assert phases[3]["yaw_change_rad"] > 0.25, "turn command did not reach the motors"
        assert np.hypot(phases[4]["body_dx_m"], phases[4]["body_dy_m"]) < 0.25, "Go2 did not stop"
    finally:
        if coordinator is not None:
            command.broadcast(None, Twist())
            coordinator.stop()
        unsubscribe()
        command.stop()
        odom.stop()


if __name__ == "__main__":
    main()
