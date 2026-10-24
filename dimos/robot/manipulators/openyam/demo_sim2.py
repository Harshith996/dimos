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

"""Bounded headless YAM motor IO demo: python -m ...openyam.demo_sim2 --artifact DIR.

Runs native CPU MuJoCo and a real SimRobotClient on a unique loopback-only world.
The small joint motion demonstrates transport and reset, not a learned policy or
hardware calibration. The input artifact remains external to the package.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

from dimos.core.global_config import global_config
from dimos.robot.manipulators.openyam.simulation import (
    MODEL_JOINTS,
    PASSIVE_JOINT,
    YamSimAdapter,
    load_yam_artifact,
)
from dimos.sim2.client import SimRobotClient
from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.runtime import SimulationRuntime


def run_demo(
    artifact_directory: Path, *, steps: int = 100, expected_contract_sha256: str | None = None
) -> dict[str, Any]:
    """Own and close one bounded local world; call in an otherwise idle process."""
    if not 1 <= steps <= 1500:
        raise ValueError("steps must be between 1 and 1500 control ticks")
    artifact = load_yam_artifact(
        artifact_directory, expected_contract_sha256=expected_contract_sha256
    )
    config = artifact.contract
    sim_id = f"yam-demo-{uuid4().hex}"
    # sim2 transports resolve these existing process settings on construction.
    # Restore them after closing every owned stream; never join a configured robot.
    overrides: dict[str, Any] = {
        "robot_ip": None,
        "robot_ips": None,
        "zenoh_mode": "peer",
        "zenoh_connect": "",
        "zenoh_scouting": False,
        "zenoh_interface": "",
        "zenoh_multicast": False,
        "zenoh_gossip": False,
        "zenoh_domain": sim_id,
        "listen_host": "127.0.0.1",
    }
    previous = {name: getattr(global_config, name) for name in overrides}
    global_config.update(**overrides)
    wait = threading.Event()
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="dimos-yam-demo-") as directory:
            scene = Path(directory) / "scene.xml"
            scene.write_text(
                '<mujoco><option cone="elliptic" iterations="10" ls_iterations="20" impratio="10"/><worldbody><geom name="floor" type="plane" size="2 2 .1"/></worldbody></mujoco>'
            )
            world = WorldConfig(
                scene, {"yam": artifact.robot_instance()}, timestep=config.physics_dt
            )
            with SimulationRuntime(world, sim_id) as runtime:
                descriptor = runtime.robots["yam"].channel.descriptor
                with SimRobotClient(descriptor, writable=True, feedback_timeout=2.0) as client:
                    adapter = YamSimAdapter(client)
                    initial = adapter.model_positions(client.read())
                    home = np.asarray([config.initial_positions[name] for name in MODEL_JOINTS])
                    if not np.allclose(initial, home, rtol=0, atol=1e-12):
                        raise RuntimeError("initial public feedback differs from artifact home")
                    applied: set[int] = set()
                    max_motion = 0.0
                    deadline = time.monotonic()
                    for index in range(steps):
                        observed = client.read()
                        target = home.copy()
                        target[0] += 0.02 * math.sin(2 * math.pi * (index + 1) / steps)
                        adapter.write_model_targets(target, observed=observed)
                        for _ in range(config.control_decimation):
                            runtime.step()
                            deadline += config.physics_dt
                            wait.wait(max(0.0, deadline - time.monotonic()))
                        frame = client.read()
                        applied.add(frame.metadata.applied_action_sequence)
                        positions = adapter.model_positions(frame)
                        max_motion = max(max_motion, float(np.max(np.abs(positions - home))))
                    stale = client.read()
                    reset = runtime.reset()
                    passive = float(
                        runtime.data.qpos[runtime.model.joint(f"yam/{PASSIVE_JOINT}").qposadr[0]]
                    )
                    if not np.isclose(
                        passive, config.initial_positions[PASSIVE_JOINT], atol=1e-12, rtol=0
                    ):
                        raise RuntimeError("reset did not restore the passive finger")
                    # Wait for the published reset generation, not an arbitrary delay.
                    expires = time.monotonic() + 2.0
                    while client.read().metadata.episode_id != reset.generation:
                        if time.monotonic() >= expires:
                            raise TimeoutError("client did not receive reset generation")
                        wait.wait(0.001)
                    try:
                        adapter.write_model_targets(home, observed=stale)
                    except (ValueError, RuntimeError):
                        stale_rejected = True
                    else:
                        raise RuntimeError("pre-reset observation was accepted")
                    final = adapter.model_positions(client.read())
                    if not np.allclose(final, home, rtol=0, atol=1e-12):
                        raise RuntimeError("reset public feedback differs from artifact home")
                    if not (applied - {0}):
                        raise RuntimeError("no command was applied through SimRobotClient")
                    return {
                        "status": "completed",
                        "sim_id": sim_id,
                        "hardware": False,
                        "artifact_contract_sha256": artifact.contract_sha256,
                        "control_steps": steps,
                        "physics_steps": steps * config.control_decimation,
                        "simulation_seconds": steps * config.control_decimation * config.physics_dt,
                        "wall_seconds": time.monotonic() - started,
                        "applied_command_sequences": len(applied - {0}),
                        "max_native_coordinate_motion": max_motion,
                        "initial_model_positions": initial.tolist(),
                        "reset_model_positions": final.tolist(),
                        "reset_passive_finger": passive,
                        "stale_observation_rejected": stale_rejected,
                        "loopback_only": True,
                    }
    finally:
        global_config.update(**previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_demo(
        args.artifact, steps=args.steps, expected_contract_sha256=args.contract_sha256
    )
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as output:
            output.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
