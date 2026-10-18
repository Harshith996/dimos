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

"""Measure a Point-LIO blueprint; truth is read only by this probe."""

import argparse
from collections import deque
from dataclasses import replace
import json
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.get_all_blueprints import get_blueprint_by_name
from dimos.sim2.module import SimulationModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--robot", choices=("g1", "go2"), default="g1")
    args = parser.parse_args()
    global_config.update(simulation="mujoco", viewer="none", transport="zenoh")
    name = (
        "unitree-g1-groot-mid360-pointlio"
        if args.robot == "g1"
        else "unitree-go2-freewalk-pointlio"
    )
    blueprint = get_blueprint_by_name(name)
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
    estimates: deque[Odometry] = deque(maxlen=2000)
    imu_times: deque[float] = deque(maxlen=20000)
    raw_times: deque[float] = deque(maxlen=2000)
    raw_fields: dict[str, Any] = {}

    def raw_sample(cloud: PointCloud2) -> None:
        raw_times.append(cloud.ts)
        offsets, lines = cloud.offset_times_u32(), cloud.lines_u8()
        if offsets is None or lines is None or not len(offsets):
            raw_fields["error"] = "empty raw scan or timed fields lost in transport"
            return
        raw_fields.update(
            returns=len(offsets), max_offset_ns=int(offsets.max()), lines=np.unique(lines).tolist()
        )

    odom_stream = make_transport("odometry", Odometry)
    imu_stream = make_transport("imu_raw", Imu)
    raw_stream = make_transport("lidar_raw", PointCloud2)
    command = make_transport("cmd_vel", Twist)
    subscriptions = [
        odom_stream.subscribe(estimates.append),
        imu_stream.subscribe(lambda sample: imu_times.append(sample.ts)),
        raw_stream.subscribe(raw_sample),
    ]
    coordinator = None
    try:
        before = time.monotonic()
        coordinator = ModuleCoordinator.build(blueprint, parsed_config=parsed)
        startup = time.monotonic() - before
        sim = coordinator.get_instance(SimulationModule)
        lidar = coordinator.get_instance(f"{args.robot}_lidar")
        deadline = time.monotonic() + 20
        while not estimates:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Point-LIO produced no odometry: {lidar.sensor_status()}")
            time.sleep(0.05)
        command.start()
        start = time.monotonic()
        initial = sim.status()
        imu_start, raw_start, odom_start = len(imu_times), len(raw_times), len(estimates)
        errors = []
        angles = []
        matched = []
        alignment = None
        last_stamp = 0.0
        while time.monotonic() - start < args.seconds:
            elapsed = time.monotonic() - start
            speed = 0.2 if args.move and 2 < elapsed < args.seconds * 0.65 else 0.0
            yaw = 0.2 if args.move and args.seconds * 0.65 <= elapsed < args.seconds * 0.85 else 0.0
            command.broadcast(None, Twist(linear=(speed, 0, 0), angular=(0, 0, yaw)))
            truth = lidar.sensor_status().get("truth_pose")
            if truth is not None and truth["timestamp"] > last_stamp and estimates:
                estimate = min(tuple(estimates), key=lambda pose: abs(pose.ts - truth["timestamp"]))
                if abs(estimate.ts - truth["timestamp"]) < 0.025:
                    last_stamp = truth["timestamp"]
                    truth_p = np.asarray(truth["position"])
                    truth_r = Rotation.from_quat(truth["orientation"])
                    estimated_p = np.asarray(estimate.position.to_tuple())
                    estimated_r = Rotation.from_quat(estimate.orientation.to_tuple())
                    if alignment is None:
                        alignment = truth_r * estimated_r.inv()
                        translation = truth_p - alignment.apply(estimated_p)
                    errors.append(
                        float(np.linalg.norm(alignment.apply(estimated_p) + translation - truth_p))
                    )
                    angles.append(
                        float(np.rad2deg((truth_r.inv() * alignment * estimated_r).magnitude()))
                    )
                    matched.append(truth_p)
            time.sleep(0.05)
        elapsed = time.monotonic() - start
        status = sim.status()
        sensor = lidar.sensor_status()
        evidence = {
            "startup_seconds": startup,
            "elapsed_seconds": elapsed,
            "realtime_factor": (status["sim_time"] - initial["sim_time"]) / elapsed,
            "lidar_hz": (len(raw_times) - raw_start) / elapsed,
            "imu_hz": (len(imu_times) - imu_start) / elapsed,
            "odometry_hz": (len(estimates) - odom_start) / elapsed,
            "imu_max_gap_seconds": float(np.max(np.diff(tuple(imu_times)))),
            "matched_poses": len(errors),
            "position_error_max_m": max(errors) if errors else None,
            "position_error_rms_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "orientation_error_max_degrees": max(angles) if angles else None,
            "truth_displacement_m": float(np.linalg.norm(matched[-1] - matched[0]))
            if matched
            else None,
            "raw_fields": raw_fields,
            "sensor": sensor,
        }
        print(json.dumps(evidence, indent=2))
        assert sensor["error"] is None and len(errors) >= 10, "no sustained sensor/estimator output"
        assert "error" not in raw_fields, raw_fields
        assert evidence["imu_hz"] > 150 and evidence["lidar_hz"] > 7
        assert max(errors) < 0.25, "Point-LIO diverged from independently observed motion"
        if args.move:
            assert evidence["truth_displacement_m"] > 0.1, "robot did not move"
    finally:
        if coordinator is not None:
            command.broadcast(None, Twist())
            coordinator.stop()
        for unsubscribe in subscriptions:
            unsubscribe()
        for stream in (command, raw_stream, imu_stream, odom_stream):
            stream.stop()


if __name__ == "__main__":
    main()
