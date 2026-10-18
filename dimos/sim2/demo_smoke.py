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

"""Exercise the actual migrated blueprint with forkserver workers and live sensors."""

import argparse
from dataclasses import replace
import json
from queue import Queue
import socket
import time

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.protocol.service.zenohservice import ZenohService
from dimos.robot.get_all_blueprints import get_blueprint_by_name
from dimos.sim2.module import SimulationModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", choices=["g1", "xarm"])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--mid360", action="store_true", help="Use the G1 rolling Mid360 preset")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Include the Rerun bridge and viewer")
    parser.add_argument("--transport", choices=["lcm", "zenoh"], default="zenoh")
    parser.add_argument("--local-router", action="store_true")
    parser.add_argument(
        "--move", action="store_true", help="Check motion through ordinary command streams"
    )
    args = parser.parse_args()
    if args.mid360 and args.robot != "g1":
        parser.error("--mid360 requires g1")
    global_config.update(
        simulation="mujoco", viewer="rerun" if args.rerun else "none", transport=args.transport
    )
    router = None
    if args.local_router:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{reservation.getsockname()[1]}"
        router = ZenohService(mode="router", listen=[endpoint], multicast=False)
        router.start()
        global_config.update(zenoh_mode="client", zenoh_connect=endpoint, zenoh_multicast=False)
    name = "unitree-g1-groot-wbc" if args.robot == "g1" else "xarm7-planner-coordinator"
    if args.mid360:
        name = "unitree-g1-groot-mid360"
    blueprint = get_blueprint_by_name(name)
    atoms = tuple(
        replace(atom, kwargs={**atom.kwargs, "viewer": args.viewer})
        if atom.module is SimulationModule
        else atom
        for atom in blueprint.blueprints
    )
    blueprint = replace(blueprint, blueprints=atoms)
    before = time.monotonic()
    try:
        parsed = BlueprintConfigParser(blueprint).parse(
            global_overrides=global_config.model_dump(mode="python")
        )
        coordinator = ModuleCoordinator.build(blueprint, parsed_config=parsed)
    except BaseException:
        if router is not None:
            router.stop()
        raise
    startup = time.monotonic() - before
    command = None
    try:
        sim = coordinator.get_instance(SimulationModule)
        camera_name = "g1_camera" if args.robot == "g1" else "arm_wrist_camera"
        camera = coordinator.get_instance(camera_name)
        rgb = camera.peek_stream("color_image", 5.0)
        depth = camera.peek_stream("depth_image", 5.0)
        assert rgb is not None, "no RGB published"
        assert depth is not None, "no depth published"
        assert camera.sensor_status()["error"] is None
        if args.robot == "g1":
            lidar = coordinator.get_instance("g1_lidar")
            assert lidar.peek_stream("pointcloud", 5.0) is not None, "no lidar published"
            assert lidar.sensor_status()["error"] is None
            lidar_initial = lidar.sensor_status()
            if args.mid360:
                # Inspect the typed wire format, not the RPC's lossy cloud pickle.
                raw_stream = make_transport("raw_pointcloud", PointCloud2)
                samples: Queue[PointCloud2] = Queue()
                unsubscribe = raw_stream.subscribe(samples.put)
                try:
                    raw = samples.get(timeout=5.0)
                    assert raw.offset_times_u32() is not None
                    assert raw.lines_u8() is not None
                finally:
                    unsubscribe()
                    raw_stream.stop()
        started = time.monotonic()
        initial = sim.status()
        if args.move:
            command = (
                make_transport("cmd_vel", Twist)
                if args.robot == "g1"
                else make_transport("joint_command", JointState)
            )
            command.start()
        arm_target = None
        if args.move and args.robot == "xarm":
            connection = coordinator.get_instance("arm_connection")
            joints = connection.peek_stream("joint_states", 5.0)
            assert joints is not None, "no arm joint feedback"
            target = list(joints.position)
            target[0] += 0.25
            arm_target = JointState(name=list(joints.name), position=target)
        deadline = started + args.seconds + 5
        while time.monotonic() < deadline:
            state = sim.status()
            assert state["error"] is None, state
            if command is not None:
                if args.robot == "g1":
                    speed = (
                        0.2 if state["sim_time"] - initial["sim_time"] < args.seconds * 0.7 else 0.0
                    )
                    command.broadcast(None, Twist(linear=(speed, 0, 0)))
                else:
                    command.broadcast(None, arm_target)
            if state["sim_time"] - initial["sim_time"] >= args.seconds:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("physics did not keep up with real time")
        elapsed = time.monotonic() - started
        if args.robot == "g1":
            assert state["robots"]["g1"][2] > 0.5, state
            if args.move:
                assert state["robots"]["g1"][0] - initial["robots"]["g1"][0] > 0.1, (
                    "G1 did not walk forward"
                )
        elif args.move:
            joints = connection.peek_stream("joint_states", 5.0)
            assert joints is not None and arm_target is not None
            assert abs(joints.position[0] - arm_target.position[0]) < 0.03, (
                "arm did not reach commanded joint angle"
            )
        print(
            json.dumps(
                {
                    "startup_seconds": startup,
                    "realtime_factor": (state["sim_time"] - initial["sim_time"]) / elapsed,
                    "rgb": rgb.data.shape,
                    "depth": depth.data.shape,
                    "status": state,
                    "lidar": lidar.sensor_status() if args.robot == "g1" else None,
                    "lidar_hz": (
                        (lidar.sensor_status()["scans"] - lidar_initial["scans"]) / elapsed
                        if args.mid360
                        else None
                    ),
                },
                indent=2,
            )
        )
    finally:
        if command is not None:
            command.stop()
        stopping = time.monotonic()
        try:
            coordinator.stop()
        finally:
            if router is not None:
                router.stop()
        print(json.dumps({"shutdown_seconds": time.monotonic() - stopping}))


if __name__ == "__main__":
    main()
