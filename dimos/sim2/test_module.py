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

"""A world module drives a robot through ordinary ports, with nothing sim-specific on the wire."""

from contextlib import ExitStack
from dataclasses import replace
import threading
import time

import pytest

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.blueprint import simulation
from dimos.sim2.module import SimulationModule, robot_ports


class _World(SimulationModule):
    left_motor_command: In[MotorCommandArray]
    left_motor_states: Out[JointState]
    left_imu: Out[Imu]
    left_odom: Out[PoseStamped]
    left_tf: Out[TFMessage]


class _Driver(Module):
    """Stands in for the coordinator's transport adapter: standard messages both ways."""

    left_motor_states: In[JointState]
    left_imu: In[Imu]
    left_odom: In[PoseStamped]
    left_motor_command: Out[MotorCommandArray]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._joints = None
        self._imu = None
        self._odom = None

    @rpc
    def start(self):
        super().start()
        self.left_motor_states.subscribe(self._on_joints)
        self.left_imu.subscribe(self._on_imu)
        self.left_odom.subscribe(self._on_odom)

    def _on_joints(self, msg):
        with self._lock:
            self._joints = msg

    def _on_imu(self, msg):
        with self._lock:
            self._imu = msg

    def _on_odom(self, msg):
        with self._lock:
            self._odom = msg

    @rpc
    def observed(self):
        with self._lock:
            if self._joints is None or self._imu is None or self._odom is None:
                return None
            return {
                "names": list(self._joints.name),
                "position": list(self._joints.position),
                "frame": self._joints.frame_id,
                "imu_frame": self._imu.frame_id,
                "odom_frame": self._odom.frame_id,
                "same_tick": self._joints.ts == self._imu.ts == self._odom.ts,
            }

    @rpc
    def command(self, q, kp, kd):
        self.left_motor_command.publish(
            MotorCommandArray(q=q, dq=[0.0] * len(q), kp=kp, kd=kd, tau=[0.0] * len(q))
        )


def test_robot_ports_are_matched_by_name_and_unknown_robots_are_an_error():
    ports = robot_ports(_World, ["left"])
    assert ports == {
        "left": {
            "motor_command": "left_motor_command",
            "motor_states": "left_motor_states",
            "imu": "left_imu",
            "odom": "left_odom",
            "tf": "left_tf",
        }
    }
    assert robot_ports(SimulationModule, ["left"]) == {"left": {}}
    with pytest.raises(ValueError, match="names robot 'left'"):
        robot_ports(_World, ["right"])


def _wait(predicate, timeout=15.0, every=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(every)
    raise AssertionError("condition not met in time")


def test_world_publishes_and_consumes_standard_messages_over_streams(world_config):
    instance = world_config.robots["left"]
    robot = replace(instance.config, sensors=instance.config.sensors[:1])
    world = replace(world_config, robots={"left": replace(instance, config=robot)})
    blueprint = autoconnect(simulation(world=world, module=_World), _Driver.blueprint())
    with ExitStack() as stack:
        app = ModuleCoordinator.build(
            blueprint.global_config(n_workers=1, viewer="none", transport="zenoh")
        )
        stack.callback(app.stop)
        sim = app.get_instance(_World)
        driver = app.get_instance(_Driver)

        observed = _wait(driver.observed)
        assert observed["names"] == ["shoulder", "gripper"]
        assert observed["position"] == pytest.approx([0.1, 850], abs=1e-3)
        assert observed["frame"] == observed["imu_frame"] == "left/base"
        assert observed["odom_frame"] == "world"
        assert observed["same_tick"]

        driver.command([0.5, 850.0], [10.0, 0.0], [1.0, 0.0])
        _wait(lambda: driver.observed()["position"][0] > 0.3)
        status = sim.status()
        assert status["running"]
        assert status["rejected_commands"] == {}

        driver.command([0.5], [10.0], [1.0])
        _wait(lambda: sim.status()["rejected_commands"].get("left", 0) == 1)
        episode = sim.status()["episode"]
        sim.reset()
        assert sim.status()["episode"] == episode + 1
