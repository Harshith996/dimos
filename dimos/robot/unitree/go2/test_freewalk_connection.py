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

import numpy as np
import pytest

from dimos.control.go2_freewalk.policy import JOINT_NAMES, FreePolicy
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.go2.freewalk_connection import Go2FreewalkConnection


@pytest.fixture
def connection(mocker):
    clock = mocker.patch(
        "dimos.robot.unitree.go2.freewalk_connection.time.monotonic", return_value=10.0
    )
    module = Go2FreewalkConnection(joint_names=tuple(reversed(JOINT_NAMES)))
    policy = mocker.Mock(spec=FreePolicy)
    policy.default_pose = np.zeros(12)
    policy.kp = np.full(12, 40.0)
    policy.kd = np.ones(12)
    policy.act.return_value = np.arange(12.0)
    mocker.patch.object(module, "_policy", policy)
    publish = mocker.patch.object(module.motor_command, "publish")
    yield module, policy, clock, publish
    module.stop()


def feedback(module, stamp=1.0):
    module._receive_states(
        JointState(
            name=list(reversed(JOINT_NAMES)),
            position=list(reversed(range(12))),
            velocity=[0.0] * 12,
            ts=stamp,
        )
    )
    module._receive_imu(Imu(ts=stamp))


def test_waits_for_matching_feedback_and_maps_both_joint_orders(connection):
    module, policy, clock, _ = connection
    module._receive_states(
        JointState(name=list(JOINT_NAMES), position=[0.0] * 12, velocity=[0.0] * 12, ts=1)
    )
    module._receive_imu(Imu(ts=2))
    assert not module.arm()
    feedback(module, 3)
    assert module.arm()
    clock.return_value = 11.1
    feedback(module, 4)
    result = module._step(11.1)
    assert result.q == list(reversed(range(12)))
    obs, command = policy.act.call_args.args
    np.testing.assert_array_equal(obs.joint_position, np.arange(12))
    np.testing.assert_array_equal(obs.gravity, [0, 0, -1])
    np.testing.assert_array_equal(command, [0, 0, 0])


def test_repeated_old_feedback_cannot_keep_motor_control_alive(connection):
    module, _, clock, _ = connection
    feedback(module)
    assert module.arm()
    clock.return_value = 10.2
    feedback(module)  # same timestamp is not a fresh sensor sample
    assert module._step(10.2).kp == [0] * 12
    assert module.status()["fault"] == "coherent motor/IMU feedback timed out"
    feedback(module, 2)
    assert module._step(10.2).kp == [0] * 12
    assert module.arm()  # explicit recovery only


def test_expired_velocity_balances_at_zero_instead_of_disarming(connection):
    module, _, clock, _ = connection
    feedback(module)
    assert module.arm()
    clock.return_value = 11.1
    feedback(module, 2)
    module._receive_command(Twist(linear=(0.3, 0, 0)))
    module._step(11.1)
    assert module.status()["command"][0] == pytest.approx(0.05)
    clock.return_value = 12
    feedback(module, 3)
    module._step(12)
    assert module.status()["command"] == [0, 0, 0]
    assert module.status()["armed"]


def test_disarm_cannot_be_undone_by_velocity_or_auto_arm(connection):
    module, _, _, publish = connection
    feedback(module)
    assert module.arm()
    module.disarm()
    assert publish.call_args.args[0].kp == [0] * 12
    module._receive_command(Twist(linear=(0.2, 0, 0)))
    assert module._step(10).kp == [0] * 12
    assert not module.status()["armed"]


def test_nonfinite_command_latches_fault(connection):
    module, _, _, _ = connection
    feedback(module)
    assert module.arm()
    module._receive_command(Twist(linear=(float("nan"), 0, 0)))
    assert module._step(10).kp == [0] * 12
    assert module.status()["fault"] == "non-finite velocity command"
