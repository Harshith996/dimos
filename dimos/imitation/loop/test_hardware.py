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

from dataclasses import replace

import pytest

from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import MotorCommand, MotorState, WholeBodyAdapter
from dimos.imitation.loop.hardware import PhysicalYamDevice


@pytest.fixture
def device(mocker):
    adapter = mocker.create_autospec(WholeBodyAdapter, instance=True)
    adapter.activate = mocker.Mock(return_value=True)
    adapter.read_motor_states.return_value = [MotorState(q=0.1)] * 6 + [MotorState(q=0.8)]
    adapter.write_motor_commands.return_value = True
    value = PhysicalYamDevice(
        adapter,
        JointLimits(
            position_lower=[-2.0] * 6 + [0.0],
            position_upper=[2.0] * 6 + [1.0],
            velocity_max=[0.5] * 7,
        ),
    )
    yield value
    value.disconnect()


def test_connection_does_not_enable_or_produce_training_feedback(device):
    device.connect()
    device.read_motor_states()
    device.adapter.activate.assert_not_called()
    assert device.snapshot() is None
    assert not device.write_motor_commands([MotorCommand(q=0.2)] * 7)


def test_driver_opening_is_converted_once_and_only_accepted_writes_are_recorded(device):
    assert device.activate()
    device.transition("human")
    feedback = device.read_motor_states()
    assert feedback[-1].q == pytest.approx(0.2)
    commands = [MotorCommand(q=0.1, kp=80, kd=5)] * 6 + [MotorCommand(q=0.7)]
    assert device.write_motor_commands(commands)
    dispatched = device.adapter.write_motor_commands.call_args.args[0]
    assert dispatched[:-1] == commands[:-1]
    assert dispatched[-1].q == pytest.approx(0.3)
    device.read_motor_states()
    accepted = device.snapshot()[2]
    assert accepted.positions[-1] == 0.7
    assert accepted.authority == "human"
    assert accepted.generation == device.observed_generation() == 1
    device.adapter.write_motor_commands.return_value = False
    assert not device.write_motor_commands([replace(command, q=0.4) for command in commands])
    device.read_motor_states()
    assert device.snapshot()[2] == accepted


def test_authority_change_invalidates_a_tick_that_read_before_takeover(device):
    device.activate()
    device.read_motor_states()
    device.transition("correction")
    assert not device.write_motor_commands([MotorCommand(q=0.3)] * 7)
    device.adapter.write_motor_commands.assert_not_called()
    device.read_motor_states()
    assert device.write_motor_commands([MotorCommand(q=0.3)] * 7)


def test_failed_activation_never_unlocks_commands(device):
    device.adapter.activate.return_value = False
    assert not device.activate()
    device.read_motor_states()
    assert not device.write_motor_commands([MotorCommand(q=0.3)] * 7)
    assert not device.enabled
