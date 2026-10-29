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

"""The coordinator's sim2 adapter reads coherent samples and forwards complete commands."""

from contextlib import ExitStack
import time

import pytest

from dimos.hardware.whole_body.registry import whole_body_adapter_registry
from dimos.hardware.whole_body.spec import MotorCommand, WholeBodyAdapter
from dimos.sim2.control.adapters import Sim2WholeBodyAdapter
from dimos.sim2.world.spec import SimulationSpec


@pytest.fixture
def adapters(mocker):
    with ExitStack() as stack:

        def create(world, robot_id="left", **kwargs):
            source = mocker.Mock(spec=SimulationSpec)
            source.describe_robot.side_effect = lambda name: world.robots[name].channel.descriptor
            adapter = Sim2WholeBodyAdapter(simulation=source, hardware_id=robot_id, **kwargs)
            stack.callback(adapter.disconnect)
            return adapter, source

        yield create


def test_registry_creates_the_sim2_adapter_by_name():
    assert "sim2" in whole_body_adapter_registry.available()
    with pytest.raises(ValueError, match="SimulationModule"):
        whole_body_adapter_registry.create("sim2", dof=2, hardware_id="left", address=None)


def test_adapter_reads_one_coherent_sample_per_tick(worlds, adapters):
    world = worlds()
    adapter, _ = adapters(world, dof=2, joint_names=("shoulder", "gripper"))
    assert isinstance(adapter, WholeBodyAdapter)
    assert not adapter.has_motor_states()
    assert adapter.read_motor_states() == []

    assert adapter.connect()

    assert adapter.is_connected()
    assert adapter.has_motor_states()
    states = adapter.read_motor_states()
    assert [s.q for s in states] == pytest.approx([0.1, 850])
    imu = adapter.read_imu()
    assert imu.quaternion == pytest.approx((1, 0, 0, 0))  # (w, x, y, z), as on hardware
    assert imu.gyroscope == pytest.approx((0, 0, 0))
    assert adapter.get_limits() is None


def test_commands_name_the_last_sample_and_map_stop_sentinels(worlds, adapters):
    world = worlds()
    adapter, _ = adapters(world)
    adapter.connect()
    assert not adapter.write_motor_commands([MotorCommand(), MotorCommand()])  # nothing read yet
    adapter.read_motor_states()
    assert adapter.write_motor_commands(
        [MotorCommand(q=0.3, dq=0.2, kp=2, kd=3, tau=0.4), MotorCommand(q=425, dq=0)]
    )
    world.step()
    binding = world.robots["left"]
    assert world.data.ctrl[binding.actuators] == pytest.approx([1.4, 0.02])

    # POS_STOP/VEL_STOP mean "no target": the gains that would act on them are dropped.
    adapter.read_motor_states()
    assert adapter.write_motor_commands([MotorCommand(kp=5, kd=7, tau=0.1), MotorCommand(q=425)])
    world.step()
    assert world.data.ctrl[binding.actuators[0]] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="expected 2"):
        adapter.write_motor_commands([MotorCommand()])


def test_connect_waits_for_the_world_to_describe_the_robot(worlds, adapters, mocker):
    world = worlds()
    adapter, source = adapters(world, connect_timeout=5)
    descriptor = world.robots["left"].channel.descriptor
    source.describe_robot.side_effect = [
        RuntimeError("SimulationModule.build() has not completed"),
        descriptor,
    ]
    sleep = mocker.patch("dimos.sim2.client.time.sleep")

    assert adapter.connect()

    assert source.describe_robot.call_count == 2
    sleep.assert_called_once()


def test_connect_gives_up_at_the_deadline(worlds, adapters, mocker):
    world = worlds()
    adapter, source = adapters(world, connect_timeout=0)
    source.describe_robot.side_effect = RuntimeError("SimulationModule.build() has not completed")

    assert not adapter.connect()
    assert not adapter.is_connected()


def test_wrong_dof_is_rejected_at_connect(worlds, adapters):
    adapter, _ = adapters(worlds(), dof=3)
    with pytest.raises(ValueError, match="expected 3"):
        adapter.connect()


def test_stale_feedback_means_no_motor_states(worlds, adapters, mocker):
    world = worlds()
    adapter, _ = adapters(world)
    adapter.connect()
    assert adapter.has_motor_states()
    mocker.patch("dimos.sim2.client.time.monotonic", return_value=time.monotonic() + 1)

    assert not adapter.has_motor_states()
    assert all(s.q == 0 for s in adapter.read_motor_states())
    assert adapter.read_imu().quaternion == (1, 0, 0, 0)
    assert not adapter.write_motor_commands([MotorCommand(), MotorCommand()])
