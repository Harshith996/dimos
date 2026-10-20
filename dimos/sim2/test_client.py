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
import json
import multiprocessing
import subprocess
import sys
import time

import pytest

from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.sim2.client import SimRobotClient
from dimos.sim2.ipc.abi import ChannelDescriptor
from dimos.sim2.spec import ControlInterface, RobotInstance


def command():
    return MotorCommandArray(q=[0.3, 425], dq=[0.2, 0], kp=[2, 0], kd=[3, 0], tau=[0.4, 0])


def test_complete_command_and_coherent_feedback(worlds, clients):
    world = worlds()
    client = clients(world, writable=True)
    observed = client.read()
    assert observed.values["position"] == pytest.approx([0.1, 850])
    assert observed.values["imu_quaternion"] == pytest.approx([1, 0, 0, 0])
    sequence = client.write_motors(command(), observed=observed)

    world.step()

    binding = world.robots["left"]
    assert world.data.ctrl[binding.actuators] == pytest.approx([1.4, 0.02])
    feedback = client.read()
    assert feedback.metadata.applied_action_sequence == sequence
    assert feedback.metadata.physics_tick == 1
    assert feedback.metadata.sim_time == pytest.approx(world.config.timestep)
    assert observed.metadata.physics_tick == 0
    assert observed.values["position"] == pytest.approx([0.1, 850])


def test_two_robots_use_independent_channels(world_config, worlds, clients):
    robot = world_config.robots["left"]
    config = replace(
        world_config,
        robots={
            "left": robot,
            "right": RobotInstance(robot.config, xyz=(0, 2, 0.3)),
        },
    )
    world = worlds(config=config)
    left, right = clients(world, "left", writable=True), clients(world, "right", writable=True)
    left.write_motors(command(), observed=left.read())
    world.step()

    assert world.data.ctrl[world.robots["left"].actuators[0]] == pytest.approx(1.4)
    assert world.data.ctrl[world.robots["right"].actuators[0]] == pytest.approx(0)
    assert right.read().values["root_position"] == pytest.approx([0, 2, 0.3])


def test_position_device_moves_and_preserves_gripper_units(worlds, clients):
    world = worlds(control=ControlInterface.MANIPULATOR)
    client = clients(world, writable=True)
    for _ in range(300):
        client.write_positions([0.25, 0], observed=client.read())
        world.step()

    assert client.read().values["position"] == pytest.approx([0.25, 0], abs=0.03)


def test_read_only_clients_and_single_command_owner(worlds, clients):
    world = worlds()
    owner = clients(world, writable=True)
    reader = clients(world)
    with pytest.raises(PermissionError, match="read-only"):
        reader.write_motors(command(), observed=reader.read())
    with pytest.raises(RuntimeError, match="already has a command owner"):
        clients(world, writable=True)
    owner.close()
    replacement = clients(world, writable=True)
    assert replacement.write_motors(command(), observed=replacement.read()) > 0


def test_old_observation_cannot_command_after_reset(worlds, clients):
    world = worlds()
    client = clients(world, writable=True)
    old = client.read()
    client.write_motors(command(), observed=old)

    world.reset()

    with pytest.raises(RuntimeError, match="reset"):
        client.write_motors(command(), observed=old)
    world.step()
    assert world.data.ctrl[world.robots["left"].actuators] == pytest.approx([0, 0])
    fresh = client.read()
    assert fresh.metadata.episode_id == old.metadata.episode_id + 1
    assert fresh.metadata.applied_action_sequence == 0
    assert client.write_motors(command(), observed=fresh) > 0


def test_reset_between_client_validation_and_publication_rejects_late_command(
    worlds, clients, mocker
):
    world = worlds()
    client = clients(world, writable=True)
    observed = client.read()
    publish = client._channel.publish_action

    def reset_then_publish(values, metadata):
        world.reset()
        return publish(values, metadata)

    mocker.patch.object(client._channel, "publish_action", side_effect=reset_then_publish)
    client.write_motors(command(), observed=observed)
    world.step()

    assert world.data.ctrl[world.robots["left"].actuators] == pytest.approx([0, 0])
    assert client.read().metadata.applied_action_sequence == 0


def test_stale_feedback_is_not_accepted(worlds, clients, mocker):
    world = worlds()
    client = clients(world, writable=True)
    observed = client.read()
    mocker.patch("dimos.sim2.client.time.monotonic", return_value=time.monotonic() + 1)

    with pytest.raises(TimeoutError, match="stale"):
        client.read()
    with pytest.raises(TimeoutError, match="stale"):
        client.write_motors(command(), observed=observed)


def test_foreign_observation_is_not_a_command_basis(worlds, clients):
    world = worlds()
    writer, reader = clients(world, writable=True), clients(world)
    with pytest.raises(ValueError, match="this client's latest observation"):
        writer.write_motors(command(), observed=reader.read())


@pytest.mark.parametrize("field,value", [("q", [0]), ("tau", [float("nan"), 0]), ("kp", [-1, 0])])
def test_invalid_command_does_not_replace_held_action(worlds, clients, field, value):
    world = worlds()
    client = clients(world, writable=True)
    observed = client.read()
    expected = client.write_motors(command(), observed=observed)
    invalid = command()
    setattr(invalid, field, value)

    with pytest.raises(ValueError):
        client.write_motors(invalid, observed=observed)

    world.step()
    assert client.read().metadata.applied_action_sequence == expected


def test_mismatched_joint_order_is_rejected_at_attach(worlds):
    world = worlds()
    descriptor = world.robots["left"].channel.descriptor
    wrong = replace(descriptor, joint_names=tuple(reversed(descriptor.joint_names)))
    with pytest.raises(ValueError, match="descriptor mismatch"):
        with SimRobotClient(wrong):
            pytest.fail("mismatched joint order attached")


def test_faulted_world_cannot_be_reactivated_by_spawn(worlds, clients):
    world = worlds()
    client = clients(world)
    world.snapshots.set_lifecycle("faulted")
    world.robots["left"].channel.set_lifecycle("faulted")
    with pytest.raises(RuntimeError, match="faulted"):
        client.read()
    with pytest.raises(RuntimeError, match="faulted"):
        world.set_spawn("left", (1, 2, 3), (0, 0, 0))
    assert world.snapshots.lifecycle == "faulted"


def test_close_invalidates_clients_and_restart_has_a_new_generation(worlds, clients):
    world = worlds(sim_id="restart")
    client = clients(world)
    old = world.robots["left"].channel.descriptor
    world.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.read()
    with pytest.raises(FileNotFoundError):
        with SimRobotClient(old):
            pytest.fail("closed world reattached")
    replacement = worlds(sim_id="restart")
    new = replacement.robots["left"].channel.descriptor
    assert new.generation != old.generation
    assert new.shm_name != old.shm_name
    assert clients(replacement).read().metadata.physics_tick == 0


def _command_in_spawned_process(descriptor, pipe):
    pipe.send("ready")
    pipe.recv()
    with SimRobotClient(ChannelDescriptor.from_dict(descriptor), writable=True) as client:
        observation = client.read()
        sequence = client.write_motors(command(), observed=observation)
        pipe.send((sequence, observation.values["position"].tolist()))
    pipe.close()


def test_client_controls_world_across_spawned_process(worlds):
    world = worlds()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_command_in_spawned_process,
        args=(world.robots["left"].channel.descriptor.to_dict(), child),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(15), "child did not start"
        assert parent.recv() == "ready"
        world.step()
        parent.send("sample now")
        assert parent.poll(15), "child did not return a command"
        sequence, position = parent.recv()
        world.step()
        assert position == pytest.approx([0.1, 850])
        assert world.robots["left"].applied_action_sequence == sequence
        assert world.data.ctrl[world.robots["left"].actuators] == pytest.approx([1.4, 0.02])
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        process.close()


def test_independent_reader_exit_does_not_unlink_the_world(worlds, clients):
    world = worlds()
    descriptor = world.robots["left"].channel.descriptor
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from dimos.sim2.client import SimRobotClient; "
            "from dimos.sim2.ipc.abi import ChannelDescriptor; "
            "client=SimRobotClient(ChannelDescriptor.from_dict(json.loads(sys.argv[1])), "
            "feedback_timeout=10); client.connect(); "
            "print(client.read().metadata.episode_id); client.close()",
            json.dumps(descriptor.to_dict()),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert result.stdout.strip() == str(world.episode)
    assert "resource_tracker" not in result.stderr
    world.step()
    assert clients(world).read().metadata.physics_tick == 1
