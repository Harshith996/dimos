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
import time

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.sim2.scene_types import SceneUpdate


def test_expired_motor_command_stops_using_feedforward(worlds, clients, mocker):
    world = worlds()
    client = clients(world, writable=True)
    client.write_motors(
        MotorCommandArray(q=[0.1, 850], kp=[0, 0], kd=[0, 0], tau=[3, 0]),
        observed=client.read(),
    )
    world.step()
    binding = world.robots["left"]
    assert world.data.ctrl[binding.actuators[0]] == pytest.approx(3)
    velocity = world.data.qvel[binding.dofs[0]]
    mocker.patch("dimos.sim2.runtime.time.monotonic", return_value=time.monotonic() + 1)

    world.step()

    assert binding.applied_action_sequence == 0
    assert world.data.ctrl[binding.actuators[0]] == pytest.approx(-velocity)


def test_transient_channel_contention_does_not_kill_physics(worlds, clients, mocker):
    world = worlds()
    client = clients(world, writable=True)
    client.write_motors(
        MotorCommandArray(q=[0.1, 850], kp=[0, 0], kd=[0, 0], tau=[2, 0]),
        observed=client.read(),
    )
    world.step()
    channel = world.robots["left"].channel
    mocker.patch.object(channel, "read_action", side_effect=BlockingIOError("busy"))

    world.step()

    assert world.tick == 2
    assert channel.lifecycle == "ready"
    assert world.data.ctrl[world.robots["left"].actuators[0]] == pytest.approx(2)


@pytest.mark.parametrize("disabled", [False, True])
def test_scene_edit_preserves_robot_pose_and_motor_enable(worlds, clients, disabled):
    world = worlds()
    client = clients(world, writable=True)
    for _ in range(200):
        client.write_motors(
            MotorCommandArray(q=[0.4, 850], kp=[10, 0], kd=[1, 0]),
            observed=client.read(),
        )
        world.step()
    position = client.read().values["position"].copy()
    assert position[0] == pytest.approx(0.4, abs=0.03)
    if disabled:
        client.disable(observed=client.read())
        world.step()
        position = client.read().values["position"].copy()

    state = world.set_scene_state(SceneUpdate(poses={"box": Pose(2, 0, 0.3)}))
    assert client.read().values["position"] == pytest.approx(position)
    world.step()

    assert state.entities["box"].pose.position.to_tuple() == pytest.approx((2, 0, 0.3))
    assert client.read().values["position"][0] == pytest.approx(position[0], abs=0.002)
    assert world.robots["left"].idle_position[0] == pytest.approx(position[0])
    assert world.robots["left"].enabled is not disabled
    assert world.robots["left"].applied_action_sequence == 0


def test_disabled_effort_outputs_do_not_reenable_when_command_expires(worlds, clients, mocker):
    world = worlds()
    client = clients(world, writable=True)
    client.disable(observed=client.read())
    world.step()
    mocker.patch("dimos.sim2.runtime.time.monotonic", return_value=time.monotonic() + 1)

    world.step()

    assert not world.robots["left"].enabled
    assert world.data.ctrl[world.robots["left"].actuators[0]] == 0


def test_scene_update_is_atomic_and_reset_restores_authored_state(worlds):
    world = worlds()
    before = world.scene_state()
    with pytest.raises(ValueError, match="outside joint limits"):
        world.set_scene_state(SceneUpdate(poses={"box": Pose(2, 0, 0.3)}, joints={"door": 5}))
    assert world.scene_state().entities["box"] == before.entities["box"]

    changed = world.set_scene_state(
        SceneUpdate(poses={"box": Pose(2, 0, 0.3)}, joints={"door": 0.5})
    )
    assert changed.joints["door"] == pytest.approx(0.5)
    reset = world.reset()
    assert reset.entities["box"] == before.entities["box"]
    assert reset.joints["door"] == 0
    assert reset.generation == changed.generation + 1


def test_reset_publishes_snapshot_before_next_step(worlds):
    world = worlds()
    world.step()
    previous = world.snapshots.read_observation()

    world.reset()

    current = world.snapshots.read_observation()
    assert current.metadata.episode_id == previous.metadata.episode_id + 1
    assert current.metadata.sequence > previous.metadata.sequence
    assert current.metadata.sim_time == 0


def test_pause_does_not_advance_physics(worlds):
    world = worlds()
    before = world.data.qpos.copy()
    world.set_paused(True)
    world.step()
    np.testing.assert_array_equal(world.data.qpos, before)
    assert world.tick == 0
    world.set_paused(False)
    world.step()
    assert world.tick == 1


def test_robot_cannot_replace_world_snapshot_channel(worlds):
    config = worlds().config
    with pytest.raises(ValueError, match="'world' is reserved"):
        replace(config, robots={"world": config.robots["left"]})


@pytest.mark.parametrize("disabled", [False, True])
def test_resume_preserves_pose_and_motor_enable_but_invalidates_old_work(
    worlds, clients, mocker, disabled
):
    world = worlds()
    client = clients(world, writable=True)
    if disabled:
        client.disable(observed=client.read())
    else:
        client.write_motors(
            MotorCommandArray(q=[0.4, 850], kp=[10, 0], kd=[1, 0]), observed=client.read()
        )
    world.step()
    old_frame = client.read()
    pose = world.data.qpos.copy()
    sim_time = float(world.data.time)
    previous = world.snapshots.read_observation()
    world.set_paused(True)
    mocker.patch("dimos.sim2.runtime.time.time", return_value=previous.values["wall_time"][0] + 10)

    world.set_paused(False)

    current = world.snapshots.read_observation()
    assert current.metadata.episode_id == previous.metadata.episode_id + 1
    assert current.metadata.sim_time == sim_time
    assert current.values["wall_time"][0] == previous.values["wall_time"][0] + 10
    np.testing.assert_array_equal(world.data.qpos, pose)
    assert world.robots["left"].enabled is not disabled
    assert not world.robots["left"].command_active
    with pytest.raises(RuntimeError, match="reset"):
        client.disable(observed=old_frame)
    world.set_paused(False)
    assert world.episode == current.metadata.episode_id
    world.step()
    assert world.robots["left"].applied_action_sequence == 0
    assert world.robots["left"].enabled is not disabled


def test_pause_cannot_restore_faulted_world(worlds):
    world = worlds()
    world.snapshots.set_lifecycle("faulted")
    with pytest.raises(RuntimeError, match="faulted"):
        world.set_paused(False)
    assert world.snapshots.lifecycle == "faulted"
