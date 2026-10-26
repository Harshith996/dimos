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

from dimos.robot.manipulators.dual_openyam.molmoact2_sim import (
    MolmoAct2SceneResetController,
    MolmoAct2YamSimConfig,
    MolmoAct2YamSimModule,
    dimos_positions_to_maniskill_action,
    maniskill_qpos_to_dimos,
    maniskill_qvel_to_dimos,
)


def test_simulator_defaults_to_easy_cube_task() -> None:
    config = MolmoAct2YamSimConfig()

    assert config.env_id == "BimanualYAMEasyPickCubes-v1"
    assert config.task_module == "dimos.robot.manipulators.dual_openyam.maniskill_easy_task"


def test_full_scene_reset_discards_active_recording_before_reset(mocker) -> None:
    controller = MolmoAct2SceneResetController()
    episode = mocker.Mock()
    simulator = mocker.Mock()
    calls = []
    episode.get_status.return_value.state = "recording"
    episode.command.side_effect = lambda event: calls.append(("episode", event))
    simulator.reset_scene.side_effect = lambda: calls.append(("simulator", "reset")) or "reset"
    controller._episode_control = episode
    controller._simulator = simulator

    try:
        controller._perform_reset()
    finally:
        controller.stop()

    assert calls == [("episode", "discard"), ("simulator", "reset")]


def test_full_scene_reset_does_not_discard_when_idle(mocker) -> None:
    controller = MolmoAct2SceneResetController()
    episode = mocker.Mock()
    simulator = mocker.Mock()
    episode.get_status.return_value.state = "idle"
    simulator.reset_scene.return_value = "reset"
    controller._episode_control = episode
    controller._simulator = simulator

    try:
        controller._perform_reset()
    finally:
        controller.stop()

    episode.command.assert_not_called()
    simulator.reset_scene.assert_called_once_with()


def test_simulator_reset_rpc_queues_reset_on_simulation_thread(mocker) -> None:
    simulator = MolmoAct2YamSimModule()
    simulator._thread = mocker.Mock()
    simulator._thread.is_alive.return_value = True
    wait = mocker.patch.object(simulator._reset_complete, "wait", return_value=True)

    try:
        result = simulator.reset_scene()
    finally:
        simulator._thread = None
        simulator.stop()

    assert result == "ManiSkill scene reset to its initial state"
    assert simulator._reset_requested.is_set()
    wait.assert_called_once_with(timeout=simulator.config.reset_timeout_s)


def test_dimos_targets_map_to_maniskill_controller_order() -> None:
    positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0.25, 0.75]

    action = dimos_positions_to_maniskill_action(positions)

    assert action.tolist() == pytest.approx([1, 2, 3, 4, 5, -6, 0.5, 7, 8, 9, 10, 11, -12, -0.5])


def test_maniskill_state_maps_to_dimos_joint_order() -> None:
    qpos = [
        1,
        7,
        2,
        8,
        3,
        9,
        4,
        10,
        5,
        11,
        6,
        12,
        -0.02375,
        -0.02375,
        -0.0475,
        -0.0475,
    ]

    positions = maniskill_qpos_to_dimos(qpos)

    assert positions.tolist() == pytest.approx([1, 2, 3, 4, 5, -6, 7, 8, 9, 10, 11, -12, 0.5, 1.0])


def test_maniskill_velocity_preserves_signed_gripper_rate() -> None:
    qvel = np.zeros(16, dtype=np.float32)
    qvel[10] = 0.25
    qvel[11] = -0.5
    qvel[12] = -0.0475
    qvel[14] = 0.02375

    velocities = maniskill_qvel_to_dimos(qvel)

    assert velocities[[5, 11, 12, 13]].tolist() == pytest.approx([-0.25, 0.5, 1.0, -0.5])


@pytest.mark.parametrize(
    ("converter", "value", "message"),
    [
        (dimos_positions_to_maniskill_action, [0.0] * 13, "Expected 14"),
        (maniskill_qpos_to_dimos, [0.0] * 15, "Expected 16"),
        (maniskill_qvel_to_dimos, [0.0] * 17, "Expected 16"),
        (dimos_positions_to_maniskill_action, [*[0.0] * 13, np.nan], "must all be finite"),
    ],
)
def test_joint_mapping_rejects_invalid_input(converter, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        converter(value)
