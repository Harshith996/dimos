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

from concurrent.futures import Future

import numpy as np
import pytest

from dimos.imitation.loop.device import Observation, OpenYamSimulation
from dimos.imitation.loop.recording import read_episode
from dimos.imitation.loop.session import Authority, LearningSession
from dimos.robot.manipulators.openyam.device import AcceptedTarget


@pytest.fixture
def session(tmp_path):
    value = LearningSession(
        tmp_path, ["joint", "gripper"], (-1.0, 0.0), (1.0, 1.0), (0.0, 1.0), image_shape=(2, 3, 3)
    )
    try:
        yield value
    finally:
        value.close()


def test_takeover_discards_queued_and_inflight_actions_even_after_resume():
    authority = Authority(np.array([0.0, 1.0]))
    authority.switch("policy", np.array([0.0, 1.0]))
    requested = authority.generation
    assert authority.accept(requested, [[0.2, 0.8], [0.3, 0.7]])

    authority.switch("correction", np.array([0.1, 0.9]))
    assert not authority.actions
    assert not authority.accept(requested, [[0.4, 0.6]])
    np.testing.assert_array_equal(authority.target, [0.1, 0.9])
    authority.switch("policy", np.array([0.1, 0.9]))
    assert not authority.accept(requested, [[0.4, 0.6]])
    assert authority.accept(authority.generation, [[0.2, 0.8]])


def test_invalid_chunks_fail_before_becoming_commands():
    authority = Authority(np.zeros(2))
    authority.switch("policy", np.zeros(2))
    for chunk in ([[float("nan"), 0]], [[0]], []):
        with pytest.raises(ValueError, match="malformed"):
            authority.accept(authority.generation, chunk)
    assert not authority.actions


def test_recording_uses_only_device_acknowledged_targets(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    session.record()
    session.set_target(0, 0.8)
    session.tick(device, Observation(np.array([0.0, 1.0]), 1.0, image, 1.0, 0, 0))
    accepted = AcceptedTarget(11, 1.04, (0.03, 1.0), "human", 1, 0)
    session.tick(device, Observation(np.array([0.01, 1.0]), 1.05, image, 1.05, 0, 11, accepted))
    path = session.finish(keep=True)

    _, frames = read_episode(path)
    assert frames[0]["applied"] is False
    assert frames[1]["applied"] is True
    assert frames[1]["command_sequence"] == 11
    assert frames[1]["command_ts"] == 1.04
    np.testing.assert_array_equal(device.command.call_args_list[0].args[0], [0.8, 1.0])


def test_late_inference_failure_does_not_override_human_correction(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    observation = Observation(np.zeros(2), 1.0, np.zeros((2, 3, 3), np.uint8), 1.0, 0, 0)
    session.latest = observation
    session.authority.switch("policy", observation.state)
    session._request_generation = session.authority.generation
    pending = Future()
    session._future = pending
    session.switch("correction")
    session.set_target(0, 0.4)
    pending.set_exception(RuntimeError("old request failed"))

    session.tick(device, observation)

    assert session.authority.mode == "correction"
    assert session.authority.target[0] == 0.4
    assert session.error is None


def test_reset_generation_cannot_acknowledge_a_previous_episode_command(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    device.command.side_effect = [11, 12]
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    session.record()
    session.tick(device, Observation(np.zeros(2), 1.0, image, 1.0, 3, 0))
    # Sequence numbers alone are insufficient after a device reset.
    session.tick(device, Observation(np.zeros(2), 1.05, image, 1.05, 4, 11))
    path = session.finish(keep=True)

    _, frames = read_episode(path)
    assert frames[0]["applied"] is False
    assert len(frames) == 2
    assert frames[1]["applied"] is False
    assert session.authority.mode == "hold"
    assert device.command.call_count == 1


def test_hold_stops_commands_and_rejects_previous_inference(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    observation = Observation(np.zeros(2), 1.0, np.zeros((2, 3, 3), np.uint8), 1.0, 0, 0)
    session.latest = observation
    session.authority.switch("policy", observation.state)
    requested = session.authority.generation
    session.switch("hold")

    session.tick(device, observation)

    device.command.assert_not_called()
    assert not session.authority.accept(requested, [[0.2, 0.5]])


def test_invalid_manual_target_does_not_enable_control(session):
    session.authority.switch("hold", np.zeros(2))

    with pytest.raises(ValueError, match="outside joint limits"):
        session.set_target(0, float("nan"))

    assert session.authority.mode == "hold"


def test_elapsed_policy_steps_are_skipped_and_completed_horizons_replan(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    observation = Observation(np.zeros(2), 1.0, np.zeros((2, 3, 3), np.uint8), 1.0, 0, 0)
    session.latest = observation
    session.policy = mocker.Mock()
    session.switch("policy")
    session._request_generation = session.authority.generation
    session._request_observed_at = 1.0
    pending = Future()
    pending.set_result([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    session._future = pending
    clock = mocker.patch("dimos.imitation.loop.session.time.time", return_value=1.12)
    submit = mocker.patch.object(session._executor, "submit", return_value=Future())

    session.tick(device, observation)

    np.testing.assert_allclose(device.command.call_args.args[0], [0.3, 0.3])
    clock.return_value = 1.23
    session.tick(device, observation)
    np.testing.assert_allclose(device.command.call_args.args[0], [0, 0])
    assert session.authority.mode == "policy"
    submit.assert_called_once()


def test_inference_past_the_deadline_requires_explicit_resume(session, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    observation = Observation(np.zeros(2), 1.0, np.zeros((2, 3, 3), np.uint8), 1.0, 0, 0)
    session.latest = observation
    session.policy = mocker.Mock()
    session.switch("policy")
    session._request_generation = session.authority.generation
    session._request_observed_at = 1.0
    pending = Future()
    pending.set_result([[0.5, 0.5]])
    session._future = pending
    mocker.patch("dimos.imitation.loop.session.time.time", return_value=1.6)

    session.tick(device, observation)

    assert session.authority.mode == "hold"
    assert "deadline" in session.error
    device.command.assert_not_called()
