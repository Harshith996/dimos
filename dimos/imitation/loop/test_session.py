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

from dimos.imitation.loop.act import PolicyRunner
from dimos.imitation.loop.device import Observation, OpenYamSimulation
from dimos.imitation.loop.recording import Episode
from dimos.imitation.loop.session import LearningSession
from dimos.robot.manipulators.openyam.control import YamControl
from dimos.robot.manipulators.openyam.device import AcceptedTarget


@pytest.fixture
def session(tmp_path, mocker):
    control = mocker.create_autospec(YamControl, instance=True)
    control.teleop_engaged = False
    control.transition.side_effect = range(1, 100)
    value = LearningSession(
        tmp_path,
        ["joint", "gripper"],
        (-1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        image_shape=(2, 3, 3),
        control=control,
    )
    value.latest = Observation(np.array([0.0, 1.0]), 1.0, np.zeros((2, 3, 3), np.uint8), 1.0, 0, 0)
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def policy(session, mocker):
    runner = mocker.create_autospec(PolicyRunner, instance=True)
    runner.manifest = dict(session.contract)
    runner.identity = "checkpoint-sha"
    runner.status = {"active": True, "chunks_accepted": 3, "last_error": None}
    session.load(runner)
    return runner


def test_takeover_fences_coordinator_before_stopping_native_rollout(session, policy, mocker):
    order = mocker.Mock()
    order.attach_mock(session.control.transition, "transition")
    order.attach_mock(policy.stop_rollout, "stop_rollout")
    session.authority.mode = "policy"
    session.switch("human")
    assert session.authority.mode == "correction"
    assert [call[0] for call in order.mock_calls] == ["transition", "stop_rollout"]
    session.switch("policy")
    policy.start_rollout.assert_called_once()


def test_shared_runtime_alone_dispatches_policy_commands(session, policy, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    session.switch("policy")
    session.tick(device, session.latest)
    policy.observe.assert_called_once_with(session.latest)
    assert session.chunks == 3
    device.command.assert_not_called()


def test_failed_save_is_not_counted_and_allows_a_new_recording(session, mocker, tmp_path):
    failed = mocker.create_autospec(Episode, instance=True)
    failed.directory = tmp_path / "failed"
    failed.finish.side_effect = TimeoutError("recording did not drain")
    replacement = mocker.create_autospec(Episode, instance=True)
    replacement.directory = tmp_path / "replacement"
    mocker.patch("dimos.imitation.loop.session.Episode", side_effect=[failed, replacement])
    session.record()

    with pytest.raises(TimeoutError, match="did not drain"):
        session.finish(keep=True)

    assert session.episode is None
    assert session._saved == 0
    session.record()
    assert session.episode is replacement


def test_recording_uses_only_device_acknowledged_targets(session, mocker, tmp_path):
    episode = mocker.create_autospec(Episode, instance=True)
    episode.directory = tmp_path
    mocker.patch("dimos.imitation.loop.session.Episode", return_value=episode)
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    session.record()
    session.set_target(0, 0.8)
    session.tick(device, session.latest)
    accepted = AcceptedTarget(11, 1.04, (0.03, 1.0), "human", 1, 0)
    observation = Observation(
        np.array([0.01, 1.0]), 1.05, session.latest.image, 1.05, 0, 11, accepted
    )
    session.tick(device, observation)
    first, second = [call.args[0] for call in episode.append.call_args_list]
    assert first["applied"] is False
    assert second["applied"] is True
    assert second["accepted_action"] == [0.03, 1.0]
    assert second["command_sequence"] == 11
    assert second["command_ts"] == 1.04
    np.testing.assert_array_equal(device.command.call_args_list[0].args[0], [0.8, 1.0])


def test_reset_generation_stops_native_rollout_and_requires_explicit_resume(
    session, policy, mocker
):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    session.switch("policy")
    reset = Observation(np.zeros(2), 2.0, session.latest.image, 2.0, 1, 0)
    session.tick(device, reset)
    assert session.authority.mode == "hold"
    policy.stop_rollout.assert_called_once()
    device.command.assert_not_called()


def test_native_runtime_failure_latches_hold(session, policy, mocker):
    device = mocker.create_autospec(OpenYamSimulation, instance=True)
    session.switch("policy")
    policy.status = {
        "active": False,
        "chunks_accepted": 3,
        "last_error": "Inference finished after the observation deadline",
    }
    session.tick(device, session.latest)
    assert session.authority.mode == "hold"
    assert "deadline" in session.error
    device.command.assert_not_called()


def test_failed_preflight_cannot_release_control(session, policy):
    policy.start_rollout.side_effect = RuntimeError("Checkpoint preflight failed")
    with pytest.raises(RuntimeError, match="preflight failed"):
        session.switch("policy")
    assert session.authority.mode == "hold"
    session.control.transition.assert_called_with("hold")


def test_invalid_manual_target_does_not_enable_control(session):
    session.switch("hold")
    with pytest.raises(ValueError, match="outside joint limits"):
        session.set_target(0, float("nan"))
    assert session.authority.mode == "hold"
