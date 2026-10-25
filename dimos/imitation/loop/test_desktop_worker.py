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

from dimos.imitation.loop.desktop_worker import DesktopExperiment
from dimos.imitation.loop.device import Observation, OpenYamSimulation
from dimos.imitation.loop.session import LearningSession
from dimos.robot.manipulators.openyam.control import YamControl
from dimos.robot.manipulators.openyam.joints import OPENYAM_JOINTS

HOME = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5)
LOWER = (-3.0, 0.01, 0.01, -1.5, -1.5, -2.0, 0.0)
UPPER = (3.0, 3.6, 3.6, 1.5, 1.5, 2.0, 1.0)


@pytest.fixture
def experiment(tmp_path):
    value = DesktopExperiment(tmp_path)
    try:
        yield value
    finally:
        value.close()


def test_catalog_and_heartbeat_do_not_start_simulation(experiment, mocker):
    simulator = mocker.patch("dimos.imitation.loop.desktop_worker.OpenYamSimulation", autospec=True)

    experiment.dispatch("heartbeat", {})
    state = experiment.state()
    with pytest.raises(RuntimeError, match="Connect the station or start the simulation"):
        experiment.dispatch("target", {"index": 0, "value": 0.2})

    simulator.assert_not_called()
    assert state["running"] is False


def test_catalog_tolerates_a_checkpoint_being_published(experiment):
    checkpoint = experiment.root / "models" / "publishing"
    checkpoint.mkdir(parents=True)
    (checkpoint / "loop.json").write_text('{"training_job":')

    assert experiment.state()["models"] == []


def test_expired_operator_lease_holds_until_explicit_input(experiment, mocker):
    robot = mocker.create_autospec(OpenYamSimulation, instance=True)
    observation = Observation(np.asarray(HOME), 1, np.zeros((2, 3, 3), np.uint8), 1, 0, 0)
    robot.observe.return_value = observation
    robot.control = mocker.create_autospec(YamControl, instance=True)
    robot.control.teleop_engaged = False
    robot.control.teleop_port = None
    experiment.robot = robot
    session = LearningSession(experiment.root, list(OPENYAM_JOINTS), LOWER, UPPER, HOME)
    experiment.session = session
    experiment.stack.callback(session.close)
    experiment._next_frame = float("inf")
    experiment.last_heartbeat = 0
    mocker.patch("dimos.imitation.loop.desktop_worker.time.monotonic", return_value=10)

    experiment.update()
    experiment.dispatch("heartbeat", {})
    experiment.update()

    assert session.authority.mode == "hold"
    robot.command.assert_not_called()

    experiment.dispatch("target", {"index": 0, "value": 0.1})
    experiment.update()

    assert session.authority.mode == "human"
    robot.command.assert_called_once()
