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

from unittest.mock import MagicMock

import pytest

from dimos.evals.environments.hardware import HardwareEnvironment
from dimos.evals.environments.sim import Sim
from dimos.evals.scorers import human


def environment(**kwargs):
    return HardwareEnvironment(
        blueprint=["xarm-grasp", "mcp-server"], task="pick the cup", **kwargs
    )


def test_robot_does_not_start_until_the_operator_confirms(mocker):
    prompts = []
    mocker.patch("builtins.input", side_effect=lambda text: prompts.append(text) or "n")
    launch = mocker.patch.object(Sim, "start")

    with pytest.raises(RuntimeError, match="did not confirm"):
        environment().start(())

    assert "pick the cup" in prompts[0]
    launch.assert_not_called()


def test_stop_halts_the_robot_before_releasing_it(mocker):
    calls = []
    adapter = MagicMock()
    adapter.call_tool.side_effect = lambda name: calls.append(("halt", name))
    mocker.patch("dimos.evals.environments.hardware.McpAdapter", return_value=adapter)
    mocker.patch.object(Sim, "stop", side_effect=lambda: calls.append(("release",)))
    env = environment(stop_tool="cancel")
    env._mcp_url = "http://localhost:9990/mcp"

    env.stop()

    assert calls == [("halt", "cancel"), ("release",)]


@pytest.mark.parametrize(("answer", "score"), [("y", 1.0), ("n", 0.0), ("", 0.0)])
def test_human_grade_is_the_operators_verdict(mocker, answer, score):
    mocker.patch("builtins.input", return_value=answer)
    outcome = MagicMock()
    outcome.trajectory.final_answer = "Holding the cup."

    assert human("pick the cup")(outcome) == score
