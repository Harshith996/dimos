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

"""Operator discovery, shared results and cancellation across the RPC boundary."""

from concurrent.futures import Future
from contextlib import closing
import json
from threading import Event

import pytest

from dimos.core.introspection.module.info import ModuleInfo, RpcInfo
from dimos.core.module import SkillInfo as SkillSchema
from dimos.core.run_registry import RunEntry
from dimos.imitation.loop.operator_client import OperatorClient
from dimos.porcelain.dimos import Dimos


@pytest.fixture
def client(mocker, monkeypatch, tmp_path):
    monkeypatch.setattr("dimos.core.run_registry.REGISTRY_DIR", tmp_path)
    mocker.patch("dimos.imitation.loop.operator_client.is_pid_alive", return_value=True)
    run = RunEntry("run", 123, "plugin-arm", "", "")
    run.save()
    app = mocker.create_autospec(Dimos, instance=True)
    module = mocker.MagicMock()
    app.get_module.return_value = module
    module.describe_operator.return_value = {"run_id": "run", "session": "planner-instance"}
    module.operator_request.side_effect = lambda session, method, payload=None: {
        "model": {"xml": "<robot/>"},
        "list_planning_groups": [{"id": "arm", "joint_names": ["joint"]}],
        "get_state": {"has_pending_plan": True},
        "cancel": {"status": {"name": "ABORTED"}},
        "plan_to_poses": {"plan": {"plan_id": "reviewed-plan"}},
    }[method]
    module.get_skills.return_value = [
        SkillSchema(
            "Plugin",
            "detect",
            json.dumps({"properties": {"prompts": {"type": "array"}}, "required": ["prompts"]}),
        ),
        SkillSchema("Plugin", "move", "{}", uses=("movement",)),
    ]
    module.detect.return_value = "Detected bottle"
    info = ModuleInfo(
        "plugin",
        rpcs=[RpcInfo(name=n) for n in ("describe_operator", "operator_request", "get_skills")],
    )
    with closing(OperatorClient(run)) as operator:
        operator.start(app, [info])
        yield operator, module, {"run_id": "run", "session": operator.session}


def test_stale_requests_do_not_invoke_planner(client):
    operator, module, binding = client
    module.operator_request.reset_mock()
    with pytest.raises(RuntimeError, match="session changed"):
        operator.submit(
            {
                **binding,
                "session": "old",
                "kind": "planner",
                "module": "plugin",
                "method": "execute",
                "payload": {"plan_id": "p"},
            }
        )
    module.operator_request.assert_not_called()


def test_agent_plan_is_visible_without_executing(client):
    operator, module, binding = client
    operator.submit(
        {
            **binding,
            "kind": "planner",
            "module": "plugin",
            "method": "plan_to_poses",
            "payload": {"targets": {}},
        }
    )
    operator._future.result(timeout=2)
    snapshot = operator.snapshot()
    assert snapshot["planners"][0]["plan"] == {"plan_id": "reviewed-plan"}
    assert snapshot["job"]["status"] == "completed"
    assert all(call.args[1] != "execute" for call in module.operator_request.call_args_list)


def test_skills_use_advertised_schema_and_results(client):
    operator, module, binding = client
    operator.submit(
        {
            **binding,
            "kind": "skill",
            "module": "plugin",
            "method": "detect",
            "payload": {"prompts": ["bottle"]},
        }
    )
    operator._future.result(timeout=2)
    module.detect.assert_called_once_with(prompts=["bottle"])
    assert operator.snapshot()["job"]["result"] == "Detected bottle"
    for method in ("delete_everything", "move"):
        with pytest.raises(ValueError):
            operator.submit(
                {**binding, "kind": "skill", "module": "plugin", "method": method, "payload": {}}
            )
    module.move.assert_not_called()


def test_cancelled_planning_cannot_install_late_preview(client, mocker):
    operator, module, binding = client
    entered, release = Event(), Event()
    original = module.operator_request.side_effect

    def call(session, method, payload=None):
        if method == "plan_to_poses":
            entered.set()
            assert release.wait(2)
        return original(session, method, payload)

    mocker.patch.object(module, "operator_request", side_effect=call)
    try:
        operator.submit(
            {
                **binding,
                "kind": "planner",
                "module": "plugin",
                "method": "plan_to_poses",
                "payload": {"targets": {}},
            }
        )
        assert entered.wait(2)
        operator.cancel(binding)
    finally:
        release.set()
    operator._future.result(timeout=2)
    snapshot = operator.snapshot()
    assert snapshot["planners"][0]["plan"] is None
    assert snapshot["job"]["status"] == "cancelled"


def test_cancelling_before_dispatch_does_not_leave_the_app_busy(client, mocker):
    operator, module, binding = client
    mocker.patch.object(operator._executor, "submit", return_value=Future())
    operator.submit(
        {
            **binding,
            "kind": "skill",
            "module": "plugin",
            "method": "detect",
            "payload": {"prompts": ["cup"]},
        }
    )
    operator.cancel(binding)
    assert operator.snapshot()["job"]["status"] == "cancelled"
    module.detect.assert_not_called()


def test_an_unconfirmed_stop_is_reported_to_the_calling_operator(client, mocker):
    operator, module, binding = client
    mocker.patch.object(
        module,
        "operator_request",
        return_value={"status": {"name": "UNCERTAIN"}, "message": "Coordinator timeout"},
    )
    with pytest.raises(
        RuntimeError, match="Motion stop was not confirmed: plugin: Coordinator timeout"
    ):
        operator.cancel(binding)
