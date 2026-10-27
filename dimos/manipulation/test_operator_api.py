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

"""Operator boundary: identity is checked before dispatch and typed values survive JSON."""

import json
from unittest.mock import MagicMock

import pytest

from dimos.manipulation.manipulation_spec import (
    ExecutionResult,
    ExecutionStatus,
    PlanResult,
    PlanStatus,
)
from dimos.manipulation.operator_api import operator_call, operator_json, target_pose
from dimos.manipulation.planning.monitor.world_monitor import WorldMonitor
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryState, TrajectoryStatus


def test_session_guard_precedes_dispatch_and_changes_on_restart(module_factory, mocker):
    module = module_factory()
    module._world_monitor = MagicMock(spec=WorldMonitor)
    dispatch = mocker.patch(
        "dimos.manipulation.manipulation_module.operator_call", return_value={"ok": True}
    )
    session = module.describe_operator()["session"]
    with pytest.raises(RuntimeError, match="session changed"):
        module.operator_request("previous-session", "execute", {"plan_id": "p"})
    dispatch.assert_not_called()
    assert module.operator_request(session, "get_state") == {"ok": True}
    module.stop()
    module.start()
    assert module.describe_operator()["session"] != session
    with pytest.raises(RuntimeError, match="session changed"):
        module.operator_request(session, "execute", {"plan_id": "p"})
    assert dispatch.call_count == 1


def test_stopped_planner_cannot_dispatch(module_factory, mocker):
    module = module_factory()
    module.stop()
    dispatch = mocker.patch("dimos.manipulation.manipulation_module.operator_call")
    with pytest.raises(RuntimeError, match="not ready"):
        module.operator_request(module.describe_operator()["session"], "cancel")
    dispatch.assert_not_called()


def test_json_preserves_plan_status_trajectory_and_pose():
    trajectory = JointTrajectory(
        joint_names=["left_joint1"],
        points=[TrajectoryPoint(positions=[0.123456789], time_from_start=0.5)],
    )
    encoded = json.loads(
        json.dumps(
            operator_json(
                {
                    "result": PlanResult(PlanStatus.FAILED, "No solution"),
                    "trajectory": trajectory,
                    "pose": PoseStamped(frame_id="world", position=[0.1, 0.2, 0.3]),
                }
            )
        )
    )
    assert encoded["result"]["status"]["name"] == "FAILED"
    assert encoded["trajectory"]["points"][0]["positions"] == [0.123456789]
    assert encoded["pose"]["position"] == {"x": 0.1, "y": 0.2, "z": 0.3}


def test_cancel_result_keeps_the_actual_execution_outcome():
    result = ExecutionResult(
        ExecutionStatus.ABORTED,
        trajectory_status=TrajectoryStatus(
            state=TrajectoryState.ABORTED,
            progress=0.25,
            timestamp=42.0,
            time_elapsed=1.0,
            time_remaining=3.0,
            error="Operator hold",
        ),
    )
    encoded = json.loads(json.dumps(operator_json(result)))
    assert encoded["status"]["name"] == "ABORTED"
    assert encoded["trajectory_status"] == {
        "timestamp": 42.0,
        "state": {"name": "ABORTED", "value": 3},
        "progress": 0.25,
        "time_elapsed": 1.0,
        "time_remaining": 3.0,
        "error": "Operator hold",
    }


def test_joint_targets_keep_group_identity_and_joint_order(module_factory, mocker):
    module = module_factory()
    plan = mocker.patch.object(
        module, "plan_to_joints", return_value=PlanResult(PlanStatus.FAILED, "fixture")
    )
    operator_call(
        module,
        MagicMock(spec=WorldMonitor),
        "plan_to_joints",
        {
            "targets": {"left": {"left_joint2": 0.2, "left_joint1": 0.1}},
            "speed": 0.25,
        },
    )
    target = plan.call_args.args[0]
    assert set(target) == {"left"}
    assert dict(zip(target["left"].name, target["left"].position, strict=True)) == {
        "left_joint1": 0.1,
        "left_joint2": 0.2,
    }
    assert plan.call_args.kwargs == {"speed_scale": 0.25}


@pytest.mark.parametrize("payload", [{}, {"plan_id": ""}, {"plan_id": 3}])
def test_execute_requires_an_inspected_plan(module_factory, mocker, payload):
    module = module_factory()
    execute = mocker.patch.object(module, "execute")
    with pytest.raises(ValueError, match="plan_id"):
        operator_call(module, MagicMock(spec=WorldMonitor), "execute", payload)
    execute.assert_not_called()


@pytest.mark.parametrize("orientation", [[0, 0, 0, 0], [0, 0, 0, float("nan")], [0, 0, 1]])
def test_bad_pose_rejected_before_planning(orientation):
    with pytest.raises(ValueError):
        target_pose({"position": [0, 0, 0], "orientation": orientation})
