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

from contextlib import closing
from pathlib import Path
import time

import pytest

from dimos.control.coordinator import ControlCoordinator
from dimos.control.tasks.trajectory_task.trajectory_task import (
    TrajectoryExecutionResult,
    TrajectoryExecutionStatus,
)
from dimos.core.run_registry import RunEntry
from dimos.imitation.loop.attached import AttachedDualArm
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos
from dimos.robot.assets.model import LoadedRobotModel
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_JOINTS,
)


@pytest.fixture
def attached(mocker, monkeypatch, tmp_path):
    monkeypatch.setattr("dimos.core.run_registry.REGISTRY_DIR", tmp_path)
    mocker.patch("dimos.imitation.loop.attached.global_config")
    mocker.patch("dimos.imitation.loop.attached.is_pid_alive", return_value=True)
    mocker.patch("dimos.imitation.loop.attached.time.monotonic", return_value=100.0)
    mocker.patch("dimos.imitation.loop.attached.discover_webcams", return_value=[])
    model = mocker.patch("dimos.imitation.loop.attached.DUAL_OPENYAM_MODEL")
    model.load.return_value = LoadedRobotModel(
        '<robot name="test"><link name="base"/>'
        + "".join(
            f'<link name="{name}"/><joint name="{name}" type="revolute">'
            f'<parent link="base"/><child link="{name}"/>'
            '<axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint>'
            for name in DUAL_OPENYAM_ARM_JOINTS
        )
        + "</robot>",
        Path("test.urdf"),
        {},
    )
    app = mocker.create_autospec(Dimos, instance=True)
    coordinator = mocker.create_autospec(ControlCoordinator, instance=True)
    mocker.patch.object(coordinator, "get_stream", create=True)
    coordinator.get_joint_positions.return_value = dict.fromkeys(DUAL_OPENYAM_JOINTS, 0.1)
    coordinator.get_trajectory_generation.return_value = 4
    coordinator.execute_trajectory.return_value = TrajectoryExecutionResult(
        TrajectoryExecutionStatus.ACCEPTED
    )
    coordinator.task_invoke.return_value = True
    app.get_module.return_value = coordinator
    app.list_streams.return_value = []
    mocker.patch("dimos.imitation.loop.attached.Dimos.connect", return_value=app)
    run = RunEntry("test-run", 1234, "teleop-webxr-dual-openyam", "", "")
    run.save()
    mocker.patch("dimos.imitation.loop.attached.running_dual_arm", return_value=run)
    viewer = AttachedDualArm()
    try:
        viewer.start()
        coordinator.get_stream.return_value.subscribe.call_args.args[0](
            JointState(name=DUAL_OPENYAM_JOINTS, position=[0.1] * 14, ts=time.time())
        )
        yield viewer, coordinator, app
    finally:
        viewer.close()


def test_attach_poll_and_close_leave_both_arms_owned_by_the_blueprint(attached):
    viewer, coordinator, app = attached
    viewer.update()
    state = viewer.state()
    viewer.close()

    assert state["run_id"] == "test-run"
    assert state["joint_names"] == DUAL_OPENYAM_JOINTS
    assert state["joints"] == [0.1] * 14
    assert state["running"] is True
    app.stop.assert_called_once_with()
    coordinator.start.assert_not_called()
    coordinator.stop.assert_not_called()
    coordinator.execute_trajectory.assert_not_called()
    coordinator.cancel_trajectory.assert_not_called()
    coordinator.task_invoke.assert_not_called()
    coordinator.get_joint_positions.assert_not_called()


def test_right_joint_command_routes_to_existing_coordinator_and_can_be_cancelled(attached):
    viewer, coordinator, _ = attached
    state = viewer.dispatch("target", {"run_id": "test-run", "index": 7, "value": 0.2})
    trajectory = coordinator.execute_trajectory.call_args.args[0]
    assert trajectory.joint_names == ["right_joint2"]
    assert [point.positions for point in trajectory.points] == [[0.1], [0.2]]
    assert trajectory.duration == pytest.approx(0.4)
    assert coordinator.execute_trajectory.call_args.kwargs == {"expected_generation": 4}
    assert state["targets"][7] == 0.2
    viewer.dispatch("hold", {"run_id": "test-run"})
    coordinator.cancel_trajectory.assert_called_once_with(expected_generation=4)


@pytest.mark.parametrize(("index", "task"), [(12, "left_arm_gripper"), (13, "right_arm_gripper")])
def test_each_gripper_uses_its_existing_task(attached, index, task):
    viewer, coordinator, _ = attached
    viewer.dispatch("target", {"run_id": "test-run", "index": index, "value": 0.3})
    coordinator.task_invoke.assert_called_once_with(task, "set_normalized", {"values": [0.3]})
    coordinator.execute_trajectory.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [
        {"run_id": "old-run", "index": 0, "value": 0.2},
        {"run_id": "test-run", "index": 0, "value": float("nan")},
        {"run_id": "test-run", "index": 0, "value": 3.0},
    ],
)
def test_stale_run_or_invalid_target_never_sends_a_command(attached, params):
    viewer, coordinator, _ = attached
    with pytest.raises((ValueError, RuntimeError)):
        viewer.dispatch("target", params)
    coordinator.execute_trajectory.assert_not_called()
    coordinator.task_invoke.assert_not_called()


def test_stopped_blueprint_is_not_replaced_with_another_session(attached, mocker):
    viewer, coordinator, _ = attached
    mocker.patch("dimos.imitation.loop.attached.is_pid_alive", return_value=False)
    with pytest.raises(RuntimeError, match="reconnects automatically"):
        viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 0.2})
    coordinator.execute_trajectory.assert_not_called()


def test_lost_viewer_heartbeat_cancels_its_inflight_jog(attached, mocker):
    viewer, coordinator, _ = attached
    clock = mocker.patch("dimos.imitation.loop.attached.time.monotonic", return_value=100.0)
    viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 1.0})
    clock.return_value = 103.0
    viewer.update()
    coordinator.cancel_trajectory.assert_called_once_with(expected_generation=4)


@pytest.mark.parametrize("stop_between", [False, True])
def test_follows_replacement_run_without_replaying_controls(attached, mocker, stop_between):
    viewer, coordinator, app = attached
    old_feedback = coordinator.get_stream.return_value.subscribe.call_args.args[0]
    viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 0.2})
    replacement = RunEntry("replacement", 5678, "teleop-webxr-dual-openyam", "", "")
    replacement.save()
    discovery = mocker.patch("dimos.imitation.loop.attached.running_dual_arm", return_value=None)
    clock = mocker.patch("dimos.imitation.loop.attached.time.monotonic", return_value=101.0)
    if stop_between:
        viewer.update()
        assert viewer.state()["mode"] == "Waiting for blueprint"
        assert viewer.state()["run_id"] is None
        assert viewer.state()["joints"] == []
    discovery.return_value = replacement
    clock.return_value = 102.0

    viewer.update()
    old_feedback(JointState(name=DUAL_OPENYAM_JOINTS, position=[0.8] * 14, ts=time.time()))
    assert viewer.state()["joints"] == []
    assert viewer.state()["running"] is False
    assert viewer.state()["run_id"] == "replacement"
    assert viewer.state()["attached"]["pid"] == 5678
    with pytest.raises(RuntimeError, match="displayed run changed"):
        viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 0.4})
    coordinator.get_stream.return_value.subscribe.call_args.args[0](
        JointState(name=DUAL_OPENYAM_JOINTS, position=[0.3] * 14, ts=time.time())
    )
    assert viewer.state()["running"] is True
    assert viewer.state()["targets"] == [0.3] * 14
    assert viewer.state()["error"] is None
    app.stop.assert_called_once_with()
    coordinator.execute_trajectory.assert_called_once()
    coordinator.cancel_trajectory.assert_not_called()
    coordinator.start.assert_not_called()
    coordinator.stop.assert_not_called()
    coordinator.task_invoke.assert_not_called()


def test_viewer_opened_before_blueprint_attaches_when_it_appears(attached, mocker):
    viewer, coordinator, _ = attached
    run = viewer.run
    viewer.close()
    discovery = mocker.patch("dimos.imitation.loop.attached.running_dual_arm", return_value=None)
    with closing(AttachedDualArm()) as waiting:
        waiting.start()
        assert waiting.state()["mode"] == "Waiting for blueprint"
        assert waiting.state()["attached"]["pid"] is None
        discovery.return_value = run
        waiting.update()
        coordinator.get_stream.return_value.subscribe.call_args.args[0](
            JointState(name=DUAL_OPENYAM_JOINTS, position=[0.2] * 14, ts=time.time())
        )
        assert waiting.state()["running"] is True
        assert waiting.state()["run_id"] == "test-run"
    coordinator.execute_trajectory.assert_not_called()
    coordinator.start.assert_not_called()
    coordinator.stop.assert_not_called()


def test_blueprint_not_ready_yet_is_retried_without_restarting_worker(attached, mocker):
    viewer, coordinator, app = attached
    viewer.close()
    mocker.patch(
        "dimos.imitation.loop.attached.Dimos.connect",
        side_effect=[RuntimeError("Coordinator is starting"), app],
    )
    with closing(AttachedDualArm()) as waiting:
        waiting.start()
        assert waiting.state()["running"] is False
        assert "Coordinator is starting" in waiting.state()["error"]
        waiting.update()
        coordinator.get_stream.return_value.subscribe.call_args.args[0](
            JointState(name=DUAL_OPENYAM_JOINTS, position=[0.2] * 14, ts=time.time())
        )
        assert waiting.state()["running"] is True
        assert waiting.state()["error"] is None
        assert len(waiting.state()["lower"]) == 14
    coordinator.execute_trajectory.assert_not_called()
