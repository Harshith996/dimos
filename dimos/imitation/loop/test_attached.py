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
import time

import numpy as np
import pytest

from dimos.control.coordinator import ControlCoordinator
from dimos.control.tasks.trajectory_task.trajectory_task import (
    TrajectoryExecutionResult,
    TrajectoryExecutionStatus,
)
from dimos.core.introspection.module.info import ModuleInfo, RpcInfo, StreamInfo
from dimos.core.run_registry import RunEntry
from dimos.imitation.loop.attached import AttachedBlueprint
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.porcelain.dimos import Dimos

JOINTS = ["gantry/slide", "wrist/rotation", "tool/opening"]
OWNER = "renamed-controller"


@pytest.fixture
def attached(mocker, monkeypatch, tmp_path):
    monkeypatch.setattr("dimos.core.run_registry.REGISTRY_DIR", tmp_path)
    mocker.patch("dimos.imitation.loop.attached.global_config")
    mocker.patch("dimos.imitation.loop.attached.is_pid_alive", return_value=True)
    clock = mocker.patch("dimos.imitation.loop.attached.time.monotonic", return_value=100.0)
    run = RunEntry("test-run", 1234, "unrelated-plugin-blueprint", "", "")
    run.save()
    discovery = mocker.patch("dimos.imitation.loop.attached.list_runs", return_value=[run])
    app = mocker.create_autospec(Dimos, instance=True)
    owner = mocker.create_autospec(ControlCoordinator, instance=True)
    mocker.patch.object(owner, "get_stream", create=True)
    owner.list_joints.return_value = JOINTS
    owner.describe_controller.return_value = {
        "run_id": "test-run",
        "joints": [
            {"name": n, "lower": -2 if i < 2 else 0, "upper": 2 if i < 2 else 1}
            for i, n in enumerate(JOINTS)
        ],
        "tasks": [
            {"name": "move", "joints": JOINTS[:2], "commands": [], "trajectory": True},
            {
                "name": "tool_task_with_any_name",
                "joints": JOINTS[2:],
                "commands": ["set_position"],
                "trajectory": False,
            },
        ],
    }
    owner.get_trajectory_generation.return_value = 4
    owner.execute_trajectory.return_value = TrajectoryExecutionResult(
        TrajectoryExecutionStatus.ACCEPTED
    )
    owner.task_invoke.return_value = True
    app.get_module.return_value = owner
    app.list_modules.return_value = [
        ModuleInfo(
            name=OWNER,
            instance_name=OWNER,
            class_name="ExternalPluginController",
            rpcs=[
                RpcInfo(name=n)
                for n in ["list_joints", "get_joint_positions", "describe_controller"]
            ],
        )
    ]
    app.list_streams.return_value = [
        StreamInfo(
            name="coordinator_joint_state",
            module_name=OWNER,
            direction="out",
            type_name="JointState",
        )
    ]
    mocker.patch("dimos.imitation.loop.attached.Dimos.connect", return_value=app)
    with closing(AttachedBlueprint()) as viewer:
        viewer.start()
        owner.get_stream.return_value.subscribe.call_args.args[0](
            JointState(name=JOINTS, position=[0.1, 0.2, 0.3], ts=time.time())
        )
        yield viewer, owner, app, discovery, clock


def test_discovers_renamed_plugin_and_arbitrary_joints_without_owning_devices(attached):
    viewer, owner, app, _, _ = attached
    state = viewer.state()
    assert state["attached"]["blueprint"] == "unrelated-plugin-blueprint"
    assert state["joint_names"] == JOINTS
    assert state["joints"] == [0.1, 0.2, 0.3]
    assert state["commandable"] == [True, True, True]
    viewer.close()
    app.stop.assert_called_once_with()
    owner.start.assert_not_called()
    owner.stop.assert_not_called()
    owner.execute_trajectory.assert_not_called()
    owner.task_invoke.assert_not_called()


def test_joint_jog_uses_advertised_owner_and_cancellation_fence(attached):
    viewer, owner, _, _, _ = attached
    viewer.dispatch("target", {"run_id": "test-run", "index": 1, "value": 0.4})
    trajectory = owner.execute_trajectory.call_args.args[0]
    assert trajectory.joint_names == [JOINTS[1]]
    assert [p.positions for p in trajectory.points] == [[0.2], [0.4]]
    assert trajectory.duration == pytest.approx(0.8)
    assert owner.execute_trajectory.call_args.kwargs == {"expected_generation": 4}
    viewer.dispatch("hold", {"run_id": "test-run"})
    owner.cancel_trajectory.assert_called_once_with(expected_generation=4)


def test_scalar_position_uses_advertised_task_without_gripper_name_guessing(attached):
    viewer, owner, _, _, _ = attached
    viewer.dispatch("target", {"run_id": "test-run", "index": 2, "value": 0.8})
    owner.task_invoke.assert_called_once_with(
        "tool_task_with_any_name", "set_position", {"values": [0.8]}
    )
    owner.execute_trajectory.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [
        {"run_id": "old", "index": 0, "value": 0.2},
        {"run_id": "test-run", "index": True, "value": 0.2},
        {"run_id": "test-run", "index": 0, "value": True},
        {"run_id": "test-run", "index": 0, "value": float("nan")},
        {"run_id": "test-run", "index": 0, "value": 3},
    ],
)
def test_stale_or_invalid_input_sends_no_command(attached, params):
    viewer, owner, _, _, _ = attached
    with pytest.raises((RuntimeError, ValueError)):
        viewer.dispatch("target", params)
    owner.execute_trajectory.assert_not_called()
    owner.task_invoke.assert_not_called()


def test_unknown_limits_or_missing_owner_disable_control(attached):
    viewer, owner, _, _, _ = attached
    viewer.close()
    owner.describe_controller.return_value["joints"][0]["lower"] = None
    owner.describe_controller.return_value["tasks"] = owner.describe_controller.return_value[
        "tasks"
    ][:1]
    viewer.start()
    assert viewer.state()["commandable"] == [False, True, False]
    with pytest.raises(ValueError, match="advertised manual control"):
        viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 0.2})
    owner.execute_trajectory.assert_not_called()


def test_heartbeat_loss_cancels_only_workcells_own_jog(attached):
    viewer, owner, _, _, clock = attached
    viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 1.5})
    clock.return_value = 103
    viewer.update()
    owner.cancel_trajectory.assert_called_once_with(expected_generation=4)


def test_follows_restart_and_rejects_old_feedback_and_commands(attached):
    viewer, owner, app, discovery, clock = attached
    old_feedback = owner.get_stream.return_value.subscribe.call_args.args[0]
    replacement = RunEntry("next", 5678, "different-robot", "", "")
    replacement.save()
    discovery.return_value = [replacement]
    owner.describe_controller.return_value["run_id"] = "next"
    clock.return_value = 102
    viewer.update()
    old_feedback(JointState(name=JOINTS, position=[0.8] * 3, ts=time.time()))
    assert viewer.state()["joints"] == []
    assert viewer.state()["run_id"] == "next"
    with pytest.raises(RuntimeError, match="displayed run changed"):
        viewer.dispatch("target", {"run_id": "test-run", "index": 0, "value": 0.2})
    owner.get_stream.return_value.subscribe.call_args.args[0](
        JointState(name=JOINTS, position=[0.4] * 3, ts=time.time())
    )
    assert viewer.state()["joints"] == [0.4] * 3
    app.stop.assert_called_once_with()
    owner.execute_trajectory.assert_not_called()


def test_opening_before_run_and_multiple_run_selection(attached):
    viewer, owner, _, discovery, clock = attached
    original = viewer.run
    viewer.close()
    discovery.return_value = []
    viewer.start()
    assert viewer.state()["run_id"] is None
    second = RunEntry("second", 2222, "another-robot", "", "")
    second.save()
    discovery.return_value = [original, second]
    clock.return_value = 102
    viewer.update()
    assert viewer.state()["run_id"] is None
    assert "Select" in viewer.state()["error"]
    owner.describe_controller.return_value["run_id"] = "second"
    viewer.dispatch("select_run", {"selected_run_id": "second"})
    assert viewer.state()["run_id"] == "second"
    owner.start.assert_not_called()
    owner.execute_trajectory.assert_not_called()


def test_rejects_a_controller_from_another_run_on_the_same_bus(attached):
    viewer, owner, _, _, _ = attached
    viewer.close()
    owner.describe_controller.return_value["run_id"] = "another-run"
    viewer.start()
    assert viewer.state()["run_id"] is None
    assert "different run" in viewer.state()["error"]
    owner.execute_trajectory.assert_not_called()
    owner.task_invoke.assert_not_called()


def test_optional_views_and_recording_are_discovered_by_interface(attached, mocker):
    viewer, owner, app, _, _ = attached
    viewer.close()
    provider = mocker.Mock()
    provider.get_web_views.return_value = [
        {"title": "Other view", "path": "/world", "port": 12345, "scheme": "https", "embed": True}
    ]
    provider.get_status.return_value = {"collection_available": True, "collection_ready": True}
    provider.record.return_value = {"recording": "episode"}
    app.list_modules.return_value.append(
        ModuleInfo(
            name="plugin",
            instance_name="plugin",
            class_name="UnknownPlugin",
            rpcs=[
                RpcInfo(name=n)
                for n in [
                    "get_web_views",
                    "record",
                    "finish",
                    "get_status",
                    "describe_run",
                    "scene_state",
                ]
            ],
        )
    )
    app.get_module.side_effect = lambda name: provider if name == "plugin" else owner
    viewer.start()
    assert viewer.state()["web_views"][0]["path"] == "/world"
    assert viewer.state()["environment"] == "simulation"
    viewer.dispatch("record", {"run_id": "test-run", "task": "Move an object", "split": "eval"})
    provider.record.assert_called_once_with(task="Move an object", split="eval")
    owner.execute_trajectory.assert_not_called()


def test_late_camera_frame_from_previous_run_is_discarded(attached):
    viewer, _, _, discovery, clock = attached
    image = Image.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), format=ImageFormat.RGB)
    viewer._image("test-run", "camera", image)
    assert "camera" in viewer.images()
    discovery.return_value = []
    clock.return_value = 102
    viewer.update()
    viewer._image("test-run", "camera", image)
    assert viewer.images() == {}
