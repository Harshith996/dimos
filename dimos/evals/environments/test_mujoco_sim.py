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
from pathlib import Path
import time

import numpy as np
import pytest

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.e2e_tests.dimos_cli_call import DimosCliCall
from dimos.evals.constants import RAW_ARM_README, RAW_README
from dimos.evals.environments.mujoco_sim import MujocoEnvironment
from dimos.evals.environments.runtime.recorded_poses import last_body_transform
from dimos.evals.scorers import lifted
from dimos.evals.types import Outcome, truth_recording
from dimos.memory.store.memory import MemoryStore
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.manipulators.xarm.blueprints.simulation import xarm_perception_sim
from dimos.robot.manipulators.xarm.simulation import xarm_definition
from dimos.sim2.spec import WorldConfig
from dimos.sim2.world.runtime import SimulationRuntime
from dimos.sim2.world.scene import scene_robot
from dimos.sim2.world.scene_types import EntityState, SceneControlSpec, SceneState, SceneUpdate
from dimos.utils.data import LfsPath


def environment(**kwargs):
    return MujocoEnvironment(blueprint=["xarm-perception-sim", "mcp-server"], **kwargs)


def _tf(ts: float, child: str, z: float) -> TFMessage:
    return TFMessage(
        Transform(translation=Vector3(0.4, 0.08, z), frame_id="world", child_frame_id=child, ts=ts)
    )


def test_launch_flags(monkeypatch):
    monkeypatch.delenv("SIMULATIONMODULE__VIEWER", raising=False)
    proc = DimosCliCall()
    environment(tracked_bodies=("apple", "cup"), scene=Path("scenes/table.xml")).configure_launch(
        proc
    )
    assert proc.simulator == "mujoco"
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "false"
    assert proc.global_args == [
        "--record-topics",
        "color_image,camera_info,coordinator_joint_state,tf,odom",
        "--scene-package",
        str(Path("scenes/table.xml").resolve()),
    ]
    assert not any("TRACKED_BODIES" in key or "HARDWARE" in key for key in proc.extra_env)
    monkeypatch.setenv("SIMULATIONMODULE__VIEWER", "true")
    proc = DimosCliCall()
    environment(
        module_env={"OBJECTSCENEREGISTRATIONMODULE__DETECTOR_BACKEND": "yoloe"}
    ).configure_launch(proc)
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "true"
    assert proc.extra_env["OBJECTSCENEREGISTRATIONMODULE__DETECTOR_BACKEND"] == "yoloe"


def test_module_env_reaches_blueprint_parser():
    proc = DimosCliCall()
    environment(
        module_env={"OBJECTSCENEREGISTRATIONMODULE__DETECTOR_BACKEND": "owlv2"}
    ).configure_launch(proc)
    parsed = BlueprintConfigParser(xarm_perception_sim).parse(environ=proc.extra_env)
    assert parsed.module_kwargs("objectsceneregistrationmodule")["detector_backend"] == "owlv2"
    assert parsed.module_kwargs("simulationmodule")["viewer"] is False


def test_latest_pose_needs_odom():
    env = environment()
    with MemoryStore() as store:
        with pytest.raises(LookupError):
            env.latest_pose(store)
        store.stream("odom", PoseStamped).append(PoseStamped(ts=5, frame_id="world"))
        assert env.latest_pose(store).ts == 5


def test_raw_arm_blueprint_configures_its_bridge():
    from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
    from dimos.robot.manipulators.xarm.blueprints.simulation import xarm_sim

    env = MujocoEnvironment(
        blueprint=["xarm-sim", "mcp-server"], raw_bridge=True, raw_guide=RAW_ARM_README
    )
    assert env.provides_raw_robot
    parsed = BlueprintConfigParser(xarm_sim).parse(
        environ={"RAWROBOTBRIDGE__ENDPOINT": "tcp/127.0.0.1:12345"}
    )
    bridge = parsed.module_kwargs("rawrobotbridge")
    assert bridge["ee_frame"] == "link_tcp" and bridge["gripper_joint"] == "arm/gripper"
    assert bridge["endpoint"] == "tcp/127.0.0.1:12345"  # the per-run endpoint still applies
    tasks = parsed.module_kwargs("ControlCoordinator")["tasks"]
    assert any(task["type"] == "eef_twist" for task in tasks)


def test_suite_guide_becomes_robot_md(tmp_path):
    from dimos.evals.agents.pi import PiAdapter
    from dimos.evals.suites.mujoco_xarm_pick import SUITE
    from dimos.evals.types import RunningEnvironment

    env = RunningEnvironment(
        mcp_url="unused",
        streams=(),
        artifacts={},
        raw_endpoint="tcp/127.0.0.1:12345",
        raw_guide=SUITE[0].environment.config.raw_guide,
    )
    guide = PiAdapter(no_dimos=True)._no_dimos_files(env, tmp_path)["robot"].read_text()
    assert "tcp/127.0.0.1:12345" in guide
    assert "robot/arm/twist/json" in guide and "cmd_vel/json" not in guide
    assert "0.1 m/s" in guide and "0.5 rad/s" in guide  # limits filled in
    assert "xArm7" in guide  # suite notes appended


def test_no_dimos_run_without_a_guide_is_refused_before_launch():
    from dimos.evals.agents.pi import PiAdapter

    unguided = MujocoEnvironment(blueprint=["xarm-sim"], raw_bridge=True)
    with pytest.raises(ValueError, match="raw_guide"):
        PiAdapter(no_dimos=True).preflight(unguided)


def test_navigation_suites_keep_the_go2_guide():
    from dimos.evals.suites.belief_apartment_qa import SUITE as BELIEF
    from dimos.evals.suites.dimsim_apartment_qa import SUITE as DIMSIM

    for suite in (DIMSIM, BELIEF):
        assert {case.environment.config.raw_guide for case in suite} == {RAW_README}


def test_ready_needs_fresh_streams_and_tracked_body_poses():
    env = environment(tracked_bodies=("apple",))
    with MemoryStore() as store:
        with pytest.raises(TimeoutError, match="apple"):
            env.wait_ready(store, deadline=time.monotonic())
        now = time.time()
        store.stream("color_image", Image).append(
            Image(data=np.zeros((1, 1, 3), dtype=np.uint8), format=ImageFormat.RGB, ts=now)
        )
        store.stream("coordinator_joint_state", JointState).append(
            JointState(ts=now, name=["j1"], position=[0.0], velocity=[0.0])
        )
        # Robot observations alone must not satisfy privileged truth readiness.
        store.stream("tf", TFMessage).append(_tf(now, "apple", 0.17))
        with pytest.raises(TimeoutError):
            env.wait_ready(store, deadline=time.monotonic())
        with MemoryStore() as truth:
            env._truth = truth
            truth.stream("tf", TFMessage).append(_tf(now, "apple", 0.17))
            env.wait_ready(store, deadline=time.monotonic() + 2.0)


def test_settle_waits_for_joints_to_stop():
    env = environment(at_rest_s=0.0, settle_poll_s=0.01)
    env.settle(1.0)
    with MemoryStore() as store:
        env._recording = store
        started = time.monotonic()
        env.settle(1.0)
        joints = store.stream("coordinator_joint_state", JointState)
        env.settle(1.0)
        assert time.monotonic() - started < 0.5
        joints.append(JointState(ts=1.0, name=["j1"], position=[0.0], velocity=[0.5]))
        env._recording = store
        started = time.monotonic()
        env.settle(0.2)
        assert time.monotonic() - started >= 0.2

        joints.append(JointState(ts=2.0, name=["j1"], position=[0.0], velocity=[0.0]))
        started = time.monotonic()
        env.settle(5.0)
        assert time.monotonic() - started < 1.0
    env._recording = None


def test_launch_and_cleanup(tmp_path, mocker):
    proc = mocker.patch("dimos.evals.environments.sim.DimosCliCall").return_value
    mocker.patch("dimos.evals.environments.sim.launched_mcp_url", return_value="http://x/mcp")
    proc.extra_env = {}
    proc.global_args = []
    mocker.patch(
        "dimos.evals.environments.sim.McpAdapter"
    ).return_value.wait_for_ready.return_value = True
    store = mocker.patch("dimos.memory.store.sqlite.SqliteStore").return_value
    env = environment(tracked_bodies=("apple",), disable=("rerun-bridge-module",))
    mocker.patch.object(env, "_wait_recording", return_value=tmp_path / "memory.db")
    mocker.patch.object(env, "setup_scene")
    ready = mocker.patch.object(
        env, "prepare_recording", return_value={"truth": tmp_path / "truth.db"}
    )
    try:
        result = env.start(("speak-skill",))
        assert proc.simulator == "mujoco"
        assert proc.global_args[0] == "--record-topics" and proc.global_args[-1] == "--record"
        assert proc.demo_args == [
            "run",
            "xarm-perception-sim",
            "mcp-server",
            "speak-skill",
            "--disable",
            "rerun-bridge-module",
        ]
        assert "SIMULATIONMODULE__VIEWER" in proc.extra_env
        ready.assert_called_once()
        assert set(result.artifacts) == {"recording", "truth"}
    finally:
        env.stop()
    proc.stop.assert_called_once()
    store.stop.assert_called_once()


def test_truth_is_recorded_separately_and_flushed_before_grading(tmp_path, mocker):
    env = environment(tracked_bodies=("apple",), ready_streams=())
    env.scene = mocker.Mock(spec=SceneControlSpec)
    state = SceneState(
        world_id="world",
        scene_id="table",
        generation=0,
        tick=0,
        sim_time=0,
        ts=time.time(),
        entities={
            "apple": EntityState(
                pose=Pose(0.4, 0, 0.1),
                velocity=(0, 0, 0),
                angular_velocity=(0, 0, 0),
                bounds_min=(0, 0, 0),
                bounds_max=(1, 1, 1),
            )
        },
        robots={},
        joints={},
        regions={},
        contacts=(),
    )
    env.scene.scene_state.return_value = state
    with MemoryStore() as observations:
        try:
            artifacts = env.prepare_recording(
                observations, tmp_path / "memory.db", time.monotonic() + 2
            )
            moved = state.entities["apple"].model_copy(update={"pose": Pose(0.4, 0, 0.2)})
            env.scene.scene_state.return_value = state.model_copy(
                update={"entities": {"apple": moved}, "ts": state.ts + 1}
            )
            assert "tf" not in observations.streams
        finally:
            env.stop()
    assert not env._truth_thread.is_alive()
    outcome = Outcome(trajectory=mocker.Mock(), artifacts=artifacts)
    with truth_recording(outcome) as store:
        assert last_body_transform(store, "apple").translation.z == pytest.approx(0.2)
    assert lifted("apple", by_m=0.1)(outcome) == pytest.approx(1)
    with pytest.raises(KeyError, match="truth"):
        lifted("apple", by_m=0.1)(
            Outcome(trajectory=mocker.Mock(), artifacts={"recording": tmp_path / "memory.db"})
        )


def test_truth_recorder_failure_is_not_silently_graded(tmp_path, mocker):
    env = environment(ready_streams=())
    env.scene = mocker.Mock(spec=SceneControlSpec)
    env.scene.scene_state.side_effect = RuntimeError("world stopped")
    env._truth_stop = mocker.Mock()
    env._truth_stop.wait.return_value = False
    with MemoryStore() as store:
        env._truth = store
        env._record_truth_loop()
        with pytest.raises(RuntimeError, match="truth recording failed"):
            env._stop_truth()


@pytest.mark.self_hosted
@pytest.mark.parametrize(
    "scene",
    [
        "xarm7",
        "robosuite/lift",
        "robosuite/door",
        "robosuite/pick_place",
        "robosuite/stack",
        "robosuite/tool_hang",
        "robosuite/nut_assembly",
    ],
)
def test_exported_eval_world_composes_one_arm_and_supports_reset(scene):
    path = LfsPath(scene) / "world.xml"
    robot = replace(xarm_definition(7), sensors=())
    instance = scene_robot(path, robot, "workbench", default=(0, 0, 0))
    with SimulationRuntime(WorldConfig(scene=path, robots={"arm": instance}), "eval-test") as world:
        assert world.model.body("arm/link_base").id > 0
        with pytest.raises(KeyError):
            world.model.body("link_base")
        assert world.model.body("arm/link_base").pos == pytest.approx(instance.xyz)
        assert len(world.robots) == 1
        assert np.isfinite(world.data.qpos).all()
        initial = world.scene_state()
        movable = next(
            (name for name, entity in world.description.entities.items() if entity.movable),
            None,
        )
        if movable is not None:
            pose = initial.entities[movable].pose
            moved = Pose(pose.position + Vector3(0, 0, 0.05), pose.orientation)
            result = world.set_scene_state(SceneUpdate(poses={movable: moved}))
            assert result.entities[movable].pose.position.z == pytest.approx(pose.position.z + 0.05)
        for _ in range(20):
            world.step()
        assert np.isfinite(world.data.qpos).all()
        restored = world.reset()
        assert restored.entities == initial.entities
        assert world.data.time == 0
        assert world.model.nq >= len(robot.joints)
