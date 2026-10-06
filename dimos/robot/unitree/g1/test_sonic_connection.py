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

from dimos.control.sonic.sonic_pipeline import DEFAULT_ANGLES_DDS
from dimos.control.sonic.sonic_safety import damping_commands
from dimos.hardware.whole_body.spec import IMUState, MotorState
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.robot.unitree.g1.sonic_connection import G1SonicConnection


@pytest.fixture
def connection(mocker, tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    pipeline = mocker.patch("dimos.robot.unitree.g1.sonic_connection.SonicPipeline").return_value
    pipeline.step.return_value = np.ones(29, dtype=np.float32)
    pipeline.snapshot.return_value = {"stream_active": False}
    adapter = mocker.patch(
        "dimos.robot.unitree.g1.sonic_connection.SimMujocoG1WholeBodyAdapter"
    ).return_value
    adapter.read_motor_states.return_value = [MotorState()] * 29
    adapter.read_imu.return_value = IMUState()
    c = G1SonicConnection(
        encoder_onnx=model,
        decoder_onnx=model,
        planner_onnx=model,
        simulation_address=tmp_path / "sim.xml",
        auto_dry_run=False,
    )
    # Clock ticks are explicit in these tests; ONNX and simulator IO are the boundaries.
    mocker.patch.object(c, "_run")
    mocker.patch.object(c, "_auto_bind_handlers")
    mocker.patch.object(c.joint_state, "publish")
    mocker.patch.object(c.imu, "publish")
    yield c
    c.stop()


def _step(connection, t_now, positions=0.0):
    connection._adapter.read_motor_states.return_value = [MotorState(q=positions)] * 29
    return connection._step(t_now)


def _arm(connection):
    connection.start()
    connection.arm()
    _step(connection, 0.0)
    return _step(connection, 3.0)


def test_hold_and_measured_pose_ramp_precede_policy_output(connection):
    connection.start()
    hold = _step(connection, 9.0, positions=0.25)
    assert hold == pytest.approx([0.25] * 29)

    connection.arm()
    first = _step(connection, 10.0, positions=0.4)
    halfway = _step(connection, 11.5, positions=0.4)
    complete = _step(connection, 13.0, positions=0.4)

    assert first == pytest.approx([0.4] * 29)
    assert halfway == pytest.approx((0.4 + 0.5 * (DEFAULT_ANGLES_DDS - 0.4)).tolist())
    assert complete == pytest.approx(DEFAULT_ANGLES_DDS.tolist())
    connection._policy.step.assert_not_called()
    assert _step(connection, 13.02) == [1.0] * 29


def test_dry_run_republishes_hold_while_policy_runs(connection):
    ramp = _arm(connection)
    connection.set_dry_run(True)

    first = _step(connection, 3.02)
    second = _step(connection, 3.04)

    assert first == second == ramp
    assert first != [1.0] * 29
    assert connection._policy.step.call_count == 2


def test_runtime_reset_clears_velocity_and_repeats_the_measured_pose_ramp(connection):
    _arm(connection)
    connection.set_velocity_command(0.5, 0.0, 0.0, t_now=3.0)
    _step(connection, 3.02)
    connection._policy.set_velocity.assert_called_with(0.5, 0.0, 0.0)

    assert connection.reset_runtime_state()
    first = _step(connection, 4.0, positions=0.4)
    complete = _step(connection, 7.0, positions=0.4)
    live = _step(connection, 7.02)

    assert first == pytest.approx([0.4] * 29)
    assert complete == pytest.approx(DEFAULT_ANGLES_DDS.tolist())
    assert live == [1.0] * 29
    connection._policy.set_velocity.assert_called_with(0.0, 0.0, 0.0)


def test_disarm_holds_measured_pose_until_explicit_rearm(connection):
    _arm(connection)
    _step(connection, 3.02)
    connection._policy.step.reset_mock()

    assert connection.disarm()
    assert _step(connection, 4.0, positions=0.6) == pytest.approx([0.6] * 29)
    assert _step(connection, 7.0, positions=0.6) == pytest.approx([0.6] * 29)
    assert connection.state_snapshot()["control_state"] == "unarmed"
    connection._policy.step.assert_not_called()

    connection.arm(ramp_seconds=0.0)
    _step(connection, 8.0, positions=0.6)
    assert _step(connection, 8.02) == [1.0] * 29
    assert connection.state_snapshot()["control_state"] == "control"


def test_velocity_timeout_stops_motion_but_keeps_balancing(connection):
    _arm(connection)
    connection.set_velocity_command(0.5, 0.0, 0.0, t_now=3.0)
    _step(connection, 3.5)
    connection._policy.set_velocity.assert_called_with(0.5, 0.0, 0.0)

    assert _step(connection, 4.02) == [1.0] * 29
    connection._policy.set_velocity.assert_called_with(0.0, 0.0, 0.0)
    assert connection.policy_active


@pytest.mark.parametrize(
    "operator_stop, reason", [(False, "decoder failed"), (True, "operator stop")]
)
def test_fault_during_inference_discards_targets_and_stays_latched(
    connection, operator_stop, reason
):
    _arm(connection)

    def inference(**kwargs):
        if operator_stop:
            connection.set_estop(True)
            return np.ones(29, dtype=np.float32)
        raise RuntimeError("decoder failed")

    connection._policy.step.side_effect = inference

    assert _step(connection, 3.02) is None
    assert _step(connection, 3.04, positions=0.4) is None
    assert connection.joint_state.publish.call_args.args[0].position == pytest.approx([0.4] * 29)
    assert connection.fault_reason == reason
    connection._adapter.write_motor_commands.assert_called_with(damping_commands(29))
    connection._policy.step.assert_called_once()
    assert connection.reset_runtime_state(reactivate=True) is False
    with pytest.raises(RuntimeError, match="restart"):
        connection.arm()
    with pytest.raises(RuntimeError, match="restart"):
        connection.set_estop(False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, reason", [("operator", "operator stop"), ("hardware", "stale feedback")]
)
async def test_hardware_faults_do_not_trigger_motor_takeover(connection, mocker, source, reason):
    # Before startup there is no simulator adapter: the real DDS connection owns the stop.
    publish = mocker.patch.object(connection.sonic_fault, "publish")
    motor_publish = mocker.patch.object(connection.motor_command, "publish")
    if source == "operator":
        connection.set_estop(True)
    else:
        await connection.handle_g1_fault(String(reason))
    assert connection.fault_reason == reason
    assert publish.call_args.args[0].data == reason
    motor_publish.assert_not_called()


@pytest.mark.asyncio
async def test_hardware_feedback_drives_commands_and_staleness_latches_stop(
    connection, mocker, monkeypatch
):
    monkeypatch.setattr(connection.config, "simulation_address", None)
    publish = mocker.patch.object(connection.motor_command, "publish")
    fault = mocker.patch.object(connection.sonic_fault, "publish")
    connection.start()
    await connection.handle_motor_states(
        JointState(position=[0.4] * 29, velocity=[0.0] * 29, effort=[0.0] * 29)
    )
    await connection.handle_low_level_imu(Imu(orientation=Quaternion(0, 0, 0, 1)))
    now = max(connection._feedback_at, connection._imu_at)
    connection.arm(ramp_seconds=0.0)
    connection._step(now)
    connection._step(now + 0.02)

    assert publish.call_args.args[0].q == [1.0] * 29
    np.testing.assert_array_equal(
        connection._policy.step.call_args.kwargs["base_quat_wxyz"], [1, 0, 0, 0]
    )
    assert connection.joint_state.publish.call_args.args[0].position == [0.4] * 29
    publish.reset_mock()

    assert connection._step(now + connection.config.feedback_timeout) is None
    assert fault.call_args.args[0].data == "SONIC feedback timeout"
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_standard_inputs_drive_policy_and_publish_measured_state(connection):
    _arm(connection)
    await connection.handle_base_command(
        Twist(linear=Vector3(0.3, 0.1, 0), angular=Vector3(0, 0, 0.2))
    )
    arm_name = connection.config.joint_names[15]
    await connection.handle_position_command(JointState(name=[arm_name], position=[0.7]))
    _step(connection, 3.02, positions=0.4)

    assert connection._policy.set_velocity.call_args.args == pytest.approx((0.3, 0.1, 0.2))
    expected = DEFAULT_ANGLES_DDS[15:].copy()
    expected[0] = 0.7
    np.testing.assert_allclose(connection._policy.set_upper_body.call_args.args[0], expected)
    feedback = connection.joint_state.publish.call_args.args[0]
    assert feedback.name == connection.config.joint_names
    assert feedback.position == pytest.approx([0.4] * 29)
    assert [m.q for m in connection._adapter.write_motor_commands.call_args.args[0]] == [1.0] * 29

    with pytest.raises(ValueError, match="only accepts arm"):
        await connection.handle_position_command(
            JointState(name=[connection.config.joint_names[0]], position=[0.7])
        )
    np.testing.assert_allclose(connection._policy.set_upper_body.call_args.args[0], expected)
