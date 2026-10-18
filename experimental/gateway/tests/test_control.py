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

from unittest.mock import call

import pytest

from dimos.core.coordination.module_coordinator import ModuleDescriptor
from dimos.core.run_registry import RunEntry
from experimental.gateway import control


@pytest.fixture
def controller(state, mocker):
    runs = [RunEntry("run-1", 1, "unitree-g1-sonic-wbc", "now", "/tmp/log")]
    mocker.patch.object(control, "list_runs", return_value=runs)
    raw = {
        "active": True,
        "control_state": "unarmed",
        "dry_run": True,
        "fault_reason": None,
        "feedback_received": True,
        "feedback_age_ms": 5.0,
        "imu_age_ms": 5.0,
        "feedback_timeout_ms": 500.0,
        "model": "sonic_v1_1",
    }
    motor = {
        "fault_reason": None,
        "feedback_age_ms": 4.0,
        "latest_command_age_ms": 15.0,
        "configured_rate_hz": 500.0,
    }
    descriptors = [
        ModuleDescriptor(
            "G1SonicPolicyModule",
            "x.G1SonicPolicyModule",
            ["activate", "status", "disarm"],
            "policy",
        ),
        ModuleDescriptor(
            "MovementManager", "x.MovementManager", ["stop_motion", "movement_status"], "movement"
        ),
        ModuleDescriptor(
            "G1WholeBodyConnection", "x.G1WholeBodyConnection", ["set_estop"], "hardware"
        ),
    ]

    def answer(name, **kwargs):
        if name == "Coordinator/list_modules":
            return descriptors
        if name == "policy/status":
            return raw
        if name == "hardware/command_stream_status":
            return motor
        if name == "movement/movement_status":
            return {"source": "teleoperation"}
        if name == "policy/activate":
            raw.update(arm_pending=True, dry_run=False)
            return True
        if name == "hardware/set_estop":
            motor["fault_reason"] = "operator stop"
            return True
        if name == "movement/stop_motion":
            return None
        raise AssertionError(f"Unexpected robot call: {name}")

    bus = mocker.Mock(spec=control.ControlBus)
    bus.call.side_effect = answer
    state.controls = control.Controls(bus)
    return bus, raw, motor, runs


def request(action):
    return {"run_id": "run-1", "controller_id": "policy", "action": action}


def test_status_is_read_only_and_activation_uses_the_atomic_rpc(client, controller):
    bus, _, _, _ = controller
    status = client.get("/control").json()
    assert status["phase"] == "pose-hold"
    assert status["can_activate"] is True
    assert not any(
        c.args[0] in ("policy/activate", "movement/stop_motion") for c in bus.call.call_args_list
    )

    activated = client.post("/control/actions", json=request("activate"))
    assert activated.status_code == 200
    assert activated.json()["phase"] == "activating"
    assert activated.json()["dry_run"] is False
    assert bus.call.call_args_list.count(call("policy/activate")) == 1


def test_old_browser_run_cannot_activate_a_replacement_stack(client, controller):
    bus, _, _, runs = controller
    runs[0].run_id = "run-2"

    response = client.post("/control/actions", json=request("activate"))

    assert response.status_code == 409
    bus.call.assert_not_called()


@pytest.mark.parametrize("failure", ["fault", "stale", "dry-run"])
def test_activation_refused_without_actuation_when_not_ready(client, controller, failure):
    bus, raw, motor, _ = controller
    if failure == "fault":
        motor["fault_reason"] = "feedback timeout"
    elif failure == "stale":
        raw["imu_age_ms"] = 700.0
    else:
        raw["control_state"] = "control"

    response = client.post("/control/actions", json=request("activate"))

    assert response.status_code == 409
    assert call("policy/activate") not in bus.call.call_args_list
    assert call("movement/stop_motion") not in bus.call.call_args_list


def test_damping_stop_reaches_motor_writer_before_policy_status(client, controller):
    bus, _, _, _ = controller

    response = client.post("/control/actions", json=request("damping-stop"))

    assert response.status_code == 200
    calls = bus.call.call_args_list
    assert calls.index(call("hardware/set_estop", estopped=True)) < calls.index(
        call("policy/status")
    )
    assert response.json()["phase"] == "fault"
    assert response.json()["can_activate"] is False


def test_stop_walking_keeps_policy_active_and_cancels_navigation(client, controller):
    bus, raw, _, _ = controller
    raw.update(control_state="control", dry_run=False)

    response = client.post("/control/actions", json=request("stop-motion"))

    assert response.status_code == 200
    assert response.json()["phase"] == "balancing"
    assert call("movement/stop_motion") in bus.call.call_args_list
    assert call("policy/disarm") not in bus.call.call_args_list


def test_status_timeout_is_unavailable_instead_of_healthy(client, controller):
    bus, _, _, _ = controller
    bus.call.side_effect = TimeoutError("offline")

    response = client.get("/control")

    assert response.status_code == 503
    assert "unknown" in response.json()["error"]


def test_unknown_actions_are_rejected_before_contacting_robot(client, controller):
    bus, _, _, _ = controller
    response = client.post("/control/actions", json=request("reset-fault"))
    assert response.status_code == 400
    bus.call.assert_not_called()


def test_world_scene_composition_keeps_sonic_controls(client, controller):
    _, _, _, runs = controller
    runs[0].blueprint = "unitree-g1-sonic-wbc-workshop.scene-data"
    runs[0].cli_args = ["unitree-g1-sonic-wbc", "workshop.scene-data"]

    response = client.get("/control")

    assert response.status_code == 200
    assert response.json()["can_activate"] is True
    assert response.json()["controller"]["name"] == "SONIC"
