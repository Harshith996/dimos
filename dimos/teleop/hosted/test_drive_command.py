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

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dimos.core.module import Module
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.teleop.hosted.drive_command import DriveCommandModule


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> DriveCommandModule:
    monkeypatch.setattr(Module, "__init__", lambda self, **kwargs: None)
    module = DriveCommandModule()
    module.config = SimpleNamespace(
        cmd_stale_after_sec=0.5, max_linear_mps=0.5, max_angular_rps=1.0
    )
    for port in ("cmd_ack", "cmd_vel", "robot_state"):
        setattr(module, port, MagicMock())
    return module


def _twist(ts: float, vx: float = 0.3, wz: float = 0.0) -> TwistStamped:
    return TwistStamped(ts=ts, linear=[vx, 0.0, 0.0], angular=[0.0, 0.0, wz])


def _sent(module: DriveCommandModule) -> list[Twist]:
    return [call.args[0] for call in module.cmd_vel.publish.call_args_list]


def test_drive_is_clamped_and_idle_zeros_go_out_once(module: DriveCommandModule) -> None:
    now = time.time()
    module._on_cmd_vel_in(_twist(now - 0.03, vx=2.0, wz=-5.0))
    module._on_cmd_vel_in(_twist(now - 0.02, vx=0.0))
    module._on_cmd_vel_in(_twist(now - 0.01, vx=0.0))
    sent = _sent(module)
    assert [(t.linear.x, t.angular.z) for t in sent] == [(0.5, -1.0), (0.0, 0.0)]


@pytest.mark.parametrize("age", [0.6, -0.1, float("nan")])
def test_stale_future_and_nan_stamps_are_dropped(module: DriveCommandModule, age: float) -> None:
    module._on_cmd_vel_in(_twist(time.time() - age))
    assert not _sent(module)


def test_estop_stops_and_blocks_drive_until_cleared(module: DriveCommandModule) -> None:
    module._on_state_json(json.dumps({"type": "estop", "nonce": 1}).encode())
    module._on_cmd_vel_in(_twist(time.time()))
    assert [t.is_zero() for t in _sent(module)] == [True]
    module._on_state_json(json.dumps({"type": "estop_clear", "nonce": 2}).encode())
    module._on_cmd_vel_in(_twist(time.time()))
    assert _sent(module)[-1].linear.x == pytest.approx(0.3)


def test_operator_lost_stops(module: DriveCommandModule) -> None:
    module._on_cmd_vel_in(_twist(time.time()))
    module._on_state_json(json.dumps({"type": "operator_lost"}).encode())
    assert _sent(module)[-1].is_zero()


def test_go2_only_commands_are_rejected(module: DriveCommandModule) -> None:
    module._on_state_json(json.dumps({"type": "sport_cmd", "nonce": 7}).encode())
    ack = json.loads(module.cmd_ack.publish.call_args.args[0])
    assert ack == {"type": "cmd_ack", "nonce": 7, "ok": False}
