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

"""Drive-only command plane for a base driven by a velocity stream (no driver RPCs).

The operator's drive twist passes the same E-STOP/stale/reorder/clamp guard as
``Go2CommandModule`` and goes out on ``cmd_vel``; E-STOP and a lost operator send
a zero twist. Discrete Go2 commands (sport, light, nav) are acked as rejected.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_UNSUPPORTED_KINDS = frozenset(
    {"sport_cmd", "set_mode", "obstacle_avoidance", "light", "nav_goal", "nav_cancel"}
)


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


class DriveCommandConfig(ModuleConfig):
    cmd_stale_after_sec: float = 0.5
    max_linear_mps: float = 0.5
    max_angular_rps: float = 1.0


class DriveCommandModule(Module):
    """Operator E-STOP and guarded drive twist → ``cmd_vel``."""

    config: DriveCommandConfig

    state_json: In[bytes]
    cmd_ack: Out[bytes]

    cmd_vel_in: In[TwistStamped]
    cmd_vel: Out[Twist]

    robot_state: Out[bytes]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._estopped = False
        self._last_cmd_ts = 0.0
        self._last_cmd_nonzero = False

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.state_json.subscribe(self._on_state_json)))
        self.register_disposable(Disposable(self.cmd_vel_in.subscribe(self._on_cmd_vel_in)))
        self._publish_robot_state()

    def _send_ack(self, nonce: Any, ok: bool) -> None:
        self.cmd_ack.publish(json.dumps({"type": "cmd_ack", "nonce": nonce, "ok": ok}).encode())

    def _on_state_json(self, data: Any) -> None:
        if isinstance(data, str):
            data = data.encode()
        if not data.startswith(b"{"):
            return
        try:
            msg = json.loads(data)
        except ValueError:
            logger.warning("state_reliable: malformed JSON: %r", data[:80])
            return
        kind = msg.get("type")
        if kind == "estop":
            self._estopped = True
            logger.warning("E-STOP latched by operator")
            self._stop()
            self._publish_robot_state()
            self._send_ack(msg.get("nonce"), True)
        elif kind == "estop_clear":
            self._estopped = False
            logger.warning("E-STOP cleared by operator")
            self._publish_robot_state()
            self._send_ack(msg.get("nonce"), True)
        elif kind == "operator_lost":  # synthetic, injected by the provider
            logger.warning("operator link lost: stopping motion")
            self._stop()
        elif kind in _UNSUPPORTED_KINDS:
            self._send_ack(msg.get("nonce"), False)

    def _stop(self) -> None:
        self._last_cmd_nonzero = False
        self.cmd_vel.publish(Twist())

    def _on_cmd_vel_in(self, twist: TwistStamped) -> None:
        if self._estopped:
            return
        ts = float(twist.ts)
        age = time.time() - ts
        # NaN ts fails every check; a future stamp would stall later commands.
        if not 0 <= age <= self.config.cmd_stale_after_sec or ts <= self._last_cmd_ts:
            return
        self._last_cmd_ts = ts
        values = (
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
        if not all(math.isfinite(v) for v in values):
            logger.warning("dropping non-finite cmd_vel")
            return
        lin, ang = self.config.max_linear_mps, self.config.max_angular_rps
        drive = Twist(
            linear=[_clamp(twist.linear.x, lin), _clamp(twist.linear.y, lin), 0.0],
            angular=[0.0, 0.0, _clamp(twist.angular.z, ang)],
        )
        # The base's velocity task zeroes itself after its timeout, so idle zeros
        # need only go out once, on release.
        moving = not drive.is_zero()
        if not moving and not self._last_cmd_nonzero:
            return
        self._last_cmd_nonzero = moving
        self.cmd_vel.publish(drive)

    def _publish_robot_state(self) -> None:
        self.robot_state.publish(json.dumps({"estopped": self._estopped}).encode())
