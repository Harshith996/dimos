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

"""
MovementManager: click-to-goal relay + teleop/nav velocity mux.

NOTE: this should be majorly updated/reworked when mustafa's trajectory controller lands
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.spec import NavigationInterfaceSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# without this you can (basically) click into infinity in rerun (not good for the planner)
MAX_CLICK_HORIZONTAL_M = 500.0
MAX_CLICK_VERTICAL_M = 50.0


class MovementManagerConfig(ModuleConfig):
    tele_cooldown_sec: float = 1.0
    tele_cmd_vel_scaling: Twist = Twist(Vector3(1, 1, 1), Vector3(1, 1, 1))


class MovementManager(Module):
    """Combine tele_cmd_vel (keyboard controls) and nav_cmd_vel in a sane way, output cmd_vel"""

    config: MovementManagerConfig

    _navigation: NavigationInterfaceSpec | None = None

    clicked_point: In[PointStamped]
    nav_cmd_vel: In[Twist]
    tele_cmd_vel: In[Twist]

    way_point: Out[PointStamped]
    cmd_vel: Out[Twist]
    stop_movement: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._teleop_active = False
        self._last_teleop_time = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.clicked_point.subscribe(self._on_click)))
        self.register_disposable(Disposable(self.nav_cmd_vel.subscribe(self._on_nav)))
        self.register_disposable(Disposable(self.tele_cmd_vel.subscribe(self._on_teleop)))

    @rpc
    def stop(self) -> None:
        with self._lock:
            self._teleop_active = False
        super().stop()

    def _on_click(self, msg: PointStamped) -> None:
        if not all(math.isfinite(v) for v in (msg.x, msg.y, msg.z)):
            logger.warning("Ignored invalid click", x=msg.x, y=msg.y, z=msg.z)
            return
        if (
            abs(msg.x) > MAX_CLICK_HORIZONTAL_M
            or abs(msg.y) > MAX_CLICK_HORIZONTAL_M
            or abs(msg.z) > MAX_CLICK_VERTICAL_M
        ):
            logger.warning("Ignored out-of-range click", x=msg.x, y=msg.y, z=msg.z)
            return

        logger.debug("Goal", x=round(msg.x, 1), y=round(msg.y, 1), z=round(msg.z, 1))
        self.way_point.publish(msg)
        if self._navigation is not None:
            self._navigation.set_goal(msg.to_pose_stamped())

    def _cancel_goal(self) -> None:
        self.stop_movement.publish(Bool(data=True))
        cancel = PointStamped(
            ts=time.time(), frame_id="map", x=float("nan"), y=float("nan"), z=float("nan")
        )
        self.way_point.publish(cancel)
        if self._navigation is not None:
            # Off the teleop thread, so a planner that does not answer cannot hold up cmd_vel.
            threading.Thread(
                target=self._cancel_navigation, args=(self._navigation,), daemon=True
            ).start()
        logger.debug("Navigation cancelled — waiting for new goal")

    def _cancel_navigation(self, navigation: NavigationInterfaceSpec) -> None:
        try:
            navigation.cancel_goal()
        except Exception:
            logger.exception("Failed to cancel the navigation goal")

    def _on_nav(self, msg: Twist) -> None:
        with self._lock:
            if self._teleop_active:
                # check if cooldown has expired
                elapsed = time.monotonic() - self._last_teleop_time
                if elapsed < self.config.tele_cooldown_sec:
                    return
                self._teleop_active = False
            self.cmd_vel.publish(msg)

    def _on_teleop(self, msg: Twist) -> None:
        with self._lock:
            now = time.monotonic()
            starting = now - self._last_teleop_time >= self.config.tele_cooldown_sec
            self._teleop_active = True
            self._last_teleop_time = now

        scale = self.config.tele_cmd_vel_scaling
        scaled = Twist(
            linear=Vector3(
                msg.linear.x * scale.linear.x,
                msg.linear.y * scale.linear.y,
                msg.linear.z * scale.linear.z,
            ),
            angular=Vector3(
                msg.angular.x * scale.angular.x,
                msg.angular.y * scale.angular.y,
                msg.angular.z * scale.angular.z,
            ),
        )
        self.cmd_vel.publish(scaled)
        if starting:
            self._cancel_goal()
