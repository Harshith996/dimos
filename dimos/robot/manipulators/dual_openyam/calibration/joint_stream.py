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

"""Joint angles from a running coordinator, for captures taken during teleop.

While a teleop session drives the arms its coordinator owns the CAN buses, so
captures listen to the joint states it publishes instead of querying the
motors. Listening sends nothing to the robot.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import threading
import time

import numpy as np

from dimos.core.transport import PubSubTransport
from dimos.core.transport_factory import make_transport
from dimos.msgs.sensor_msgs.JointState import JointState

JOINT_STATE_TOPIC = "coordinator_joint_state"
# Same stillness bar as the encoder reads: a couple of encoder counts.
_STILL_SPREAD_RAD = float(np.radians(0.1))


class JointStreamError(RuntimeError):
    """No usable joint states, or the arm was moving."""


class JointStream:
    def __init__(
        self,
        topic: str = JOINT_STATE_TOPIC,
        *,
        history: int = 4_000,
        transport: PubSubTransport[JointState] | None = None,
    ) -> None:
        self._topic = topic
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, dict[str, float]]] = deque(maxlen=history)
        self._transport = transport or make_transport(topic, JointState)
        self._unsubscribe = self._transport.subscribe(self._on_message)

    def _on_message(self, message: JointState) -> None:
        sample = dict(zip(message.name, (float(p) for p in message.position), strict=False))
        with self._lock:
            self._samples.append((time.monotonic(), sample))

    def capture(
        self,
        joint_names: Sequence[str],
        *,
        window_s: float = 0.5,
        timeout_s: float = 5.0,
    ) -> np.ndarray:
        """Mean of the named joints over the last ``window_s``; the arm must be still."""
        deadline = time.monotonic() + timeout_s
        start = time.monotonic()
        while True:
            now = time.monotonic()
            with self._lock:
                recent = [
                    sample
                    for stamp, sample in self._samples
                    if stamp >= max(start, now - window_s)
                    and all(name in sample for name in joint_names)
                ]
            if now - start >= window_s and len(recent) >= 5:
                break
            if now > deadline:
                raise JointStreamError(
                    f"no joint states for {', '.join(joint_names)} on {self._topic}; "
                    "is the teleop session running?"
                )
            time.sleep(0.02)
        values = np.array([[sample[name] for name in joint_names] for sample in recent])
        spread = float(np.ptp(values, axis=0).max())
        if spread > _STILL_SPREAD_RAD:
            raise JointStreamError(
                f"the arm moved during the capture ({np.degrees(spread):.2f} deg); "
                "let go of the controller, wait a second and capture again"
            )
        return values.mean(axis=0)

    def close(self) -> None:
        self._unsubscribe()
        self._transport.stop()
