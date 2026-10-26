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

"""Explicit-clock communication conditions, independent of transports and robots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["to_robot", "to_operator"]
T = TypeVar("T")
JITTER_STEP_S = 0.01


class Delay(BaseModel):
    """Additional one-way delay; zero jitter is fixed delay.

    Jitter is a stationary Gaussian process on a 10 ms grid. Its correlation
    decays exponentially with this time constant. Clipping at zero and reliable
    delivery ordering can change the final delay distribution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    base_s: float = Field(default=0.0, ge=0)
    jitter_std_s: float = Field(default=0.0, ge=0)
    correlation_s: float = Field(default=0.25, gt=0)


class Stall(BaseModel):
    """Prevent delivery during [at_s, at_s + duration_s) of the link timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    at_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    direction: Literal["to_robot", "to_operator", "both"] = "both"


class LinkConditions(BaseModel):
    """Opt-in application delivery experiment; limits cover the entire link."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    to_robot: Delay = Field(default_factory=Delay)
    to_operator: Delay = Field(default_factory=Delay)
    seed: int = 0
    stalls: tuple[Stall, ...] = ()
    max_pending_messages: int = Field(default=256, gt=0)
    max_pending_bytes: int = Field(default=32 * 1024 * 1024, gt=0)


class DelayTimeline:
    """Sample one direction in nondecreasing elapsed time, without a host clock.

    Random numbers are consumed per time step, never per message. Another camera
    or polling rate therefore cannot change the seeded condition timeline.
    """

    def __init__(self, conditions: LinkConditions, direction: Direction) -> None:
        self.profile = conditions.to_robot if direction == "to_robot" else conditions.to_operator
        self._random = random.Random(2 * conditions.seed + (direction == "to_operator"))
        self._index = 0
        self._value = self._random.gauss(0, self.profile.jitter_std_s)
        self._rho = math.exp(-JITTER_STEP_S / self.profile.correlation_s)
        self._innovation = self.profile.jitter_std_s * math.sqrt(1 - self._rho**2)
        self._stalls = sorted(
            (s for s in conditions.stalls if s.direction in (direction, "both")),
            key=lambda s: s.at_s,
        )

    def delay_at(self, elapsed_s: float) -> float:
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        index = math.floor(elapsed_s / JITTER_STEP_S)
        if index < self._index:
            raise ValueError("link timeline cannot move backwards")
        if self.profile.jitter_std_s:
            for _ in range(self._index, index):
                self._value = self._rho * self._value + self._random.gauss(0, self._innovation)
        self._index = index
        due = elapsed_s + max(0.0, self.profile.base_s + self._value)
        for stall in self._stalls:
            if stall.at_s <= due < stall.at_s + stall.duration_s:
                due = stall.at_s + stall.duration_s
        return due - elapsed_s


class LinkOverflowError(RuntimeError):
    """The experiment cannot retain more work without exceeding its budget."""


@dataclass(frozen=True)
class Delivery(Generic[T]):
    value: T
    arrival: float
    due: float
    requested_delay_s: float
    size: int

    @property
    def ordering_wait_s(self) -> float:
        return max(0.0, self.due - self.arrival - self.requested_delay_s)


class DeliveryQueue(Generic[T]):
    """Bounded FIFO deadlines. The caller owns waiting, delivery and cancellation."""

    def __init__(self, *, max_messages: int, max_bytes: int) -> None:
        if min(max_messages, max_bytes) <= 0:
            raise ValueError("delivery limits must be positive")
        self._max_messages, self._max_bytes = max_messages, max_bytes
        self._pending: deque[Delivery[T]] = deque()
        self.pending_bytes = 0
        self._last_arrival = -math.inf
        self._last_due = -math.inf

    def push(self, value: T, *, now: float, delay_s: float, size: int) -> None:
        if not math.isfinite(now) or now < self._last_arrival:
            raise ValueError("arrival time must be finite and nondecreasing")
        if not math.isfinite(delay_s) or delay_s < 0 or size < 0:
            raise ValueError("delay and size must be nonnegative and finite")
        if len(self._pending) >= self._max_messages or self.pending_bytes + size > self._max_bytes:
            raise LinkOverflowError("link delivery buffer limit exceeded")
        due = max(now + delay_s, self._last_due)
        if not math.isfinite(due):
            raise ValueError("delivery deadline must be finite")
        self._pending.append(Delivery(value, now, due, delay_s, size))
        self.pending_bytes += size
        self._last_arrival, self._last_due = now, due

    @property
    def next_due(self) -> float | None:
        return self._pending[0].due if self._pending else None

    def pop_ready(self, now: float) -> Delivery[T] | None:
        if self._pending and self._pending[0].due <= now:
            delivery = self._pending.popleft()
            self.pending_bytes -= delivery.size
            return delivery
        return None

    def clear(self) -> tuple[int, int]:
        """Cancel queued work and return the discarded message/byte counts."""
        counts = len(self._pending), self.pending_bytes
        self._pending.clear()
        self.pending_bytes = 0
        self._last_arrival = self._last_due = -math.inf
        return counts
