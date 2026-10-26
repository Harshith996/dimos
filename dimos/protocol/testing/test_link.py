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

import math
import statistics

from pydantic import ValidationError
import pytest

from dimos.protocol.testing.link import (
    Delay,
    DelayTimeline,
    DeliveryQueue,
    LinkConditions,
    LinkOverflowError,
    Stall,
)


def test_fixed_delay_shifts_stream_without_throttling_or_restamping():
    queue = DeliveryQueue(max_messages=10, max_bytes=100)
    timeline = DelayTimeline(LinkConditions(to_robot=Delay(base_s=0.1)), "to_robot")
    for now in (0, 0.02, 0.04):
        queue.push({"ts": now}, now=now, delay_s=timeline.delay_at(now), size=10)
    assert queue.pop_ready(0.099) is None
    deliveries = [queue.pop_ready(t) for t in (0.101, 0.121, 0.141)]
    assert [d.value["ts"] for d in deliveries] == [0, 0.02, 0.04]
    assert [d.due for d in deliveries] == pytest.approx([0.1, 0.12, 0.14])
    assert queue.pending_bytes == 0


def test_later_message_waits_behind_earlier_message():
    queue = DeliveryQueue(max_messages=3, max_bytes=100)
    queue.push("grab", now=0, delay_s=0.15, size=4)
    queue.push("release", now=0.02, delay_s=0.05, size=7)
    assert queue.pop_ready(0.07) is None
    assert queue.pop_ready(0.15).value == "grab"
    release = queue.pop_ready(0.15)
    assert release.value == "release"
    assert release.ordering_wait_s == pytest.approx(0.08)


@pytest.mark.parametrize("max_messages,max_bytes", [(1, 100), (10, 4)])
def test_overflow_is_explicit_and_cancel_removes_old_work(max_messages, max_bytes):
    queue = DeliveryQueue(max_messages=max_messages, max_bytes=max_bytes)
    queue.push("old", now=1, delay_s=1, size=4)
    with pytest.raises(LinkOverflowError):
        queue.push("release", now=1.01, delay_s=0, size=1)
    assert queue.clear() == (1, 4)
    assert queue.pop_ready(10) is None
    queue.push("new", now=0, delay_s=0, size=1)
    assert queue.pop_ready(0).value == "new"


def test_seeded_timeline_is_independent_of_polling_rate_and_direction():
    profile = Delay(base_s=0.1, jitter_std_s=0.02, correlation_s=0.25)
    conditions = LinkConditions(to_robot=profile, to_operator=profile, seed=7)
    dense = DelayTimeline(conditions, "to_robot")
    dense_values = [dense.delay_at(i / 100) for i in range(501)]
    sparse = DelayTimeline(conditions, "to_robot")
    assert [sparse.delay_at(i) for i in range(6)] == [dense_values[i * 100] for i in range(6)]
    other_direction = DelayTimeline(conditions, "to_operator")
    assert other_direction.delay_at(5) != dense_values[-1]


def test_jitter_has_configured_spread_and_time_correlation():
    timeline = DelayTimeline(
        LinkConditions(to_robot=Delay(base_s=0.1, jitter_std_s=0.02, correlation_s=0.25), seed=7),
        "to_robot",
    )
    values = [timeline.delay_at(i / 100) for i in range(20000)]
    assert statistics.mean(values) == pytest.approx(0.1, abs=0.004)
    assert statistics.pstdev(values) == pytest.approx(0.02, abs=0.003)
    assert statistics.correlation(values[:-1], values[1:]) == pytest.approx(
        math.exp(-0.01 / 0.25), abs=0.015
    )


def test_jitter_never_produces_negative_delay():
    timeline = DelayTimeline(LinkConditions(to_robot=Delay(jitter_std_s=1), seed=7), "to_robot")
    values = [timeline.delay_at(i / 10) for i in range(100)]
    assert min(values) == 0
    assert max(values) > 0


def test_stall_catches_already_pending_deliveries_and_overlapping_stalls():
    conditions = LinkConditions(
        to_robot=Delay(base_s=0.15),
        stalls=(
            Stall(at_s=0.1, duration_s=0.3, direction="to_robot"),
            Stall(at_s=0.3, duration_s=0.2, direction="to_robot"),
        ),
    )
    timeline = DelayTimeline(conditions, "to_robot")
    assert timeline.delay_at(0) == pytest.approx(0.5)
    assert timeline.delay_at(0.3) == pytest.approx(0.2)
    assert timeline.delay_at(0.5) == pytest.approx(0.15)
    assert DelayTimeline(conditions, "to_operator").delay_at(0.2) == 0


@pytest.mark.parametrize("value", [-1, math.inf, math.nan])
def test_invalid_timing_is_rejected_before_starting(value):
    with pytest.raises(ValidationError):
        Delay(base_s=value)
    with pytest.raises(ValidationError):
        Stall(at_s=0, duration_s=value)
