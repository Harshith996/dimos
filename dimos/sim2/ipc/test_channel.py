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
import struct
import threading
from uuid import uuid4

import numpy as np
import pytest

from dimos.sim2.ipc.channel import FrameMetadata, RobotChannel


@pytest.fixture
def channel(worlds, request):
    world = worlds()
    descriptor = replace(
        world.robots["left"].channel.descriptor,
        shm_name="dms2_" + uuid4().hex[:24],
        observation_slots=request.param,
    )
    with RobotChannel.create(descriptor) as owner:
        try:
            yield owner
        finally:
            owner.unlink()


@pytest.mark.parametrize(
    ("direction", "sequence_offset", "channel"),
    [("action", 24, 2), ("observation", 32, 2), ("observation", 32, 42)],
    indirect=["channel"],
)
def test_reader_keeps_committed_frame_until_writer_switches_slots(
    channel, direction, sequence_offset, mocker
):
    layout = getattr(channel.descriptor, direction + "_layout")
    publish = getattr(channel, "publish_" + direction)
    read = getattr(channel, "read_" + direction)
    old = {field.name: np.ones(field.shape) for field in layout.fields}
    new = {field.name: np.full(field.shape, 2) for field in layout.fields}
    publish(old, FrameMetadata(0, 1, 1, 0, 0.01))
    pending, resume, finished = threading.Event(), threading.Event(), threading.Event()
    pack = struct.pack_into

    def pause_before_commit(fmt, buffer, offset, *values):
        pack(fmt, buffer, offset, *values)
        if offset == sequence_offset:
            pending.set()
            assert resume.wait(5), "reader did not release the writer"

    def writer():
        publish(new, FrameMetadata(0, 1, 2, 0, 0.02))
        finished.set()

    mocker.patch("dimos.sim2.ipc.channel.struct.pack_into", side_effect=pause_before_commit)
    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        assert pending.wait(5), "writer did not reach commit boundary"
        frame = read(retries=1)
        assert frame.metadata.physics_tick == 1
        np.testing.assert_array_equal(frame.values["position"], old["position"])
    finally:
        resume.set()
        thread.join(5)
    assert finished.is_set()
    frame = read(retries=1)
    assert frame.metadata.physics_tick == 2
    np.testing.assert_array_equal(frame.values["position"], new["position"])


@pytest.mark.parametrize("channel", [2], indirect=True)
def test_overwritten_slot_is_reported_busy_not_returned(channel):
    values = {
        field.name: np.ones(field.shape) for field in channel.descriptor.observation_layout.fields
    }
    channel.publish_observation(values, FrameMetadata(0, 1, 1, 0, 0.01))
    active = struct.unpack_from("<I", channel._buffer, 16)[0]
    offset = (
        channel.descriptor.observation_offset
        + active * channel.descriptor.observation_layout.slot_size
    )
    struct.pack_into("<Q", channel._buffer, offset, 0)
    with pytest.raises(BlockingIOError, match="coherent observation"):
        channel.read_observation(retries=1)


@pytest.mark.parametrize("channel", [2], indirect=True)
def test_invalid_frame_does_not_replace_last_complete_command(channel):
    values = {field.name: np.ones(field.shape) for field in channel.descriptor.action_layout.fields}
    channel.publish_action(values, FrameMetadata(0, 1, 1, 0, 0.01))
    invalid = dict(values, position=[1])
    with pytest.raises(ValueError, match="shape"):
        channel.publish_action(invalid, FrameMetadata(0, 1, 2, 0, 0.02))
    frame = channel.read_action()
    assert frame.metadata.physics_tick == 1
    np.testing.assert_array_equal(frame.values["position"], [1, 1])


def test_failed_create_releases_shared_memory_name(worlds):
    descriptor = replace(
        worlds().robots["left"].channel.descriptor,
        shm_name="dms2_" + uuid4().hex[:24],
    )
    with pytest.raises(ValueError, match="unsupported sim2 ABI"):
        RobotChannel.create(replace(descriptor, abi_version=0))

    with RobotChannel.create(descriptor) as owner:
        try:
            assert owner.lifecycle == "starting"
        finally:
            owner.unlink()
