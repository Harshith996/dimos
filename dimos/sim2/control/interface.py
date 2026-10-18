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

"""Device channel addressing and complete control frames, independent of Modules."""

from __future__ import annotations

import hashlib
import time

from dimos.sim2.ipc.abi import ChannelDescriptor, make_channel_descriptor
from dimos.sim2.ipc.channel import ChannelFrame, RobotChannel
from dimos.sim2.spec import ControlInterface


def descriptor(address: str, interface: ControlInterface, dof: int) -> ChannelDescriptor:
    sim_id, robot_id = address.split("/", 1)
    return make_channel_descriptor(
        sim_id=sim_id,
        robot_id=robot_id,
        generation=sim_id,
        shm_name="dms2_" + hashlib.sha256(address.encode()).hexdigest()[:24],
        control_interface=interface,
        dof=dof,
        physics_dt=0.005,
        control_decimation=1,
    )


def connect(address: str, interface: ControlInterface, dof: int) -> RobotChannel:
    channel = RobotChannel.attach(descriptor(address, interface, dof))
    deadline = time.monotonic() + 30.0
    while channel.lifecycle == "starting":
        if time.monotonic() >= deadline:
            channel.close()
            raise TimeoutError(f"sim2 device {address!r} did not become ready")
        time.sleep(0.01)
    if channel.lifecycle != "ready":
        state = channel.lifecycle
        channel.close()
        raise RuntimeError(f"sim2 device {address!r} is {state}")
    return channel


def observation(channel: RobotChannel) -> ChannelFrame:
    if channel.lifecycle != "ready":
        raise RuntimeError(f"sim2 device is {channel.lifecycle}")
    frame = channel.read_observation()
    if frame is None:
        raise RuntimeError("sim2 device has no initial observation")
    return frame
