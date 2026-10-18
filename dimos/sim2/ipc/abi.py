# Copyright 2025-2026 Dimensional Inc.
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

"""Versioned shared-memory layouts for sim2 robot channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dimos.sim2.spec import ControlInterface

ABI_VERSION = 3
CHANNEL_MAGIC = b"DMSIM2\0\0"
CHANNEL_HEADER_SIZE = 64
FRAME_METADATA_SIZE = 48
ALIGNMENT = 64


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class FrameField:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        return int(np.dtype(self.dtype).itemsize * np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class FrameLayout:
    fields: tuple[FrameField, ...]
    slot_size: int

    def field(self, name: str) -> FrameField:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [asdict(field) for field in self.fields], "slot_size": self.slot_size}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrameLayout:
        return cls(
            fields=tuple(
                FrameField(
                    name=field["name"],
                    dtype=field["dtype"],
                    shape=tuple(field["shape"]),
                    offset=field["offset"],
                )
                for field in value["fields"]
            ),
            slot_size=value["slot_size"],
        )


@dataclass(frozen=True)
class ChannelDescriptor:
    abi_version: int
    sim_id: str
    robot_id: str
    generation: str
    shm_name: str
    control_interface: ControlInterface
    dof: int
    capabilities: tuple[str, ...]
    physics_dt: float
    control_decimation: int
    action_layout: FrameLayout
    observation_layout: FrameLayout
    observation_slots: int = 2

    def __post_init__(self) -> None:
        if self.observation_slots < 2:
            raise ValueError("observation channels require at least two slots")

    @property
    def action_offset(self) -> int:
        return CHANNEL_HEADER_SIZE

    @property
    def observation_offset(self) -> int:
        return self.action_offset + 2 * self.action_layout.slot_size

    @property
    def total_size(self) -> int:
        return self.observation_offset + self.observation_slots * self.observation_layout.slot_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "sim_id": self.sim_id,
            "robot_id": self.robot_id,
            "generation": self.generation,
            "shm_name": self.shm_name,
            "control_interface": self.control_interface.value,
            "dof": self.dof,
            "capabilities": list(self.capabilities),
            "physics_dt": self.physics_dt,
            "control_decimation": self.control_decimation,
            "action_layout": self.action_layout.to_dict(),
            "observation_layout": self.observation_layout.to_dict(),
            "observation_slots": self.observation_slots,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChannelDescriptor:
        return cls(
            abi_version=value["abi_version"],
            sim_id=value["sim_id"],
            robot_id=value["robot_id"],
            generation=value["generation"],
            shm_name=value["shm_name"],
            control_interface=ControlInterface(value["control_interface"]),
            dof=value["dof"],
            capabilities=tuple(value.get("capabilities", ())),
            physics_dt=value["physics_dt"],
            control_decimation=value["control_decimation"],
            action_layout=FrameLayout.from_dict(value["action_layout"]),
            observation_layout=FrameLayout.from_dict(value["observation_layout"]),
            observation_slots=value["observation_slots"],
        )


def make_channel_descriptor(
    *,
    sim_id: str,
    robot_id: str,
    generation: str,
    shm_name: str,
    control_interface: ControlInterface,
    dof: int,
    capabilities: tuple[str, ...] = (),
    physics_dt: float,
    control_decimation: int,
) -> ChannelDescriptor:
    action_fields, observation_fields = _interface_fields(control_interface, dof)
    return ChannelDescriptor(
        abi_version=ABI_VERSION,
        sim_id=sim_id,
        robot_id=robot_id,
        generation=generation,
        shm_name=shm_name,
        control_interface=control_interface,
        dof=dof,
        capabilities=capabilities,
        physics_dt=physics_dt,
        control_decimation=control_decimation,
        action_layout=_build_layout(action_fields),
        observation_layout=_build_layout(observation_fields),
    )


def _build_layout(fields: tuple[tuple[str, str, tuple[int, ...]], ...]) -> FrameLayout:
    offset = FRAME_METADATA_SIZE
    resolved: list[FrameField] = []
    for name, dtype, shape in fields:
        item_alignment = min(np.dtype(dtype).itemsize, 8)
        offset = _align(offset, item_alignment)
        field = FrameField(name=name, dtype=dtype, shape=shape, offset=offset)
        resolved.append(field)
        offset += field.size
    return FrameLayout(fields=tuple(resolved), slot_size=_align(offset))


def _interface_fields(
    interface: ControlInterface,
    dof: int,
) -> tuple[
    tuple[tuple[str, str, tuple[int, ...]], ...],
    tuple[tuple[str, str, tuple[int, ...]], ...],
]:
    vector = (dof,)
    if interface == ControlInterface.MANIPULATOR:
        return (
            (
                ("command_mode", "<i4", (1,)),
                ("enabled", "<u1", (1,)),
                ("position", "<f8", vector),
                ("velocity", "<f8", vector),
                ("effort", "<f8", vector),
                ("velocity_scale", "<f8", (1,)),
                ("gripper", "<f8", (1,)),
            ),
            (
                ("position", "<f8", vector),
                ("velocity", "<f8", vector),
                ("effort", "<f8", vector),
                ("gripper", "<f8", (1,)),
                ("enabled", "<u1", (1,)),
                ("error_code", "<i4", (1,)),
                ("wall_time", "<f8", (1,)),
            ),
        )
    return (
        (
            ("enabled", "<u1", (1,)),
            ("position", "<f8", vector),
            ("velocity", "<f8", vector),
            ("kp", "<f8", vector),
            ("kd", "<f8", vector),
            ("effort", "<f8", vector),
        ),
        (
            ("position", "<f8", vector),
            ("velocity", "<f8", vector),
            ("effort", "<f8", vector),
            ("imu_quaternion", "<f8", (4,)),
            ("imu_gyroscope", "<f8", (3,)),
            ("imu_accelerometer", "<f8", (3,)),
            ("imu_rpy", "<f8", (3,)),
            ("root_position", "<f8", (3,)),
            ("root_quaternion", "<f8", (4,)),
            ("root_linear_velocity", "<f8", (3,)),
            ("root_angular_velocity", "<f8", (3,)),
            ("enabled", "<u1", (1,)),
            ("wall_time", "<f8", (1,)),
        ),
    )
