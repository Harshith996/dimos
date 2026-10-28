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

"""Typed recording inputs and dataset features declared together."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.imitation.collection.recording import RecordingSchema
from dimos.imitation.dataprep.schema import (
    DataPrepConfig,
    FeatureSpec,
    OutputConfig,
    QualityConfig,
    SyncConfig,
    validate_source_kinds,
)
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.protocol.service.spec import BaseConfig


class CollectionProfile(BaseConfig):
    """One recording contract, independent of camera hardware and policy backends."""

    name: str = Field(min_length=1)
    robot_type: str = Field(min_length=1)
    observations: dict[str, FeatureSpec] = Field(min_length=1)
    actions: dict[str, FeatureSpec] = Field(min_length=1)
    sync: SyncConfig
    quality: QualityConfig = QualityConfig()

    @field_validator("observations", "actions")
    @classmethod
    def _feature_names(cls, value: dict[str, FeatureSpec]) -> dict[str, FeatureSpec]:
        if any(not key.strip() for key in value):
            raise ValueError("feature names must not be blank")
        return value

    @model_validator(mode="after")
    def validate_features(self) -> CollectionProfile:
        if set(self.observations) & set(self.actions):
            raise ValueError("observation and action feature names must be distinct")
        if self.sync.anchor not in self.observations:
            raise ValueError("sync anchor must name an observation feature")
        self.input_types()
        return self

    def input_types(self) -> dict[str, type[Any]]:
        """Deduplicate raw inputs without merging their dataset projections."""
        validate_source_kinds((*self.observations.values(), *self.actions.values()))
        inputs: dict[str, type[Any]] = {}
        for feature in (*self.observations.values(), *self.actions.values()):
            kind = feature.message_type
            if kind is None:
                raise ValueError(f"Collection stream {feature.stream!r} requires message_type")
            if feature.source_kind == "joint_position_updates" and not issubclass(kind, JointState):
                raise ValueError("joint_position_updates requires JointState messages")
            previous = inputs.setdefault(feature.stream, kind)
            if previous is not kind:
                raise ValueError(f"stream {feature.stream!r} has conflicting message types")
        return inputs

    def to_schema(self) -> RecordingSchema:
        """Snapshot dataset interpretation without runtime message classes."""
        return RecordingSchema(
            name=self.name,
            robot_type=self.robot_type,
            observation={
                key: FeatureSpec(**feature.model_dump())
                for key, feature in self.observations.items()
            },
            action={
                key: FeatureSpec(**feature.model_dump()) for key, feature in self.actions.items()
            },
            sync=self.sync.model_copy(deep=True),
            quality=self.quality.model_copy(deep=True),
        )

    def dataprep_config(self, *, source: str = "", output: OutputConfig) -> DataPrepConfig:
        """Build a low-level preparation request for an explicit raw artifact."""
        return DataPrepConfig(
            **self.to_schema().model_dump(exclude={"name", "robot_type", "payload"}),
            source=source,
            output=output,
        )


ActionSource = Literal["measured_positions", "accepted_targets"]


def joint_rgb_profile(
    *,
    name: str,
    robot_type: str,
    joints: Sequence[str],
    cameras: Mapping[str, WebcamConfig],
    action_source: ActionSource,
    sync: SyncConfig | None = None,
    extra_observations: Mapping[str, FeatureSpec] | None = None,
) -> CollectionProfile:
    """Generate the standard contract without duplicating feature declarations."""
    if not joints or len(set(joints)) != len(joints) or any(not j.strip() for j in joints):
        raise ValueError("joints must contain unique non-empty names")
    if not cameras or any(not key.isidentifier() for key in cameras):
        raise ValueError("cameras must contain identifier names")
    if action_source not in ("measured_positions", "accepted_targets"):
        raise ValueError("action_source must be measured_positions or accepted_targets")
    observations = {
        f"observation.images.{key}": FeatureSpec(
            stream=f"{key}_image",
            message_type=Image,
            field="data",
            dtype="video",
            shape=(camera.height, camera.width, 3),
            names=["height", "width", "channels"],
        )
        for key, camera in cameras.items()
    }
    state = FeatureSpec(
        stream="coordinator_joint_state",
        message_type=JointState,
        field="position",
        dtype="float32",
        shape=(len(joints),),
        names=list(joints),
    )
    observations["observation.state"] = state
    for key, feature in (extra_observations or {}).items():
        if key in observations or key == "action":
            raise ValueError(f"Extra observation {key!r} replaces a standard feature")
        observations[key] = feature
    first_camera = next(iter(cameras))
    return CollectionProfile(
        name=name,
        robot_type=robot_type,
        observations=observations,
        actions={
            "action": state.model_copy(
                update={
                    "stream": "applied_joint_position_command"
                    if action_source == "accepted_targets"
                    else "coordinator_joint_state",
                    "source_kind": "joint_position_updates"
                    if action_source == "accepted_targets"
                    else "snapshot",
                },
                deep=True,
            ),
        },
        sync=sync
        if sync is not None
        else SyncConfig(
            anchor=f"observation.images.{first_camera}",
            rate_hz=cameras[first_camera].fps,
            tolerance_ms=20,
        ),
        quality=QualityConfig(mode="strict"),
    )
