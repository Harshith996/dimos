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

"""Standard RGB and joint-position collection composition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from dimos.control.coordinator import ControlCoordinator, ControlCoordinatorConfig
from dimos.core.coordination.blueprint_config.fields import module_config_cls
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.profile import ActionSource, joint_rgb_profile
from dimos.imitation.collection.recorder import collection_recorder
from dimos.imitation.dataprep.schema import FeatureSpec, SyncConfig


def collect(
    *,
    cameras: Mapping[str, WebcamConfig],
    control: Blueprint,
    action_source: ActionSource,
    robot_type: str | None = None,
    name: str,
    recording: Path | None = None,
    format: Literal["mcap", "sqlite"] = "mcap",
    sync: SyncConfig | None = None,
    extra_observations: Mapping[str, FeatureSpec] | None = None,
    episodes: Blueprint | None = None,
) -> Blueprint:
    """Compose cameras, recorder and episode controls around an existing robot graph.

    Hardware identity and joint order come from the sole control coordinator's
    configuration. No coordinator or hardware is instantiated here.
    """
    coordinators = [
        atom for atom in control.active_blueprints if issubclass(atom.module, ControlCoordinator)
    ]
    if len(coordinators) != 1:
        raise ValueError("Collection requires exactly one control coordinator")
    coordinator = coordinators[0]
    config = cast("ControlCoordinatorConfig", module_config_cls(coordinator)(**coordinator.kwargs))
    hardware = config.hardware
    if not hardware:
        raise ValueError("Collection requires hardware declared in the coordinator config")
    if robot_type is None:
        if len(hardware) != 1:
            raise ValueError("Multi-component collection requires an explicit robot_type")
        robot_type = hardware[0].hardware_id
    joints = [joint for component in hardware for joint in component.joints]
    if set(cameras) & ({"recorder", "episodes"} | {a.name for a in control.active_blueprints}):
        raise ValueError("Camera name conflicts with a collection module")
    profile = joint_rgb_profile(
        name=name,
        robot_type=robot_type,
        joints=joints,
        cameras=cameras,
        action_source=action_source,
        sync=sync,
        extra_observations=extra_observations,
    )
    camera_blueprints = [
        CameraModule.blueprint(
            instance_name=key,
            hardware=camera.model_copy(update={"frame_id_prefix": f"{key}_image"}),
            frame_id=f"{key}_camera_link",
        )
        for key, camera in cameras.items()
    ]
    return autoconnect(
        collection_recorder(profile=profile, recording=recording, format=format),
        episodes
        if episodes is not None
        else EpisodeMonitorModule.blueprint(instance_name="episodes"),
        control,
        *camera_blueprints,
    ).remappings(
        [
            (key, port, f"{key}_{suffix}")
            for key in cameras
            for port, suffix in (
                ("color_image", "image"),
                ("camera_info", "camera_info"),
                ("tf", "tf"),
            )
        ]
    )
