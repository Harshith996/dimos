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

import platform
import sys

import pytest

# The tests extra excludes yourdfpy on Linux ARM because embreex has no wheel.
if sys.platform == "linux" and platform.machine() == "aarch64":
    pytest.importorskip(
        "yourdfpy", reason="yourdfpy is unavailable in the Linux ARM test environment"
    )

from dimos.core.coordination.blueprints import Blueprint
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.mapping.pointclouds.box_filter import PointCloudBoxFilter
from dimos.mapping.pointclouds.box_filter_spec import PointCloudBoxFilterSpec
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.robot.manipulators.xarm.blueprints.grasp import (
    xarm6_grasp,
    xarm6_grasp_graspgenx,
    xarm_grasp,
    xarm_grasp_graspgenx,
)
from dimos.robot.urdf_manager_point_clearing import UrdfManagerPointClearing
from dimos.spec.utils import spec_structural_compliance

_GRASP_BLUEPRINTS = [xarm_grasp, xarm_grasp_graspgenx, xarm6_grasp, xarm6_grasp_graspgenx]


def _topics(blueprint: Blueprint) -> dict[tuple[type, str], str]:
    """The name each module's stream connects on, keyed by (module, stream)."""
    return {
        (atom.module, stream.name): blueprint.remapping_map.get(
            (atom.name, stream.name), stream.name
        )
        for atom in blueprint.active_blueprints
        for stream in atom.streams
    }


@pytest.mark.parametrize("blueprint", _GRASP_BLUEPRINTS)
def test_the_planner_map_passes_through_the_box_filter(blueprint: Blueprint) -> None:
    topics = _topics(blueprint)
    planner_map = topics[ManipulationModule, "voxel_map"]

    assert (
        topics[UrdfManagerPointClearing, "filtered_pointcloud"]
        == topics[RayTracingVoxelMap, "lidar"]
    )
    assert topics[RayTracingVoxelMap, "global_map"] == topics[PointCloudBoxFilter, "cloud"]
    assert topics[PointCloudBoxFilter, "filtered_cloud"] == planner_map
    # Nothing else feeds the planner's map.
    assert {key for key, topic in topics.items() if topic == planner_map} == {
        (ManipulationModule, "voxel_map"),
        (PointCloudBoxFilter, "filtered_cloud"),
    }


@pytest.mark.parametrize("blueprint", _GRASP_BLUEPRINTS)
def test_pick_and_place_finds_the_box_filter(blueprint: Blueprint) -> None:
    providers = [
        atom.module
        for atom in blueprint.active_blueprints
        if spec_structural_compliance(atom.module, PointCloudBoxFilterSpec)
    ]

    assert providers == [PointCloudBoxFilter]


@pytest.mark.parametrize(
    ("blueprint", "dof"),
    [(xarm_grasp, 7), (xarm_grasp_graspgenx, 7), (xarm6_grasp, 6), (xarm6_grasp_graspgenx, 6)],
)
def test_each_arm_plans_its_own_joints_and_hangs_the_camera_off_its_flange(
    blueprint: Blueprint, dof: int
) -> None:
    manipulation = next(
        atom.kwargs for atom in blueprint.active_blueprints if atom.module is ManipulationModule
    )
    pick_and_place = next(
        atom.kwargs for atom in blueprint.active_blueprints if atom.module is PickAndPlaceModule
    )
    model = manipulation["model"]

    assert model.joint_names == [f"joint{i}" for i in range(1, dof + 1)]
    assert model.tf_extra_links == [f"link{dof}"]
    assert [(tf.frame_id, tf.child_frame_id) for tf in manipulation["static_transforms"]] == [
        (f"link{dof}", "camera_link")
    ]
    assert pick_and_place["robot"] is model
