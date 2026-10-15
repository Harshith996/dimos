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

"""One URDF and the live joint state: every joint on tf, and the robot cut out of any point cloud wired in."""

from __future__ import annotations

import importlib
import time
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.core.stream import IO
from dimos.manipulation.planning.utils.point_cloud_self_filter import (
    PointCloudSelfFilter,
    PointCloudSelfFilterConfig,
)
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.robot.assets.model import RobotModel
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import matrix_to_pose

logger = setup_logger()


# Their pose comes from tf (odometry, a planar base), so they are read from tf, never published to it.
_MULTI_DOF_JOINT_TYPES = frozenset({"floating", "planar"})


class UrdfManagerConfig(PointCloudSelfFilterConfig):
    model: RobotModel
    padding_m: float = Field(default=0.01, ge=0.0)
    # Every single-DOF and fixed URDF joint as a parent -> child tf edge; off where another module owns those frames.
    publish_joint_tf: bool = True
    joint_tf_hz: float = Field(default=30.0, gt=0.0)


class UrdfManager(PointCloudSelfFilter):
    """Poses the URDF from the joint state: publishes it on tf and drops robot returns from clouds at capture time."""

    config: UrdfManagerConfig

    tf: IO[TFMessage]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._load_scene()
        self._next_joint_tf_t = 0.0

    def add_joint_state(self, state: JointState) -> None:
        super().add_joint_state(state)
        if not self.config.publish_joint_tf or time.monotonic() < self._next_joint_tf_t:
            return
        self._next_joint_tf_t = time.monotonic() + 1.0 / self.config.joint_tf_hz
        # The scene's FK state is shared with the cloud filter's thread.
        with self._filter_lock:
            transforms = self.joint_transforms(state)
        self.tfbuffer.publish(*transforms)

    def joint_transforms(self, state: JointState) -> list[Transform]:
        """One parent -> child transform per URDF joint; joints the state leaves out stay at zero."""
        positions = dict(zip(state.name, state.position, strict=False))
        q = self._neutral_q.copy()
        for name in self._scene.getJointNames():
            info = self._scene.getJointInfo(name)
            if info.num_velocity_dofs == 1 and name in positions:
                value = positions[name]
                values = [np.cos(value), np.sin(value)] if info.num_position_dofs == 2 else [value]
                q[self._scene.getJointPositionIndices([name])] = values
        # Fixed joints are republished with the moving ones: a chained lookup is only as fresh as its stalest edge.
        return [
            Transform.from_matrix(
                np.asarray(self._context.forwardKinematics(q, joint.child_link, joint.parent_link)),
                frame_id=joint.parent_link,
                child_frame_id=joint.child_link,
                ts=state.ts,
            )
            for joint in self._joints.values()
            if joint.type not in _MULTI_DOF_JOINT_TYPES
        ]

    def _compute_keep_mask(
        self, cloud: PointCloud2, points: NDArray[np.float32]
    ) -> NDArray[np.bool_] | None:
        base_from_sensor = self.tfbuffer.get(
            self._base_link,
            cloud.frame_id,
            time_point=cloud.ts,
            time_tolerance=self.config.tf_tolerance_s,
            forward_tolerance=self.config.tf_forward_tolerance_s,
        )
        q = self._capture_configuration(cloud.ts)
        if base_from_sensor is None or q is None:
            logger.warning("Dropping cloud: capture-time robot state or TF unavailable")
            return None
        # This Scene is rooted at the URDF base, so its world is the base frame.
        transform = base_from_sensor.to_matrix()
        base_points = np.asarray(points @ transform[:3, :3].T + transform[:3, 3], dtype=np.float64)
        return ~np.asarray(self._body_filter.computeMask(q, base_points), dtype=bool)

    def _capture_configuration(self, stamp: float) -> NDArray[np.float64] | None:
        state = self._states.find_closest(stamp, self.config.state_tolerance_s)
        positions: dict[str, float] = {}
        if state is not None:
            if len(state.name) != len(state.position) or len(set(state.name)) != len(state.name):
                return None
            positions = dict(zip(state.name, state.position, strict=True))
            if not np.isfinite(list(positions.values())).all():
                return None
        # Start from the model's neutral configuration, never its latest state.
        q = self._neutral_q.copy()
        for name in self._scene.getJointNames():
            info = self._scene.getJointInfo(name)
            if info.num_velocity_dofs == 1:
                if name not in positions:
                    return None
                value = positions[name]
                values = [np.cos(value), np.sin(value)] if info.num_position_dofs == 2 else [value]
            else:
                # Multi-DOF joints have no scalar JointState representation.
                # Recover their configuration from capture-time relative TF.
                joint = self._joints[name]
                parent_from_child = self.tfbuffer.get(
                    joint.parent_link,
                    joint.child_link,
                    time_point=stamp,
                    time_tolerance=self.config.tf_tolerance_s,
                    forward_tolerance=self.config.tf_forward_tolerance_s,
                )
                if parent_from_child is None:
                    return None
                origin = np.asarray(
                    self._context.forwardKinematics(
                        self._neutral_q, joint.child_link, joint.parent_link
                    )
                )
                motion = np.linalg.inv(origin) @ parent_from_child.to_matrix()
                pose = matrix_to_pose(motion)
                if info.num_position_dofs == 4:
                    if not np.allclose(motion[2], [0, 0, 1, 0], atol=1e-6):
                        return None
                    angle = np.arctan2(motion[1, 0], motion[0, 0])
                    values = [motion[0, 3], motion[1, 3], np.cos(angle), np.sin(angle)]
                elif info.num_position_dofs == 7:
                    values = [
                        *motion[:3, 3],
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                else:
                    raise ValueError(f"Unsupported robot joint layout: {name}")
            q[self._scene.getJointPositionIndices([name])] = values
        return q

    def _load_scene(self) -> None:
        # RoboPlan is optional for stacks that do not use this module.
        native = importlib.import_module("roboplan.core")

        description = self.config.model.load()
        root = ET.fromstring(description.xml)
        root.set("version", "1.0")
        self._scene = native.Scene(
            "dimos_self_filter",
            native.loadUrdfSceneDescriptionFromXml(
                ET.tostring(root, encoding="unicode"),
                [str(path) for path in description.package_paths.values()],
            ),
        )
        self._neutral_q = np.asarray(self._scene.getCurrentJointPositions()).copy()
        self._context = native.SceneContext(self._scene)
        self._body_filter = native.RobotBodyFilter(
            self._scene,
            native.RobotBodyFilterOptions(
                padding=self.config.padding_m,
                method=native.RobotBodyFilterMethod.Narrowphase,
                num_threads=1,
            ),
        )
        # Reuse the asset loader's topology for multi-DOF TF lookup, not geometry parsing.
        self._base_link = description.root_link
        self._joints = {joint.name: joint for joint in description.joints}
