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

"""Independent instantaneous or rolling scans; raw returns never use truth deskew."""

import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dimos.core.core import rpc
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.sensors.lidar.raycast import Raycaster
from dimos.sim2.sensors.module import SensorModule, SensorModuleConfig
from dimos.sim2.sensors.spec import Lidar, RangeResponse, RayPattern, TimedRayPattern


class LidarModuleConfig(SensorModuleConfig):
    sensor: Lidar
    root_body: str


class LidarModule(SensorModule):
    config: LidarModuleConfig
    pointcloud: Out[PointCloud2]
    raw_pointcloud: Out[PointCloud2]
    imu_raw: Out[Imu]
    tf: Out[TFMessage]

    def open(self) -> None:
        # Initialize PointCloud2's lazy Open3D dependency before high-rate acquisition.
        PointCloud2()
        robot = self.config.robot_id
        self._site = self.reader.model.site(f"{robot}/{self.config.sensor.model_name}").id
        self._frame = f"{robot}/{self.config.sensor.name}"
        self._model = self.config.sensor.model(**self.config.sensor.model_kwargs)
        self._rays = self._model.directions() if isinstance(self._model, RayPattern) else None
        if isinstance(self._model, TimedRayPattern):
            # Load the pattern before any IMU publication, not halfway through a scan.
            self._model.scan(0.0, 1 / self.config.sensor.rate_hz)
        self._episode = -1
        self._last_scan = 0
        self._last_imu_sequence = -1
        self._last_imu_time: float | None = None
        self._clock_origin = 0.0
        if self.config.sensor.imu is not None:
            name = f"{robot}/sensor/{self.config.sensor.imu.name}"
            self._gyro = self.reader.model.sensor(f"{name}/gyro").id
            self._accel = self.reader.model.sensor(f"{name}/accel").id
        self._scan_status: dict[str, Any] = {
            "fidelity": "rolling" if isinstance(self._model, TimedRayPattern) else "instantaneous",
            "scans": 0,
            "dropped_scans": 0,
            "history_pending": False,
            "imu_samples": 0,
            "dropped_imu_samples": 0,
        }
        self._raycaster = Raycaster(
            self.reader.model,
            self.reader.model.body(f"{robot}/{self.config.root_body}").id,
            self_occlusion=self.config.sensor.self_occlusion,
            excluded_meshes=tuple(f"{robot}/{mesh}" for mesh in self.config.sensor.excluded_meshes),
        )

    def capture(self) -> None:
        if isinstance(self._model, TimedRayPattern):
            if self._episode != self.reader.episode:
                self._episode = self.reader.episode
                self._clock_origin = self.reader.timestamp - float(self.reader.data.time)
                self._last_scan = 0
                self._last_imu_sequence = -1
                self._last_imu_time = None
            self._capture_imu()
            self._capture_rolling(self._model)
            return
        assert self._rays is not None
        data = self.reader.data
        origin = data.site_xpos[self._site]
        rotation = data.site_xmat[self._site].reshape(3, 3)
        rays = self._rays @ rotation.T
        elevation_limit = self.config.sensor.maximum_world_elevation
        if elevation_limit is not None:
            rays = rays[rays[:, 2] <= np.sin(np.deg2rad(elevation_limit)) + 1e-12]
        points = self._raycaster.cast(
            data,
            origin,
            rays,
            self._model.min_range,
            self._model.max_range,
        )
        self._publish_cloud(points, origin, rotation)

    def _publish_cloud(
        self,
        points: NDArray[np.float64],
        origin: NDArray[np.float64],
        rotation: NDArray[np.float64],
    ) -> None:
        if not self.config.sensor.truth_outputs:
            return
        frame = "world"
        if self.config.sensor.output_frame == "sensor":
            points = (points - origin) @ rotation
            frame = self._frame
        if frame == self._frame or isinstance(self._model, TimedRayPattern):
            self.tf.publish(
                TFMessage(
                    Transform(
                        translation=Vector3(*origin),
                        rotation=Quaternion(*Rotation.from_matrix(rotation).as_quat()),
                        frame_id="world",
                        child_frame_id=self._frame,
                        ts=self.reader.timestamp,
                    )
                )
            )
        self.pointcloud.publish(
            PointCloud2.from_numpy(
                points.astype(np.float32),
                frame_id=frame,
                timestamp=self.reader.timestamp,
            )
        )

    def _capture_imu(self) -> None:
        imu = self.config.sensor.imu
        if imu is None:
            return
        frames = [
            frame
            for frame in self.reader.channel.read_observations()
            if frame.metadata.episode_id == self._episode
            and self._last_imu_sequence < frame.metadata.sequence <= self.reader.sequence
        ]
        for frame in frames:
            t = frame.metadata.sim_time
            if not math.isclose(t * imu.rate_hz, round(t * imu.rate_hz), abs_tol=1e-7):
                continue
            self.reader.restore(frame)
            if self.reader.channel.episode_id != self._episode:
                return
            self.imu_raw.publish(
                Imu(
                    angular_velocity=Vector3(*self.reader.data.sensor(self._gyro).data),
                    linear_acceleration=Vector3(*self.reader.data.sensor(self._accel).data),
                    orientation_covariance=[-1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                    frame_id=f"{self.config.robot_id}/{imu.name}",
                    ts=self._clock_origin + t,
                )
            )
            with self._error_lock:
                self._scan_status["imu_samples"] += 1
                if self._last_imu_time is not None:
                    self._scan_status["dropped_imu_samples"] += max(
                        0, round((t - self._last_imu_time) * imu.rate_hz) - 1
                    )
            self._last_imu_time = t
        if frames:
            self._last_imu_sequence = frames[-1].metadata.sequence
            self.reader.restore(frames[-1])

    def _capture_rolling(self, pattern: TimedRayPattern) -> None:
        rate = self.config.sensor.rate_hz
        index = math.floor((float(self.reader.data.time) + 1e-9) * rate)
        if index <= self._last_scan:
            return
        end, duration = index / rate, 1 / rate
        start = end - duration
        if not self.reader.history(start, end):
            with self._error_lock:
                self._scan_status["history_pending"] = True
            return
        before = time.monotonic()
        rays = pattern.scan(start, duration)
        bins = np.floor(rays.offsets * pattern.motion_sample_rate_hz).astype(np.int64)
        distances = np.full(len(rays.offsets), -1.0)
        world_points = np.empty_like(rays.directions)
        world_rays = np.empty_like(rays.directions)
        cos_incidence = np.zeros(len(rays.offsets))
        mapping_visible = np.ones(len(rays.offsets), dtype=bool)
        elevation = self.config.sensor.maximum_world_elevation
        for motion_bin in np.unique(bins):
            indices = np.flatnonzero(bins == motion_bin)
            self.reader.restore_at(start + float(np.mean(rays.offsets[indices])))
            origin = self.reader.data.site_xpos[self._site]
            rotation = self.reader.data.site_xmat[self._site].reshape(3, 3)
            directions = rays.directions[indices] @ rotation.T
            ranges, normals = self._raycaster.hits(
                self.reader.data, origin, directions, pattern.max_range
            )
            cos_incidence[indices] = np.einsum("ij,ij->i", directions, normals)
            distances[indices] = ranges
            world_rays[indices] = directions
            world_points[indices] = origin + directions * ranges[:, None]
            if elevation is not None:
                mapping_visible[indices] = directions[:, 2] <= np.sin(np.deg2rad(elevation)) + 1e-12
        if isinstance(pattern, RangeResponse):
            measured = pattern.measure(distances, cos_incidence, start)
            world_points += world_rays * (measured - distances)[:, None]
            distances = measured
        self.reader.restore_at(end)
        self.reader.timestamp = self._clock_origin + end
        if self.reader.channel.episode_id != self._episode:
            return
        hit = (distances >= pattern.min_range) & (distances <= pattern.max_range)
        raw = rays.directions[hit] * distances[hit, None]
        self.raw_pointcloud.publish(
            PointCloud2.from_numpy(
                raw.astype(np.float32),
                frame_id=self._frame,
                timestamp=self.reader.timestamp - duration,
                offset_times=np.rint(rays.offsets[hit] * 1e9).astype(np.uint32),
                lines=rays.lines[hit],
            )
        )
        self._publish_cloud(
            world_points[hit & mapping_visible],
            self.reader.data.site_xpos[self._site],
            self.reader.data.site_xmat[self._site].reshape(3, 3),
        )
        with self._error_lock:
            self._scan_status.update(
                scans=self._scan_status["scans"] + 1,
                dropped_scans=self._scan_status["dropped_scans"]
                + (max(0, index - self._last_scan - 1) if self._last_scan else 0),
                history_pending=False,
                rays=len(rays.offsets),
                returns=int(hit.sum()),
                capture_seconds=time.monotonic() - before,
                scan_start=start,
                scan_end=end,
                episode=self._episode,
                # Privileged diagnostic only; never connected to estimator inputs.
                truth_pose={
                    "timestamp": self.reader.timestamp,
                    "position": self.reader.data.site_xpos[self._site].tolist(),
                    "orientation": Rotation.from_matrix(
                        self.reader.data.site_xmat[self._site].reshape(3, 3)
                    )
                    .as_quat()
                    .tolist(),
                },
            )
        self._last_scan = index

    @rpc
    def sensor_status(self) -> dict[str, Any]:
        with self._error_lock:
            return {"error": self._error, **self._scan_status}
