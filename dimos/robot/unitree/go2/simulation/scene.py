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

"""Procedural Go2 scene and read-only diagnostics; sim2 owns all physics and sensing."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

from filelock import FileLock
import mujoco
import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dimos.constants import CACHE_DIR
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.LineSegments3D import LineSegments3D
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sim_msgs.Contacts import Contact, Contacts, Part
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.module import SimulationSpec
from dimos.sim2.sensors.reader import WorldReader
from dimos.simulation.scenes.procedural import Family, generate
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

SENSOR_FRAME_ID = "mid360_link"
ODOM_FRAME_ID = "odom"

BOX_RGBA = {
    "wall": (0.86, 0.84, 0.78, 1.0),
    "ceiling": (0.95, 0.95, 0.95, 1.0),
    "clutter": (0.45, 0.55, 0.70, 1.0),
}
FLOOR_CHECKER = ((0.62, 0.60, 0.56), (0.70, 0.68, 0.64))
BOX_EDGES = np.array(
    [
        [[-1, -1, -1], [1, -1, -1]],
        [[-1, 1, -1], [1, 1, -1]],
        [[-1, -1, 1], [1, -1, 1]],
        [[-1, 1, 1], [1, 1, 1]],
        [[-1, -1, -1], [-1, 1, -1]],
        [[1, -1, -1], [1, 1, -1]],
        [[-1, -1, 1], [-1, 1, 1]],
        [[1, -1, 1], [1, 1, 1]],
        [[-1, -1, -1], [-1, -1, 1]],
        [[1, -1, -1], [1, -1, 1]],
        [[-1, 1, -1], [-1, 1, 1]],
        [[1, 1, -1], [1, 1, 1]],
    ],
    dtype=np.float64,
)


def _dress(spec: mujoco.MjSpec) -> None:
    """A checkered floor material and a sun, since the Go2 model brings no light of its own."""
    texture = spec.add_texture()
    texture.name = "floor"
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    texture.rgb1, texture.rgb2 = FLOOR_CHECKER
    texture.width = texture.height = 256
    material = spec.add_material()
    material.name = "floor"
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    material.texrepeat = (1.0, 1.0)
    material.texuniform = True
    material.reflectance = 0.1
    light = spec.worldbody.add_light()
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = (0.0, 0.0, 10.0)
    light.dir = (0.3, 0.2, -1.0)
    light.castshadow = True
    light.diffuse = (0.6, 0.6, 0.6)
    light.specular = (0.1, 0.1, 0.1)
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.3, 0.3, 0.3)


def procedural_scene(family: Family = "office", seed: int = 1) -> Path:
    """Export the existing box generator as robot-free MJCF, without a second runtime."""
    scene = generate(family, seed)
    directory = CACHE_DIR / "sim2" / "procedural" / scene.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scene.xml"
    with FileLock(str(directory / ".lock")):
        world = mujoco.MjSpec()
        _dress(world)
        for i, box in enumerate(scene.boxes):
            geom = world.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                name=f"{box.kind}_{i}",
                pos=box.center,
                size=box.half,
                group=1 if box.kind == "ceiling" else 0,
            )
            if box.kind == "floor":
                geom.material = "floor"
            else:
                geom.rgba = BOX_RGBA[box.kind]
        world.compile()
        path.write_text(world.to_xml())
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "format": "dimos.scene.v1",
                    "id": scene.name,
                    "hidden_geom_groups": [1],
                    "spawns": {"default": {"position": scene.start, "orientation": [0, 0, 0, 1]}},
                }
            )
        )
    return path


class Go2SimTelemetryConfig(ModuleConfig):
    robot_id: str
    rate_hz: float = Field(default=50, gt=0, allow_inf_nan=False)


class Go2SimTelemetry(Module):
    """Preserve the procedural nav stack's Point-LIO-shaped truth diagnostics.

    This worker reads snapshots; it never steps physics, runs a policy or casts rays.
    The lidar worker supplies a deskewed sensor-frame cloud.
    """

    config: Go2SimTelemetryConfig
    _simulation: SimulationSpec
    dedicated_worker = True

    scan: In[PointCloud2]
    lidar: Out[PointCloud2]
    odometry: Out[Odometry]
    tf: Out[TFMessage]
    ground_truth: Out[PoseStamped]
    contacts: Out[Contacts]
    scene: Out[LineSegments3D]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reader: WorldReader | None = None
        self._error: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.reader = WorldReader(self._simulation.describe())
        try:
            self.open()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self.reader.close()
            raise

    def _run(self) -> None:
        assert self.reader is not None
        try:
            while not self._stop.is_set():
                before = time.monotonic()
                if self.reader.update() and self.reader.channel.lifecycle == "ready":
                    self.capture()
                self._stop.wait(max(0, 1 / self.config.rate_hz - (time.monotonic() - before)))
        except Exception as error:
            self._error = str(error)
            logger.exception("Go2 truth diagnostics stopped")

    @rpc
    def sensor_status(self) -> dict[str, Any]:
        return {"error": self._error}

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Go2 truth diagnostics did not stop")
        if self.reader is not None:
            self.reader.close()
        super().stop()

    def open(self) -> None:
        assert self.reader is not None
        m = self.reader.model
        prefix = f"{self.config.robot_id}/"
        self._base = m.body(prefix + "base").id
        self._site = m.site(prefix + "sensor/lidar").id
        self._feet = {m.geom(prefix + leg).id for leg in ("FL", "FR", "RL", "RR")}
        self._housing = m.geom(prefix + "mid360").id
        self._kind = {}
        for i in range(m.ngeom):
            name = m.geom(i).name
            kind = name.split("_", 1)[0]
            if m.geom_bodyid[i] == 0 and kind in ("floor", "wall", "ceiling", "clutter"):
                self._kind[i] = kind
        self._last_contacts: list[Contact] | None = None
        self._next_scene = 0.0
        self.register_disposable(Disposable(self.scan.subscribe(self._publish_scan)))

    def _publish_scan(self, cloud: PointCloud2) -> None:
        # Both frames have the same mount. Only the public frame name differs.
        cloud.frame_id = SENSOR_FRAME_ID
        self.lidar.publish(cloud)

    def capture(self) -> None:
        assert self.reader is not None
        m, d = self.reader.model, self.reader.data
        stamp = self.reader.timestamp
        rotation = d.site_xmat[self._site].reshape(3, 3)
        pose = Pose(d.site_xpos[self._site], Rotation.from_matrix(rotation).as_quat())
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self._site, velocity, 0)
        self.odometry.publish(
            Odometry(
                ts=stamp,
                frame_id=ODOM_FRAME_ID,
                child_frame_id=SENSOR_FRAME_ID,
                pose=pose,
                twist=Twist(velocity[3:].tolist(), (rotation.T @ velocity[:3]).tolist()),
            )
        )
        self.tf.publish(
            TFMessage(
                Transform(
                    translation=pose.position,
                    rotation=pose.orientation,
                    frame_id=ODOM_FRAME_ID,
                    child_frame_id=SENSOR_FRAME_ID,
                    ts=stamp,
                )
            )
        )
        self.ground_truth.publish(
            PoseStamped(
                position=d.xpos[self._base].tolist(),
                orientation=Rotation.from_matrix(d.xmat[self._base].reshape(3, 3))
                .as_quat()
                .tolist(),
                ts=stamp,
                frame_id=ODOM_FRAME_ID,
            )
        )
        contacts = self._contacts()
        if contacts != self._last_contacts:
            self.contacts.publish(Contacts(contacts, ts=stamp))
            self._last_contacts = contacts
        if stamp >= self._next_scene:
            boxes = [i for i in self._kind if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
            edges = (
                d.geom_xpos[boxes, None, None, :]
                + BOX_EDGES[None] * m.geom_size[boxes, None, None, :]
            ).reshape(-1, 2, 3)
            self.scene.publish(LineSegments3D(ts=stamp, frame_id=ODOM_FRAME_ID, segments=edges))
            self._next_scene = stamp + 2

    def _contacts(self) -> list[Contact]:
        assert self.reader is not None
        m, d = self.reader.model, self.reader.data
        found: set[Contact] = set()
        for c in d.contact:
            for scene, robot in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                kind = self._kind.get(scene)
                if kind is not None and m.geom_bodyid[robot] != 0:
                    part: Part = (
                        "lidar"
                        if robot == self._housing
                        else "foot"
                        if robot in self._feet
                        else "trunk"
                        if m.geom_bodyid[robot] == self._base
                        else "leg"
                    )
                    found.add(Contact(part, kind))
        return sorted(found)
