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

"""Anvil rig calibration; ideal rectified optics, not a D405 stereo simulator.

Base/overhead poses: saved 2026-10-15 provisional touch fit in
krishna/test/dual-openyam-bench-calibrated @ 6219280952. Wrist extrinsics:
YAM artifact CAD, not measured. Intrinsics: each attached D405's actual
640x480/15 colour profile queried 2026-10-25, identified by serial/USB topology.
Physical encoder corrections and factory lens distortion are not applied to
synthetic joints/pixels. See docs/robots/openyam-spatial.md for fidelity limits.
"""

from pathlib import Path

from scipy.spatial.transform import Rotation

from dimos.robot.manipulators.dual_openyam.spatial_station import SpatialStation
from dimos.sim2.sensors.camera.models.pinhole import PinholeIntrinsics
from dimos.sim2.sensors.camera.spec import Camera
from dimos.sim2.sensors.spec import Mount

# Rig X forward, Y left, Z up; source quaternions are xyzw.
# Optical X right/Y down/+Z forward -> MuJoCo X right/Y up/-Z forward.
_OVERHEAD_ROTATION = Rotation.from_quat((0.688681, -0.667454, 0.20986, -0.190219))
_OPTICAL_TO_MUJOCO = Rotation.from_quat((1.0, 0.0, 0.0, 0.0))

ANVIL = SpatialStation(
    scene=Path(__file__).parents[1] / "scenes" / "spatial.xml",
    bases={
        "left": Mount("world", xyz=(-0.00862, 0.22219, -0.00559)),
        "right": Mount(
            "world",
            xyz=(0.00862, -0.22219, 0.00559),
            rpy=tuple(
                Rotation.from_quat((-0.032517, 0.014623, 0.010943, 0.999304)).as_euler("xyz")
            ),
        ),
    },
    workspace=Camera(
        "workspace",
        Mount(
            "world",
            xyz=(-0.01676, 0.01263, 0.4152),
            rpy=tuple((_OVERHEAD_ROTATION * _OPTICAL_TO_MUJOCO).as_euler("xyz")),
        ),
        width=640,
        height=480,
        rate_hz=15,
        depth=True,
        pointcloud=True,
        shadows=False,
        # D405 230322272156, PCI 06:00.0.
        intrinsics=PinholeIntrinsics(
            640, 480, 393.0500793457031, 392.4581298828125, 320.0251770019531, 239.5882568359375
        ),
    ),
    wrists={
        "left": Camera(
            "wrist",
            "camera_d405",
            width=640,
            height=480,
            rate_hz=15,
            depth=False,
            shadows=False,
            # D405 260322272983, PCI 07:00.0; keep artifact CAD mount.
            intrinsics=PinholeIntrinsics(
                640, 480, 393.8309020996094, 393.284423828125, 320.8343505859375, 231.77371215820312
            ),
        ),
        "right": Camera(
            "wrist",
            "camera_d405",
            width=640,
            height=480,
            rate_hz=15,
            depth=False,
            shadows=False,
            # D405 260322276650, PCI 08:00.0; keep artifact CAD mount.
            intrinsics=PinholeIntrinsics(
                640,
                480,
                394.49407958984375,
                393.9751281738281,
                319.5488586425781,
                235.35317993164062,
            ),
        ),
    },
)
