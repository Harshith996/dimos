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

"""Rerun rendering for R1 Pro streams."""

from __future__ import annotations

from typing import Any

import numpy as np

from dimos.robot.galaxea.r1pro.head_cameras import HeadLeftCameraConfig
from dimos.robot.galaxea.r1pro.lio import R1ProLioConfig

# Both clouds share the `lidar` port, so the viewer splits them by frame_id.
_CLOUD_COLOR_UNKNOWN = [170, 170, 170]


def render_cloud(msg: Any) -> Any:
    import rerun as rr

    frame_id = getattr(msg, "frame_id", "") or "unknown"
    path = f"world/lidar/{frame_id}"
    xyz = msg.points_f32()
    head_frame = HeadLeftCameraConfig.model_fields["frame_id"].default
    lidar_frame = R1ProLioConfig.model_fields["lidar_frame"].default
    if frame_id in (head_frame, lidar_frame) and len(xyz):
        # Rainbow by the frame's own up axis (the optical frame's is -y), inverted so dark blue is never on black.
        up = -xyz[:, 1] if frame_id == head_frame else xyz[:, 2]
        level = 1.0 - np.clip((up - up.min()) / max(float(np.ptp(up)), 1e-3), 0.0, 1.0)
        colors = np.stack(
            [255 * level, 255 * (1 - np.abs(2 * level - 1)), 255 * (1 - level)], axis=1
        )
        points = rr.Points3D(xyz, colors=(0.35 * 255 + 0.65 * colors).astype(np.uint8), radii=0.02)
    else:
        points = msg.to_rerun(colors=_CLOUD_COLOR_UNKNOWN, rgb=False)
    # A list skips the bridge's frame attach, so each cloud hangs off its own tf frame here.
    return [(path, points), (path, rr.Transform3D(parent_frame=f"tf#/{frame_id}"))]
