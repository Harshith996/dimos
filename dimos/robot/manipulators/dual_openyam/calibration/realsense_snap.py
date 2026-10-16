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

"""TEMP: grab colour frames at a fixed exposure from one RealSense, outside dimos.

The dimos RealSense module only has auto exposure, which blows out the table
under the rig lights. pyrealsense2 is not in the dimos venv, so this runs via uv:

    uv run --no-project --with pyrealsense2 --with numpy --with pillow --python 3.12 \
        python realsense_snap.py SERIAL OUT_DIR EXPOSURE_US COUNT
"""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pyrealsense2 as rs

serial, out, exposure, count = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
config = rs.config()
config.enable_device(serial)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
pipe = rs.pipeline()
profile = pipe.start(config)
try:
    for sensor in profile.get_device().query_sensors():
        if sensor.supports(rs.option.exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, exposure)
    for _ in range(10):
        pipe.wait_for_frames()
    for i in range(count):
        frame = pipe.wait_for_frames().get_color_frame()
        Image.fromarray(np.asanyarray(frame.get_data())).save(out / f"color_{i:02d}.png")
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    (out / "intrinsics.json").write_text(
        json.dumps(
            {
                "serial": serial,
                "exposure_us": exposure,
                "color": {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.ppx,
                    "cy": intr.ppy,
                    "model": str(intr.model),
                    "coeffs": list(intr.coeffs),
                },
            },
            indent=2,
        )
    )
finally:
    pipe.stop()
