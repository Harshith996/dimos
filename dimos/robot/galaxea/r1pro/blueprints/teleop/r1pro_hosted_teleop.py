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

"""R1 Pro driven from the hosted (Cloudflare) teleop page, with the left head camera as its video:
``TRANSPORTS__BROKER__API_KEY=dimos_sk_... dimos run r1pro-hosted-teleop``."""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import CloudflareTransport, CloudflareVideoTransport
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.galaxea.r1pro.blueprints.basic.r1pro_coordinator import r1pro_control
from dimos.teleop.hosted.camera_mux import CameraMuxModule
from dimos.teleop.hosted.drive_command import DriveCommandModule
from dimos.teleop.hosted.hosted_stats import HostedStatsModule
from dimos.teleop.hosted.jpeg_decode import JpegDecodeModule
from dimos.teleop.hosted.robot_type import RobotType

r1pro_hosted_teleop = (
    autoconnect(
        r1pro_control(enable_wrist_color=False, enable_imu=False, head_right=False),
        DriveCommandModule.blueprint(),
        JpegDecodeModule.blueprint(),
        CameraMuxModule.blueprint(cameras=["cam1"]),
        HostedStatsModule.blueprint(),
    )
    .remappings(
        [
            (JpegDecodeModule, "jpeg_in", "head_left_color"),
            (JpegDecodeModule, "image_out", "cam1"),
        ]
    )
    .transports(
        {
            # Drive view: the page sends a twist and E-STOP; the robot sends video, acks and telemetry.
            ("cmd_vel_in", TwistStamped): CloudflareTransport.spec("cmd_unreliable", TwistStamped),
            ("cmd_raw", bytes): CloudflareTransport.spec("cmd_unreliable"),
            ("state_json", bytes): CloudflareTransport.spec(
                "state_reliable", robot_type=RobotType.GO2
            ),
            ("camera_select", bytes): CloudflareTransport.spec("state_reliable"),
            ("mux_image", Image): CloudflareVideoTransport.spec(),
            ("telemetry_out", bytes): CloudflareTransport.spec("state_reliable_back"),
            ("cmd_ack", bytes): CloudflareTransport.spec("state_reliable_back"),
        }
    )
    # One process, so every Cloudflare transport shares one broker session.
    .global_config(viewer="none", n_workers=1)
)
