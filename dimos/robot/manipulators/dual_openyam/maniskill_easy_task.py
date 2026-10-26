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

"""Easy primitive-object ManiSkill task for manual bimanual YAM practice."""

from typing import Any

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import numpy as np
import sapien
from sim_eval.robots.bimanual_yam import BimanualYAM
from sim_eval.tasks.yam_tasks.bimanual_put_everything_in_box import _build_open_box
import torch


@register_env("BimanualYAMEasyPickCubes-v1", max_episode_steps=1_000_000)
class BimanualYAMEasyPickCubesEnv(BaseEnv):
    """Move one large cube per arm into a wide, low-walled center tray."""

    SUPPORTED_ROBOTS = ["bimanual_yam"]
    agent: BimanualYAM

    cube_half_size = 0.035
    cube_positions = ((-0.30, 0.20), (-0.30, -0.20))
    tray_position = (-0.12, 0.0)
    tray_inner_half = 0.14
    tray_height = 0.025
    tray_wall = 0.006

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "bimanual_yam",
        reconfiguration_freq: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        pose = sapien_utils.look_at(eye=[-0.65, 0.0, 0.5], target=[-0.25, 0.0, 0.05])
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(eye=[0.2, 0.2, 0.5], target=[-0.3, 0.0, 0.05])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=720,
            fov=1.2,
            near=0.01,
            far=100,
            shader_pack="rt",
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        colors = ([0.9, 0.08, 0.08, 1.0], [0.05, 0.2, 0.95, 1.0])
        self.cubes = [
            actors.build_colorful_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=color,
                name=name,
                initial_pose=sapien.Pose(p=[x, y, self.cube_half_size]),
            )
            for name, color, (x, y) in zip(
                ("red_cube", "blue_cube"), colors, self.cube_positions, strict=True
            )
        ]

        self.tray = _build_open_box(
            self.scene,
            name="easy_tray",
            inner_half=self.tray_inner_half,
            height=self.tray_height,
            wall_thickness=self.tray_wall,
            color=[0.45, 0.45, 0.45, 1.0],
            initial_pose=sapien.Pose(p=[*self.tray_position, 0.0]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            count = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0.0, 0.01]))
            self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

            identity = torch.zeros((count, 4))
            identity[:, 0] = 1.0
            for cube, (x, y) in zip(self.cubes, self.cube_positions, strict=True):
                position = torch.tensor([x, y, self.cube_half_size]).repeat(count, 1)
                cube.set_pose(Pose.create_from_pq(position, identity))

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "tray_center": torch.tensor([*self.tray_position, self.tray_wall], device=self.device)
            .reshape(1, 3)
            .expand(self.num_envs, 3),
            "red_cube_pose": self.cubes[0].pose.raw_pose,
            "blue_cube_pose": self.cubes[1].pose.raw_pose,
        }

    def evaluate(self) -> dict[str, Any]:
        tray_x, tray_y = self.tray_position
        inside = []
        for cube in self.cubes:
            position = cube.pose.p
            inside.append(
                (torch.abs(position[:, 0] - tray_x) < self.tray_inner_half)
                & (torch.abs(position[:, 1] - tray_y) < self.tray_inner_half)
                & (position[:, 2] < self.tray_height + 2 * self.cube_half_size + 0.02)
            )
        in_tray = torch.stack(inside)
        return {
            "success": in_tray.all(dim=0),
            "n_in_tray": in_tray.float().sum(dim=0),
        }
