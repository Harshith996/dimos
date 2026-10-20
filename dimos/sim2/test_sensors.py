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

from contextlib import ExitStack
from dataclasses import replace
import threading
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.transport import ZenohTransport
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.sim2.blueprint import simulation
from dimos.sim2.client import SimRobotClient
from dimos.sim2.ipc.abi import ChannelDescriptor
from dimos.sim2.module import SimulationModule
from dimos.sim2.scene_types import SceneUpdate
from dimos.sim2.sensors.camera.module import SimRGBDCameraModule
from dimos.sim2.sensors.camera.renderers.mujoco import MujocoCamera
from dimos.sim2.sensors.lidar.module import LidarModule
from dimos.sim2.sensors.lidar.raycast import Raycaster
from dimos.sim2.sensors.reader import WorldReader
from dimos.sim2.spec import RobotInstance


@pytest.fixture
def reader(worlds, tmp_path):
    world = worlds()
    model_path = tmp_path / "world.mjb"
    mujoco.mj_saveModel(world.model, str(model_path), None)
    observer = WorldReader(
        {"model": str(model_path), "snapshot": world.snapshot_descriptor.to_dict()}
    )
    try:
        yield world, observer
    finally:
        observer.close()


def test_world_snapshot_renders_nonblank_rgb_and_metric_depth(reader):
    world, observer = reader
    assert observer.update()
    camera = observer.model.camera("left/view").id
    renderer = MujocoCamera(observer.model, 160, 120)
    try:
        rgb, depth = renderer.capture(observer.data, camera, True)
        assert rgb.shape == (120, 160, 3)
        assert rgb.std() > 15
        assert depth is not None
        assert depth.shape == (120, 160)
        assert np.count_nonzero((depth > 0.1) & (depth < 10)) > 1000
        assert world.tick == 0
    finally:
        renderer.close()


def test_lidar_uses_its_own_model_and_sees_scene_mutations(reader):
    world, observer = reader
    assert observer.update()
    groups = world.model.geom_group.copy()
    raycaster = Raycaster(observer.model, observer.model.body("left/base").id)
    rays = np.array([[1.0, 0, 0]])
    origin = np.array([0.0, 0, 0.3])
    before = raycaster.ranges(observer.data, origin, rays, 10)
    assert before == pytest.approx([1.3])
    world.set_scene_state(SceneUpdate(poses={"box": Pose(2.5, 0, 0.3)}))
    assert observer.update()
    after = raycaster.ranges(observer.data, origin, rays, 10)
    assert after == pytest.approx([2.3])
    np.testing.assert_array_equal(world.model.geom_group, groups)
    assert observer.model is not world.model


def test_reader_observes_reset_generation_without_recompiling(reader):
    world, observer = reader
    assert observer.update()
    model = observer.model
    generation = observer.episode
    world.reset()
    assert observer.update()
    assert observer.episode == generation + 1
    assert observer.model is model


def test_busy_snapshot_is_skipped_without_stopping_sensor(reader, mocker):
    world, observer = reader
    frame = world.snapshots.read_observation()
    mocker.patch.object(
        observer.channel, "read_observation", side_effect=[BlockingIOError("busy"), frame]
    )
    assert not observer.update()
    assert observer.update()
    assert observer.sequence == frame.metadata.sequence


def test_composition_contains_one_world_and_no_public_sim_connections(world_config):
    instance = world_config.robots["left"]
    robot = instance.config
    extra_camera = replace(robot.sensors[1], name="inspection")
    robot = robot.with_sensor(extra_camera)
    blueprint = simulation(
        scene=world_config.scene,
        robots={
            "left": replace(instance, config=robot),
            "right": RobotInstance(instance.config, xyz=(0, 2, 0.3)),
        },
    )
    atoms = blueprint.blueprints
    types = [atom.module for atom in atoms]
    assert types.count(SimulationModule) == 1
    assert types.count(SimRGBDCameraModule) == 3
    assert types.count(LidarModule) == 2
    assert len(types) == 6
    assert len({atom.name for atom in atoms}) == 6


def test_blueprint_deploys_physics_camera_and_lidar_in_workers(world_config):
    image_ready, cloud_ready = threading.Event(), threading.Event()
    images, clouds = [], []

    def on_image(value):
        if not images:
            images.append(value)
            image_ready.set()

    def on_cloud(value):
        if not clouds:
            clouds.append(value)
            cloud_ready.set()

    with ExitStack() as stack:
        prefix = "sim2-test/" + uuid4().hex
        image = ZenohTransport(prefix + "/image", Image)
        cloud = ZenohTransport(prefix + "/cloud", PointCloud2)
        stack.callback(image.stop)
        stack.callback(cloud.stop)
        stack.callback(image.subscribe(on_image))
        stack.callback(cloud.subscribe(on_cloud))
        blueprint = simulation(scene=world_config.scene, robots=world_config.robots)
        coordinator = ModuleCoordinator.build(
            blueprint.transports(
                {("color_image", Image): image, ("pointcloud", PointCloud2): cloud}
            ).global_config(n_workers=1, viewer="none", transport="zenoh")
        )
        stack.callback(coordinator.stop)
        sim = coordinator.get_instance(SimulationModule)
        assert image_ready.wait(15), "camera worker did not publish"
        assert cloud_ready.wait(15), "lidar worker did not publish"
        assert images[0].frame_id == "left/camera_optical"
        assert clouds[0].frame_id == "world"
        assert sim.status()["running"]
        assert sim.status()["steps"] > 0
        descriptor = ChannelDescriptor.from_dict(sim.describe()["robots"]["left"])
        with SimRobotClient(descriptor) as client:
            before = client.read()
            sim.reset()
            after = client.read()
            assert after.metadata.episode_id == before.metadata.episode_id + 1

    with pytest.raises(FileNotFoundError):
        with SimRobotClient(descriptor):
            pytest.fail("shutdown left a live robot channel")
