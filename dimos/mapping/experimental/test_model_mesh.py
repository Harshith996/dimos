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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.mapping.experimental.model_mesh import ModelMesh, load_glb, model_paths
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

trimesh = pytest.importorskip("trimesh")


def _model_dir(tmp_path: Path) -> Path:
    """model.glb with a wall box at gltf y = 1 (dimos z = 1) and a ceiling, plus lidar.pc2.lcm."""
    scene = trimesh.Scene()
    wall = trimesh.creation.box((4.0, 2.0, 0.2))
    wall.apply_translation((0.0, 1.0, 0.0))
    scene.add_geometry(wall, node_name="walls/wall", geom_name="wall")
    ceiling = trimesh.creation.box((4.0, 0.1, 4.0))
    ceiling.apply_translation((0.0, 3.0, 0.0))
    scene.add_geometry(ceiling, node_name="ceilings/ceiling", geom_name="ceiling")
    scene.export(tmp_path / "model.glb")
    pts = np.random.default_rng(0).uniform(-2, 2, (100, 3)).astype(np.float32)
    (tmp_path / "lidar.pc2.lcm").write_bytes(
        PointCloud2.from_numpy(pts, frame_id="map", timestamp=0.0).lcm_encode()
    )
    return tmp_path


def test_model_paths_takes_a_directory_or_a_pair(tmp_path: Path) -> None:
    d = _model_dir(tmp_path)
    assert model_paths(str(d)) == (d / "lidar.pc2.lcm", d / "model.glb")
    assert model_paths(f"{d / 'lidar.pc2.lcm'},{d / 'model.glb'}") == (
        d / "lidar.pc2.lcm",
        d / "model.glb",
    )
    with pytest.raises(ValueError):
        model_paths(str(d / "model.glb"))


def test_glb_is_y_up_and_layers_can_be_skipped(tmp_path: Path) -> None:
    v, n, f = load_glb(_model_dir(tmp_path) / "model.glb", skip_layers=("ceilings",))
    assert len(f) == 12  # the wall box only
    np.testing.assert_allclose(v.min(0), [-2.0, -0.1, 0.0], atol=1e-6)  # gltf z -> dimos -y
    np.testing.assert_allclose(v.max(0), [2.0, 0.1, 2.0], atol=1e-6)  # gltf y -> dimos z
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_a_fix_places_the_model_mesh_in_world_frame(tmp_path: Path) -> None:
    m = ModelMesh(chunk_m=1.0, model_pose=(0.0, 0.0, 0.0, 90.0))
    published: list = []
    disposables: list = []
    m.mesh = SimpleNamespace(publish=published.append)  # type: ignore[assignment]
    m.loaded_map = SimpleNamespace(publish=lambda _: None)  # type: ignore[assignment]
    m.register_disposable = disposables.append  # type: ignore[method-assign, assignment]
    try:
        m._load_premap(str(_model_dir(tmp_path)))
        fix = np.eye(4)
        fix[:3, 3] = [10.0, 0.0, -0.4]
        m.submit(Transform.from_matrix(fix, frame_id="world", child_frame_id="map"))
        assert len(published) > 1  # chunked
        assert len({msg.key for msg in published}) == len(published)
        v = np.concatenate([msg.vertices for msg in published])
        # model_pose turns the wall onto map y, the fix shifts it 10 m along odom x
        np.testing.assert_allclose(v.min(0), [9.9, -2.0, -0.4], atol=1e-5)
        np.testing.assert_allclose(v.max(0), [10.1, 2.0, 1.6], atol=1e-5)
        assert sum(len(msg.faces) for msg in published) == 12
    finally:
        for d in disposables:
            d.dispose()
        m.dispose()


# A slice of a real walk against the Athens map directory (reconstruction/pipeline/model_map.py); skipped without them.
ATHENS_DB = Path.home() / "coding/go2web/recordings/athens_walk_1.db"
ATHENS_MAP = DIMOS_PROJECT_ROOT / "reconstruction/data/map"
# The walk's odom in the map frame, from the multi-session alignment the model was built on
ATHENS_WALK_1 = np.array(
    [
        [0.999998800775903, -1.099667124823122e-05, -0.001548652907964099, 0.0],
        [-1.099667124823122e-05, 0.9998991624844508, -0.014200871169313772, 0.0],
        [0.001548652907964099, 0.014200871169313772, 0.9998979632603536, 0.41388580787166446],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _window(db: Path, t_from: float, seconds: float) -> tuple[PointCloud2, np.ndarray]:
    """pointlio_lidar scans registered into odom by pointlio_odometry, and the sensor's position at the end."""
    from scipy.spatial.transform import Rotation, Slerp

    from dimos.memory.store.sqlite import SqliteStore
    from dimos.msgs.nav_msgs.Odometry import Odometry

    store = SqliteStore(path=str(db), must_exist=True)
    try:
        odom = [
            (o.ts, o.data.pose.pose)
            for o in store.stream("pointlio_odometry", Odometry).order_by("ts")
        ]
        ts = np.array([t for t, _ in odom])
        pos = np.array([[p.position.x, p.position.y, p.position.z] for _, p in odom])
        rot = Slerp(
            ts,
            Rotation.from_quat(
                [
                    [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                    for _, p in odom
                ]
            ),
        )
        t0 = ts[0] + t_from
        chunks = []
        for obs in store.stream("pointlio_lidar", PointCloud2).order_by("ts"):
            if obs.ts < t0:
                continue
            if obs.ts > t0 + seconds:
                break
            pts = obs.data.points_f32().astype(np.float64)
            chunks.append(
                rot(obs.ts).apply(pts) + [np.interp(obs.ts, ts, pos[:, k]) for k in range(3)]
            )
        p_end = np.array([np.interp(t0 + seconds, ts, pos[:, k]) for k in range(3)])
    finally:
        store.stop()
    local = PointCloud2.from_numpy(np.concatenate(chunks), frame_id="odom", timestamp=t0 + seconds)
    return local.voxel_downsample(0.05), p_end


@pytest.mark.skipif(
    not (ATHENS_DB.exists() and (ATHENS_MAP / "lidar.pc2.lcm").exists()),
    reason="needs the Athens walk and map",
)
def test_athens_walk_places_the_model_where_the_alignment_put_it() -> None:
    from dimos.mapping.relocalization.lidar.relocalize import LidarRelocalizer

    m = ModelMesh(map_file=str(ATHENS_MAP), world_frame="odom")
    published: list = []
    disposables: list = []
    m.mesh = SimpleNamespace(publish=published.append)  # type: ignore[assignment]
    m.loaded_map = SimpleNamespace(publish=lambda _: None)  # type: ignore[assignment]
    m.register_disposable = disposables.append  # type: ignore[method-assign, assignment]
    try:
        m._load_premap(m.config.map_file)  # type: ignore[arg-type]
        assert m.premap is not None
        m._relocalizer = LidarRelocalizer(m.premap.pointcloud, m.config.relocalize)
        fixes: list = []
        m.fixes.subscribe(fixes.append)
        local, p = _window(ATHENS_DB, t_from=3.0, seconds=5.0)
        m._relocalize(local)
        assert len(fixes) == 1, "refused"
        map_from_odom = np.linalg.inv(fixes[0].to_matrix())
        err_m = np.linalg.norm(
            map_from_odom[:3, :3] @ p
            + map_from_odom[:3, 3]
            - (ATHENS_WALK_1[:3, :3] @ p + ATHENS_WALK_1[:3, 3])
        )
        R = map_from_odom[:3, :3] @ ATHENS_WALK_1[:3, :3].T
        assert err_m < 0.05
        assert abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) < 1.0
        assert published and all(
            len(msg.vertices) for msg in published
        )  # the model went out on that fix
    finally:
        for d in disposables:
            d.dispose()
        m.dispose()
