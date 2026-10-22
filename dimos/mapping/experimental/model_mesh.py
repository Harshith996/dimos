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

"""A hand-built model in place of the lidar mesh, placed by relocalizing against the recorded map it was built on.

``map_file`` names a directory holding ``lidar.pc2.lcm`` (the recorded map, frame ``map``) and ``model.glb``
(the model, authored in that frame), or the pair as ``"<cloud>.pc2.lcm,<mesh>.glb"``. The live local map is
matched against the recorded map like any premap: ``world_frame -> map`` fixes on ``tf``, the recorded map on
``loaded_map`` for the ray tracer. A clean model matches a live scan poorly (no clutter, no outdoors, surfaces
off by 10-30 cm), so it is never matched; it rides on the fix at a static ``model_pose`` in the map frame and goes
out on ``mesh`` in ``world_frame`` with its material colours, chunked like :class:`Mesh` so it replaces the
lidar mesh in a viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.core.stream import Out
from dimos.mapping.experimental.mesh import C
from dimos.mapping.relocalization.lidar.module import LocalMapConfig, LocalMapRelocalization
from dimos.mapping.relocalization.lidar.relocalize import MID360_NAV, RelocalizeConfig
from dimos.mapping.relocalization.module import MAP_SUFFIX, fix_stream
from dimos.msgs.shape_msgs.TriangleMesh import TriangleMesh
from dimos.utils.data import resolve_named_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dimos.msgs.geometry_msgs.Transform import Transform

MAP_CLOUD = "lidar.pc2.lcm"
MODEL_MESH = "model.glb"
# glTF is y-up, dimos z-up: (x, y, z)_gltf -> (x, -z, y)
GLTF_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def model_paths(spec: str) -> tuple[Path, Path]:
    """(recorded map, model mesh) from a directory or a ``cloud,mesh`` pair."""
    if "," in spec:
        cloud, mesh = (s.strip() for s in spec.split(",", 1))
        return resolve_named_path(cloud, MAP_SUFFIX), resolve_named_path(mesh)
    path = resolve_named_path(spec)
    if not path.is_dir():
        raise ValueError(
            f"model map {spec!r}: a directory with {MAP_CLOUD} + {MODEL_MESH}, or 'cloud,mesh'"
        )
    return path / MAP_CLOUD, path / MODEL_MESH


def load_glb(
    path: Path, skip_layers: tuple[str, ...] = ()
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.uint8]]:
    """Vertices, normals, faces, sRGB colours of every node baked to world, z-up.

    Nodes are named ``<layer>/<material>``; a vertex takes its material's base colour.
    """
    import trimesh

    scene = trimesh.load_scene(path, process=False)
    vs, ns, fs, cs, n = [], [], [], [], 0
    for name in scene.graph.nodes_geometry:
        if name.partition("/")[0] in skip_layers:
            continue
        tf, gname = scene.graph[name]
        geo = scene.geometry[gname].copy()
        geo.apply_transform(tf)
        vs.append(np.asarray(geo.vertices) @ GLTF_TO_ZUP.T)
        ns.append(np.asarray(geo.vertex_normals) @ GLTF_TO_ZUP.T)
        fs.append(np.asarray(geo.faces, dtype=np.int64) + n)
        cs.append(np.tile(_base_colour(geo), (len(geo.vertices), 1)))
        n += len(geo.vertices)
    if not vs:
        raise ValueError(f"{path}: no mesh outside {skip_layers}")
    return np.concatenate(vs), np.concatenate(ns), np.concatenate(fs), np.concatenate(cs)


def _base_colour(geo: Any) -> NDArray[np.uint8]:
    """The material's glTF base colour (linear) as sRGB bytes, grey without one."""
    factor = getattr(getattr(geo.visual, "material", None), "baseColorFactor", None)
    c = np.asarray(factor if factor is not None else [200, 200, 200, 255], np.float64)[:3]
    c = c / 255 if c.max() > 1 else c
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def chunk_mesh(
    vertices: NDArray[np.float64], faces: NDArray[np.int64], edge: float
) -> list[tuple[tuple[int, int, int], NDArray[np.int64], NDArray[np.int64]]]:
    """(key, vertex ids, local faces) per ``edge``-sized cube, each face in the cube holding its centroid."""
    keys = np.floor(vertices[faces].mean(1) / edge).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out = []
    for k, key in enumerate(uniq):
        used, local = np.unique(faces[inv.ravel() == k], return_inverse=True)
        out.append(((int(key[0]), int(key[1]), int(key[2])), used, local.reshape(-1, 3)))
    return out


def pose_matrix(x: float, y: float, z: float, yaw_deg: float) -> NDArray[np.float64]:
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = x, y, z
    return T


class ModelMeshConfig(LocalMapConfig):
    # Map directory or "cloud.pc2.lcm,mesh.glb" pair, see model_paths
    map_file: str | None = None
    relocalize: RelocalizeConfig = MID360_NAV
    # Keep matching to correct drift
    relocalize_once: bool = False
    reloc_interval: float = 10.0
    # Seconds between re-emits of the placed model; it also goes out on every fix
    mesh_interval: float = 30.0
    # The model in the map frame as (x, y, z, yaw deg); zero when it was authored in that frame
    model_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # glTF layers left out of the mesh
    skip_layers: tuple[str, ...] = ("ceilings",)
    # Chunk edge, metres: Mesh's 32 voxels at the nav voxel
    chunk_m: float = C * 0.08


class ModelMesh(LocalMapRelocalization):
    """Relocalizes the ray tracer's local map against the recorded map and publishes the model on ``mesh``."""

    config: ModelMeshConfig

    mesh: Out[TriangleMesh]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chunks: list[
            tuple[
                tuple[int, int, int],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.int64],
                NDArray[np.uint8],
            ]
        ] = []

    def _load_premap(self, map_file: str) -> None:
        cloud, glb = model_paths(map_file)
        super()._load_premap(str(cloud))
        v, n, f, rgb = load_glb(glb, self.config.skip_layers)
        P = pose_matrix(*self.config.model_pose)
        v, n = v @ P[:3, :3].T + P[:3, 3], n @ P[:3, :3].T
        self._chunks = [
            (key, v[ids], n[ids], local, rgb[ids])
            for key, ids, local in chunk_mesh(v, f, self.config.chunk_m)
        ]
        self.register_disposable(
            fix_stream(self.fixes, self.config.mesh_interval).subscribe(self._emit)
        )

    def placed_mesh(self, fix: Transform, ts: float | None = None) -> list[TriangleMesh]:
        """The model chunks in ``world_frame``, placed by a ``world_frame -> map`` fix."""
        T = fix.to_matrix()
        R, t = T[:3, :3], T[:3, 3]
        return [
            TriangleMesh(v @ R.T + t, f.astype(np.uint32), n @ R.T, key, ts, rgb)
            for key, v, n, f, rgb in self._chunks
        ]

    def _emit(self, fix: Transform) -> None:
        for msg in self.placed_mesh(fix):
            self.mesh.publish(msg)
