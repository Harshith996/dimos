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

import numpy as np
import pytest

from dimos.msgs.shape_msgs.TriangleMesh import TriangleMesh


def _mesh(colors: np.ndarray | None) -> TriangleMesh:
    v = np.arange(12, dtype=np.float32).reshape(4, 3)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    return TriangleMesh(v, f, v / 10, key=(1, -2, 3), ts=12.5, colors=colors)


@pytest.mark.parametrize(
    "colors", [None, np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 8, 7]], np.uint8)]
)
def test_round_trip_with_and_without_colours(colors: np.ndarray | None) -> None:
    sent = _mesh(colors)
    got = TriangleMesh.lcm_decode(sent.lcm_encode())
    np.testing.assert_array_equal(got.vertices, sent.vertices)
    np.testing.assert_array_equal(got.normals, sent.normals)
    np.testing.assert_array_equal(got.faces, sent.faces)
    assert (got.key, got.ts) == ((1, -2, 3), 12.5)
    if colors is None:
        assert got.colors is None
    else:
        np.testing.assert_array_equal(got.colors, colors)


def test_colours_trail_the_old_layout() -> None:
    """A coloured message is the uncoloured one plus 3 bytes a vertex, so old decoders read it unchanged."""
    plain = _mesh(None).lcm_encode()
    coloured = _mesh(np.zeros((4, 3), np.uint8)).lcm_encode()
    assert coloured[: len(plain)] == plain and len(coloured) == len(plain) + 12
