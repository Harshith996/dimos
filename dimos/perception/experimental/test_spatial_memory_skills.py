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

"""The agent skills on SpatialMemory, with the vector database and CLIP mocked out."""

from collections.abc import Generator
from typing import Any, NamedTuple
from unittest.mock import MagicMock, PropertyMock

import pytest
from pytest_mock import MockerFixture

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.spatial_perception import SpatialMemory
from dimos.types.robot_location import RobotLocation

_MODULE = "dimos.perception.experimental.spatial_perception"


class _Rig(NamedTuple):
    module: SpatialMemory
    db: MagicMock


@pytest.fixture()
def memory(mocker: MockerFixture) -> Generator[_Rig, None, None]:
    mocker.patch(f"{_MODULE}.ImageEmbeddingProvider")
    mocker.patch(f"{_MODULE}.SpatialVectorDB")
    mocker.patch(f"{_MODULE}.VisualMemory")
    module = SpatialMemory(chroma_client=MagicMock())
    db = MagicMock()
    db.query_tagged_location.return_value = (None, 0.0)
    db.query_by_text.return_value = []
    module.vector_db = db
    try:
        yield _Rig(module, db)
    finally:
        module._close_module()


def _seen(distance: float, x: float, y: float) -> list[dict[str, Any]]:
    return [{"distance": distance, "metadata": [{"pos_x": x, "pos_y": y}]}]


def test_find_location_prefers_a_tagged_location(memory: _Rig) -> None:
    kitchen = RobotLocation(name="kitchen", position=(1.0, 2.0, 0.0), rotation=(0.0, 0.0, 0.0))
    memory.db.query_tagged_location.return_value = (kitchen, 0.1)
    memory.db.query_by_text.return_value = _seen(0.5, 9.0, 9.0)

    assert memory.module.find_location("kitchen") == (
        "Tagged location 'kitchen' at (1.00, 2.00) in the world frame."
    )


def test_find_location_falls_back_to_what_the_camera_saw(memory: _Rig) -> None:
    memory.db.query_by_text.return_value = _seen(0.7, 3.0, -1.5)

    assert memory.module.find_location("red couch") == (
        "Seen by the camera at (3.00, -1.50) in the world frame, similarity 0.30."
    )


@pytest.mark.parametrize(
    "seen",
    [pytest.param([], id="nothing seen"), pytest.param(_seen(0.9, 3.0, 1.0), id="weak match")],
)
def test_find_location_reports_no_match(memory: _Rig, seen: list[dict[str, Any]]) -> None:
    memory.db.query_by_text.return_value = seen

    assert (
        memory.module.find_location("piano")
        == "No tagged location or remembered view matches 'piano'."
    )


def test_tag_location_stores_the_robot_position(memory: _Rig, mocker: MockerFixture) -> None:
    tf = MagicMock()
    tf.get.return_value = Transform(translation=Vector3(1.5, -2.0, 0.3))
    mocker.patch.object(SpatialMemory, "tfbuffer", new_callable=PropertyMock, return_value=tf)

    assert memory.module.tag_location("kitchen") == "Tagged 'kitchen' at (1.50, -2.00)."
    (location,), _ = memory.db.tag_location.call_args
    assert (location.name, location.position) == ("kitchen", (1.5, -2.0, 0.3))


def test_tag_location_without_a_pose_tags_nothing(memory: _Rig, mocker: MockerFixture) -> None:
    tf = MagicMock()
    tf.get.return_value = None
    mocker.patch.object(SpatialMemory, "tfbuffer", new_callable=PropertyMock, return_value=tf)

    assert memory.module.tag_location("kitchen") == "Nothing tagged: no robot position on tf."
    memory.db.tag_location.assert_not_called()
