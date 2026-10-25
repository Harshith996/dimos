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

import json
from pathlib import Path

from dimos_lerobot.act_loop import prepare_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest

from dimos.imitation.loop.recording import Episode, encode_image, export_snapshot
from dimos.imitation.loop.transport import learning_bus


@pytest.fixture(autouse=True)
def isolated_bus():
    with learning_bus():
        yield


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    contract = dict(
        split="train",
        fps=20,
        joint_names=[f"joint{i}" for i in range(7)],
        image_shape=[64, 64, 3],
        robot_type="openyam",
        action_semantics="absolute_public_joint_position",
        observation_profile="test",
        artifact_contract_sha256="test",
        task="move the cube",
    )
    episode = Episode(tmp_path, contract)
    for index, authority in enumerate(["human"] * 3 + ["policy"] * 2 + ["correction"] * 3):
        timestamp = 1.0 + index / 20
        episode.append(
            dict(
                state=[float(index)] * 7,
                state_ts=timestamp,
                image_ts=timestamp,
                command_ts=timestamp,
                command_sequence=index,
                sim_generation=0,
                accepted_action=[index + 0.1] * 7,
                authority=authority,
                applied=True,
                segment={"human": 0, "policy": 1, "correction": 2}[authority],
                image_png=encode_image(np.full((64, 64, 3), index, dtype=np.uint8)),
            )
        )
    episode.finish(keep=True)
    return export_snapshot(tmp_path, tmp_path / "snapshot")


def test_official_dataset_pads_chunks_at_human_correction_boundaries(snapshot: Path) -> None:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    root = prepare_dataset(snapshot, manifest)
    dataset = LeRobotDataset(
        repo_id="local/openyam",
        root=root,
        delta_timestamps={"action": [i / 20 for i in range(4)]},
        video_backend="pyav",
    )
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 6
    end_demo = dataset[2]
    np.testing.assert_allclose(end_demo["action"].numpy()[:, 0], [2.1] * 4)
    assert end_demo["action_is_pad"].tolist() == [False, True, True, True]
    first_correction = dataset[3]
    np.testing.assert_allclose(first_correction["action"].numpy()[:, 0], [5.1, 6.1, 7.1, 7.1])
    assert first_correction["action_is_pad"].tolist() == [False, False, False, True]
    assert first_correction["observation.images.wrist"].shape == (3, 64, 64)
    assert prepare_dataset(snapshot, manifest) == root
    (snapshot / "manifest.json").write_text(json.dumps({**manifest, "frames": 999}))
    with pytest.raises(ValueError, match="selection was modified"):
        prepare_dataset(snapshot, manifest)
