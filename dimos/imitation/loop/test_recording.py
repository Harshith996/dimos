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

import numpy as np
import pytest

from dimos.imitation.loop.recording import (
    Episode,
    encode_image,
    export_snapshot,
    read_episode,
    snapshot_samples,
)
from dimos.imitation.loop.transport import learning_bus

pytestmark = pytest.mark.native_e2e


@pytest.fixture(autouse=True)
def isolated_bus():
    with learning_bus():
        yield


@pytest.fixture
def record(tmp_path):
    def create(
        authorities,
        *,
        split="train",
        keep=True,
        acknowledged=True,
        host_offset=0.0,
        applied_delay=0.0,
        segment_size=2,
    ):
        episode = Episode(
            tmp_path,
            dict(
                split=split,
                fps=20,
                joint_names=["joint", "gripper"],
                image_shape=[2, 3, 3],
                robot_type="openyam",
                action_semantics="absolute_native_joint_position",
                observation_profile="test",
                artifact_contract_sha256="test-artifact",
                task="move the cube",
            ),
        )
        for i, authority in enumerate(authorities):
            episode.append(
                dict(
                    command_ts=1 + i / 20 + host_offset,
                    applied_ts=1 + i / 20 + applied_delay,
                    command_age=0.01,
                    state_ts=1 + i / 20,
                    image_ts=1 + i / 20,
                    sim_generation=0,
                    state=[i, 1.0],
                    accepted_action=[i + 0.1, 0.9],
                    authority=authority,
                    applied=acknowledged,
                    segment=i // segment_size,
                    image_png=encode_image(np.full((2, 3, 3), i, np.uint8)),
                )
            )
        return episode.finish(keep=keep)

    return create


def test_export_excludes_policy_eval_discarded_and_unacknowledged_frames(tmp_path, record):
    record(["human", "human", "policy", "policy", "correction", "correction"])
    record(["human"], split="eval")
    record(["human"], keep=False)
    record(["human"], acknowledged=False)

    export_snapshot(tmp_path, tmp_path / "snapshot")

    samples = list(snapshot_samples(tmp_path / "snapshot"))
    assert [sample.observation["observation.state"][0] for sample in samples] == [0, 1, 4, 5]
    assert samples[0].episode_id == samples[1].episode_id
    assert samples[2].episode_id == samples[3].episode_id
    assert samples[1].episode_id != samples[2].episode_id
    manifest = json.loads((tmp_path / "snapshot/manifest.json").read_text())
    assert manifest["frames"] == 4


def test_snapshot_refuses_mutated_recording_and_existing_output(tmp_path, record):
    path = record(["human", "human"])
    export_snapshot(tmp_path, tmp_path / "snapshot")
    with pytest.raises(FileExistsError):
        export_snapshot(tmp_path, tmp_path / "snapshot")
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed after finalization"):
        export_snapshot(tmp_path, tmp_path / "other")


def test_empty_training_selection_is_an_error(tmp_path, record):
    record(["policy", "policy"])
    with pytest.raises(ValueError, match="no contiguous"):
        export_snapshot(tmp_path, tmp_path / "snapshot")


def test_host_dispatch_clock_does_not_reorder_simulation_training_samples(tmp_path, record):
    path = record(["human", "human"], host_offset=0.02)
    export_snapshot(tmp_path, tmp_path / "snapshot")
    samples = list(snapshot_samples(tmp_path / "snapshot"))
    assert len(samples) == 2
    np.testing.assert_allclose(samples[0].action["action"], [0.1, 0.9])
    _, frames = read_episode(path)
    assert frames[0]["command_ts"] > frames[0]["state_ts"]
    assert frames[0]["applied_ts"] == frames[0]["state_ts"]


def test_short_alignment_fragments_are_rejected_without_losing_valid_segments(tmp_path, record):
    record(["human"] * 5, segment_size=3, applied_delay=0.02)
    export_snapshot(tmp_path, tmp_path / "snapshot")
    manifest = json.loads((tmp_path / "snapshot/manifest.json").read_text())
    assert manifest["segments"] == 1
    assert manifest["frames"] == 2
    assert len(manifest["rejected_segments"]) == 1
    assert "fewer than two aligned frames" in manifest["rejected_segments"][0]["reasons"]
    assert len(list(snapshot_samples(tmp_path / "snapshot"))) == 2
