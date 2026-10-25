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
import json
from pathlib import Path
import time

from mcap.reader import make_reader
import numpy as np
import pytest

from dimos.imitation.collection.recording import RecordingSchema
from dimos.imitation.dataprep.core import extract_episodes, iter_episode_samples
from dimos.imitation.dataprep.sources.mcap import open_recording
from dimos.imitation.loop.transport import learning_bus
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.dual_openyam.collection import DualOpenYamCollection
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_JOINTS
from dimos.teleop.webxr.controller_types import BUTTON_ALIASES, Buttons

pytestmark = pytest.mark.native_e2e


@pytest.fixture
def collection(tmp_path, mocker):
    clock = mocker.patch("dimos.robot.manipulators.dual_openyam.collection.time", wraps=time)
    clock.time.return_value = 1000.0
    with learning_bus():
        module = DualOpenYamCollection(
            directory=tmp_path, width=8, height=8, fps=8, camera_ids=("usb-one", "usb-two")
        )
        try:
            yield module, clock
        finally:
            module.stop()


def feedback(module, timestamp):
    # Reverse the measured order: the schema must align by joint name.
    module._joints(
        "state",
        JointState(
            ts=timestamp,
            name=list(reversed(DUAL_OPENYAM_JOINTS)),
            position=list(reversed([i / 10 for i in range(14)])),
        ),
    )
    for name, value in (("camera1", 30), ("camera2", 180)):
        module._image(
            name,
            Image.from_numpy(
                np.full((8, 8, 3), value, dtype=np.uint8), format=ImageFormat.RGB, ts=timestamp
            ),
        )


def ready(module, clock):
    feedback(module, clock.time.return_value)
    module._joints(
        "action",
        JointState(
            ts=clock.time.return_value - 0.1,
            name=DUAL_OPENYAM_JOINTS,
            position=[i / 10 + 0.05 for i in range(14)],
        ),
    )


def press(module, alias):
    buttons = Buttons()
    buttons.set_attribute(BUTTON_ALIASES[alias], True)
    module._buttons(buttons)


def test_saved_take_round_trips_both_images_states_and_held_accepted_actions(collection):
    module, clock = collection
    ready(module, clock)
    started = module.record("put the object in the tray", split="eval")
    assert started["recording"] is not None
    assert started["episodes"] == []
    for index in range(4):
        clock.time.return_value = 1000.0 + index / 8
        feedback(module, clock.time.return_value)
        module._joints(
            "action",
            JointState(
                ts=clock.time.return_value, name=["left_joint1"], position=[0.4 + index / 10]
            ),
        )
    clock.time.return_value += 1 / 8
    saved = module.finish(True, outcome="success")
    assert saved["recording"] is None
    (episode,) = saved["episodes"]
    assert episode["camera_ids"] == ["usb-one", "usb-two"]
    assert episode["split"] == "eval"
    assert episode["outcome"] == "success"
    assert episode["stream_counts"] == {
        "state": 4,
        "action": 5,
        "camera1": 4,
        "camera2": 4,
        "status": 2,
    }
    path = Path(episode["path"])
    schema = RecordingSchema.read(path.parent)
    with ExitStack() as stack:
        store = open_recording(path)
        stack.callback(store.stop)
        (take,) = extract_episodes(store, schema.episodes)
        samples = list(
            iter_episode_samples(
                store,
                take,
                {**schema.observation, **schema.action},
                schema.sync,
                schema.quality,
                obs_keys=set(schema.observation),
                action_keys=set(schema.action),
            )
        )
    assert len(samples) == 4
    np.testing.assert_allclose(samples[0].observation["observation.state"], np.arange(14) / 10)
    np.testing.assert_allclose(
        samples[0].action["action"], [0.4, *[i / 10 + 0.05 for i in range(1, 14)]]
    )
    np.testing.assert_allclose(
        samples[-1].action["action"], [0.7, *[i / 10 + 0.05 for i in range(1, 14)]]
    )
    for key, value in (("camera1", 30), ("camera2", 180)):
        image = samples[0].observation[f"observation.images.{key}"]
        np.testing.assert_allclose(image, value, atol=2)


@pytest.mark.parametrize("alias", ["Y", "B"])
def test_discard_and_zero_button_never_accept_an_episode(collection, alias):
    module, clock = collection
    ready(module, clock)
    press(module, "A")
    directory = module.config.directory / module.get_status()["recording"]
    feedback(module, clock.time.return_value)
    press(module, alias)
    state = module.get_status()
    assert state["recording"] is None
    assert state["episodes"] == []
    assert json.loads((directory / "metadata.json").read_text())["kept"] is False


def test_feedback_loss_discards_instead_of_saving_a_broken_take(collection):
    module, clock = collection
    ready(module, clock)
    module.record("move object")
    clock.time.return_value += 2
    module._health()
    state = module.get_status()
    assert state["recording"] is None
    assert state["collection_ready"] is False
    assert "discarded" in state["collection_error"]
    assert state["episodes"] == []


def test_failed_native_receipt_cannot_create_a_saved_manifest(collection, mocker):
    module, clock = collection
    ready(module, clock)
    module.record("move object")
    feedback(module, clock.time.return_value)
    mocker.patch.object(
        module._capture._recorder, "wait_written", side_effect=RuntimeError("write failed")
    )
    with pytest.raises(RuntimeError, match="write failed"):
        module.finish(True)
    assert module.get_status()["episodes"] == []
    assert list(module.config.directory.glob("*/metadata.json")) == []
    (path,) = module.config.directory.glob("*/recording.mcap")
    with path.open("rb") as source:
        assert make_reader(source).get_summary() is not None


def test_record_refuses_uninitialized_targets_without_ever_using_feedback_as_action(collection):
    module, clock = collection
    feedback(module, clock.time.return_value)
    assert module.get_status()["collection_ready"] is False
    with pytest.raises(RuntimeError, match="Engage both arms"):
        module.record("move object")
    assert list(module.config.directory.iterdir()) == []
