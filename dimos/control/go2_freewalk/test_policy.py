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

import struct

import numpy as np
import pytest

from dimos.control.go2_freewalk.models import load_weights
from dimos.control.go2_freewalk.policy import FreePolicy, Proprioception


@pytest.fixture
def blob():
    """Synthetic linear experts reveal history ordering and previous-action feedback."""
    data = bytearray(b"FREE" + struct.pack("<6I", 1, 6, 45, 12, 3, 16))

    def floats(values):
        data.extend(np.asarray(values, dtype="<f4").tobytes())

    def layer(weights, bias):
        data.extend(struct.pack("<3I", 1, *weights.shape))
        floats(weights)
        floats(bias)

    floats([100, 5])
    for values in (
        np.zeros(45),
        np.ones(45),
        np.zeros(12),
        np.ones(12),
        np.zeros(12),
        np.full(12, 40),
        np.ones(12),
        np.zeros(6),
    ):
        floats(values)
    data.extend(struct.pack("<I", 3))
    for kind in (0, 1, 3):
        data.extend(struct.pack("<I", kind))
        encoder = np.zeros((270, 19))
        encoder[45, 0] = 1  # previous frame command, not current frame
        encoder[225, 1] = 1  # oldest frame
        layer(encoder, np.r_[np.zeros(3), 3, np.zeros(15)])
        actor = np.zeros((64, 12))
        actor[45, 0] = 1
        actor[46, 1] = 1
        actor[48, 2] = 1  # normalized latent is exactly 1
        actor[33, 3] = 1  # prior raw action of joint 0
        actor[0, 4] = 1  # newest command
        layer(actor, np.r_[np.zeros(11), kind])
    return bytes(data)


def test_history_is_newest_first_and_reset_repeats_first_observation(blob):
    policy = FreePolicy(blob)
    obs = Proprioception(np.zeros(3), np.array([0, 0, -1.0]), np.zeros(12), np.zeros(12))
    first = policy.act(obs, np.array([0.2, 0, 0]))
    second = policy.act(obs, np.array([0.4, 0, 0]))
    np.testing.assert_allclose(first[:5], [0.2, 0.2, 1, 0, 0.2])
    np.testing.assert_allclose(second[:5], [0.2, 0.2, 1, 0.2, 0.4])
    policy.reset()
    np.testing.assert_allclose(policy.act(obs, np.array([0.2, 0, 0])), first)


@pytest.mark.parametrize(
    "command, kind", [([0.9, 0, 0], 0), ([1, 0, 0], 1), ([6, 0, 0], 1), ([0, 0, 0.2], 3)]
)
def test_expert_selection_matches_upstream_state_machine(blob, command, kind):
    assert FreePolicy(blob).band_for(np.array(command)) == kind


@pytest.mark.parametrize(
    "change", [lambda b: b[:20], lambda b: b[:-1], lambda b: b + b"x", lambda b: b"BAD!" + b[4:]]
)
def test_invalid_model_is_rejected(blob, change):
    with pytest.raises(ValueError):
        FreePolicy(change(blob))


def test_nonfinite_observation_does_not_poison_history(blob):
    policy = FreePolicy(blob)
    obs = Proprioception(np.zeros(3), np.array([0, 0, -1.0]), np.zeros(12), np.zeros(12))
    with pytest.raises(ValueError, match="observation"):
        policy.act(obs, np.array([np.nan, 0, 0]))
    np.testing.assert_allclose(policy.act(obs, np.zeros(3)), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_missing_model_requires_explicit_install(tmp_path):
    with pytest.raises(FileNotFoundError, match="--install"):
        load_weights(tmp_path / "missing.bin")


def test_different_policy_cannot_silently_replace_pinned_weights(tmp_path):
    path = tmp_path / "wrong.bin"
    path.write_bytes(b"not the private weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_weights(path)
