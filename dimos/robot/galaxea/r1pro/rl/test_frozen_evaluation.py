# Copyright 2025-2026 Dimensional Inc.
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


"""Verify the checked-in frozen suite against the pinned local MuJoCo model."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.rl.env import TrackingConfig
from dimos.robot.galaxea.r1pro.rl.evaluate import (
    FROZEN_MANIFEST_PATH,
    content_sha256,
    load_frozen_manifest,
    make_fk_source,
)
from dimos.robot.galaxea.r1pro.rl.model import load_model
from dimos.robot.galaxea.r1pro.rl.targets import EvaluationCase, FrozenTargetSource

pytestmark = pytest.mark.mujoco


def rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = content_sha256(payload)
    path.write_text(json.dumps(manifest))


def test_actual_pinned_samples_and_kinematic_evidence_verify() -> None:
    manifest = load_frozen_manifest(
        FROZEN_MANIFEST_PATH, load_model(), TrackingConfig(episode_steps=20), None
    )
    assert len(manifest["cases"]) == 4
    assert all(len(case["samples"]) == 21 for case in manifest["cases"])
    assert manifest["cases"][2]["samples"][8]["query_time_s"] == 0.16
    assert manifest["cases"][2]["samples"][8]["frame"] is None
    assert manifest["cases"][3]["samples"][8]["frame"]["stamp_s"] == -0.84
    for case in manifest["cases"]:
        evidence = case["sampled_kinematic_evidence"]
        assert evidence["samples"] == 401
        assert evidence["forbidden_collision_samples"] == 0
        assert evidence["minimum_joint_limit_margin_rad"] >= 0.02
        assert evidence["dynamics_validated"] is False
        assert evidence["continuous_collision_checked"] is False


def test_changed_model_source_and_control_config_are_rejected(tmp_path: Path) -> None:
    config = TrackingConfig(episode_steps=20)
    model = load_model()
    with pytest.raises(ValueError, match="configuration changed"):
        load_frozen_manifest(
            FROZEN_MANIFEST_PATH, model, replace(config, max_joint_target_rate_rad_s=0.3), None
        )
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text())
    manifest["identity"]["model_converter_sha256"] = "changed"
    path = tmp_path / "changed.json"
    rewrite_manifest(path, manifest)
    with pytest.raises(ValueError, match="identity changed"):
        load_frozen_manifest(path, model, config, None)


def test_tampered_sample_fails_even_after_json_digests_are_updated(tmp_path: Path) -> None:
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text())
    entry = manifest["cases"][0]
    entry["samples"][0]["frame"]["positions_m"][0][0] += 0.01
    entry["samples_sha256"] = content_sha256(entry["samples"])
    path = tmp_path / "changed.json"
    rewrite_manifest(path, manifest)
    with pytest.raises(ValueError, match="realized FK source"):
        load_frozen_manifest(path, load_model(), TrackingConfig(episode_steps=20), None)


def test_pretty_manifest_stays_under_size_limit_and_bounds_quantization() -> None:
    manifest = json.loads(FROZEN_MANIFEST_PATH.read_text())
    formatted = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert len(formatted.encode()) < 75 * 1024
    assert FROZEN_MANIFEST_PATH.read_text() == formatted
    entry = manifest["cases"][0]
    raw_source = make_fk_source(load_model(), EvaluationCase(**entry["case"]))
    frozen_source = FrozenTargetSource(entry["samples"])
    for sample in entry["samples"]:
        raw = raw_source.sample(sample["query_time_s"])
        frozen = frozen_source.sample(sample["query_time_s"])
        assert raw is not None and frozen is not None
        assert np.max(np.abs(raw.positions_m - frozen.positions_m)) <= 5.01e-11
        assert np.max(np.abs(raw.quaternions_wxyz - frozen.quaternions_wxyz)) <= 5.01e-11
        assert np.all(frozen.position_valid) and np.all(frozen.orientation_valid)
        assert frozen.valid and frozen.provenance == "procedural_fk"
        assert frozen.reference_frame == "world" and frozen.calibration_id is None
        assert set(sample["frame"]) == {"positions_m", "quaternions_wxyz", "stamp_s"}
