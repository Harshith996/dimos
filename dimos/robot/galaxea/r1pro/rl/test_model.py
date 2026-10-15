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


"""Checks for pinned import semantics, not hardware dynamics validation."""

import mujoco
import numpy as np
import pytest

from dimos.robot.galaxea.r1pro.rl import static_audit
from dimos.robot.galaxea.r1pro.rl.model import (
    ARM_JOINT_NAMES,
    collision_pairs,
    joint_addresses,
    load_model,
    rest_qpos,
)

pytestmark = pytest.mark.mujoco


def test_model_preserves_fixed_curriculum_and_vendor_inertials() -> None:
    model = load_model()
    assert (model.nq, model.nv, model.nu) == (14, 14, 14)
    assert tuple(model.joint(i).name for i in range(model.njnt)) == ARM_JOINT_NAMES
    assert model.body_mass.sum() == pytest.approx(92.83112138, abs=1e-8)
    assert np.all(model.body_inertia[1:] > 0)
    qpos, qvel = joint_addresses(model)
    np.testing.assert_array_equal(qpos, np.arange(14))
    np.testing.assert_array_equal(qvel, np.arange(14))
    np.testing.assert_allclose(model.actuator_ctrlrange, model.jnt_range)
    np.testing.assert_allclose(model.jnt_range[0], [-4.41569, 1.27409], atol=1e-5)


def test_neutral_fk_and_short_deterministic_simulation() -> None:
    model = load_model()
    first, second = mujoco.MjData(model), mujoco.MjData(model)
    for data in (first, second):
        data.qpos[:] = rest_qpos(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.site("left_wrist").xpos, [-0.079397, 0.25099366, 0.719602], atol=1e-7
        )
        np.testing.assert_allclose(
            data.site("right_wrist").xpos, [-0.07975203, -0.25300626, 0.719602], atol=1e-7
        )
        assert collision_pairs(model, data) == ()
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert data.time == pytest.approx(0.2)
    np.testing.assert_array_equal(first.qpos, second.qpos)


def test_full_source_to_import_audit() -> None:
    evidence = static_audit.audit_model()
    assert evidence["status"] == "static_import_checks_passed"
    assert len(evidence["links"]) == 43
    assert len(evidence["joint_origin_transforms_checked"]) == 42
    assert evidence["collision_geometries"] == 43
    assert evidence["noncolliding_visual_geometries"] == 43
    assert evidence["exclusion_pairs_checked"] == 28
    assert evidence["dynamics_validated"] is False


def _assert_corrupted_import_rejected(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    model = load_model()
    if fault == "mass":
        model.body_mass[model.body("left_arm_link1").id] += 0.1
    elif fault == "inertia":
        model.body_inertia[model.body("left_arm_link1").id] *= 2
    elif fault == "visual_mask":
        model.geom_contype[np.flatnonzero(model.geom_group == 1)[0]] = 1
    else:
        model.actuator_gainprm[0, 0] = 40
    monkeypatch.setattr(static_audit, "load_model", lambda path: model)
    with pytest.raises(AssertionError):
        static_audit.audit_model()


def test_static_audit_rejects_corrupted_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_corrupted_import_rejected(monkeypatch, "mass")


def test_static_audit_rejects_corrupted_inertia(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_corrupted_import_rejected(monkeypatch, "inertia")


def test_static_audit_rejects_visual_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_corrupted_import_rejected(monkeypatch, "visual_mask")


def test_static_audit_rejects_actuator_gain(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_corrupted_import_rejected(monkeypatch, "actuator_gain")
