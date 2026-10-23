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

"""Exercise physical gripper travel and feedback without meshes or a viewer."""

from contextlib import ExitStack
import time

import pytest

from dimos.hardware.manipulators.sim.adapter import ShmMujocoAdapter
import dimos.simulation.engines.mujoco_sim_module as sim_module

# Two coupled sliding fingers. The xArm-like actuator closes with increasing
# ctrl; the Piper-like one opens. Both have a 2cm closed and 8cm open separation.
_SCENE = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint damping="1"/>
    <geom type="sphere" size="0.005" mass="0.1" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <camera name="wrist_camera"/>
    <body name="arm">
      <joint name="arm_joint" type="hinge" axis="0 0 1"/>
      <geom/>
      <body name="left_finger" pos="-{start_x} 0 0">
        <joint name="left_slide" type="slide" axis="{left_axis} 0 0" range="0 0.03"/>
        <geom/>
      </body>
      <body name="right_finger" pos="{start_x} 0 0">
        <joint name="right_slide" type="slide" axis="{axis} 0 0" range="0 0.03"/>
        <geom/>
      </body>
    </body>
  </worldbody>
  <equality><joint joint1="right_slide" joint2="left_slide"/></equality>
  <actuator>
    <position joint="arm_joint" kp="100"/>
    <general name="gripper" joint="left_slide" ctrlrange="0 {ctrl_max}"
             gainprm="{gain}" biastype="affine" biasprm="0 -100 -2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(params=[(True, 255.0, 0.04, -1), (False, 0.03, 0.01, 1)], ids=["xarm", "piper"])
def simulated_gripper(request, tmp_path, mocker):
    gripper_reversed, ctrl_max, start_x, axis = request.param
    scene = tmp_path / "gripper.xml"
    scene.write_text(
        _SCENE.format_map(
            {
                "start_x": start_x,
                "axis": axis,
                "left_axis": -axis,
                "ctrl_max": ctrl_max,
                "gain": 100 * 0.03 / ctrl_max,
            }
        )
    )
    engine_factory = mocker.spy(sim_module, "MujocoEngine")
    # Both SHM ends share this process's resource tracker. Keep the writer's
    # registration until sim.stop() unlinks the buffers.
    mocker.patch("dimos.utils.shm.unregister", side_effect=lambda shm: shm)
    with ExitStack() as cleanup:
        sim = sim_module.MujocoSimModule(
            address=str(scene),
            dof=1,
            gripper_reversed=gripper_reversed,
            headless=True,
            enable_color=False,
            enable_depth=False,
        )
        cleanup.callback(sim.stop)
        sim.start()
        adapter = ShmMujocoAdapter(dof=2, address=str(scene))
        cleanup.callback(adapter.disconnect)
        assert adapter.connect()
        yield adapter, engine_factory.spy_return


def test_gripper_commands_move_jaws_and_report_matching_opening(simulated_gripper):
    adapter, engine = simulated_gripper
    limits = adapter.get_limits()
    low, high = limits.position_lower[-1], limits.position_upper[-1]
    assert (low, high) == pytest.approx((0.0, 0.03))

    for opening in (1.0, 0.0, 0.5, 1.0):
        assert adapter.write_joint_positions([0.0, low + opening * (high - low)])
        expected_gap = 0.02 + opening * 0.06
        deadline = time.monotonic() + 5.0
        while True:
            left, _ = engine.get_body_pose("left_finger")
            right, _ = engine.get_body_pose("right_finger")
            gap = float(right[0] - left[0])
            measured = (adapter.read_joint_positions()[-1] - low) / (high - low)
            if abs(gap - expected_gap) < 0.001 and abs(measured - opening) < 0.02:
                break
            assert time.monotonic() < deadline, (opening, gap, measured)
            time.sleep(0.01)

        assert gap == pytest.approx(expected_gap, abs=0.001)
        assert measured == pytest.approx(opening, abs=0.02)
