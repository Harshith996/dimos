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

"""The dual OpenYAM grasp stack with an MCP agent, driven from ``dimos humancli``.

```bash
dimos run dual-openyam-grasp-agent --config workcell.json --left-can-port follower_l --right-can-port follower_r
dimos humancli                                   # second terminal, same machine
```

Needs ``OPENAI_API_KEY`` in the environment or ``.env`` of the directory
``dimos run`` starts in.
"""

from __future__ import annotations

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.common.agent_prompts import MANIPULATION_AGENT_SYSTEM_PROMPT
from dimos.robot.manipulators.dual_openyam.blueprints.grasp import (
    DUAL_OPENYAM_TABLE_TOP_Z,
    dual_openyam_grasp,
    dual_openyam_grasp_graspgenx,
)

_DUAL_ARM_ADDENDUM = """\

# This rig
Two OpenYAM arms and a fixed overhead RealSense on a post between them, not a single xArm \
with a wrist camera. Every skill that moves an arm or a gripper takes planning_group: \
left_manipulator when the object's Y is positive, right_manipulator when it is negative.
World frame (meters): origin midway between the arm bases at the height of their mounting \
blocks, X forward (away from the arms), Y toward the left arm, Z up. The table top is at \
Z={table_z}. The yellow bin spans X 0.40 to 0.69 and Y -0.11 to +0.11, rim at Z=0.068; it \
is an obstacle, not a place target: this stack places top-down only and cannot clear the \
rim. "put it in the tray", "put it down" and "put it in the bin" all mean place_at \
{left_drop} for the left arm or place_at {right_drop} for the right arm.
"""

# Tray square on each arm's side, inside the fingers-down reach; z is the tcp height at release.
DUAL_OPENYAM_DROP_POSES = {"left": (0.26, 0.20, 0.05), "right": (0.26, -0.20, 0.05)}


def dual_openyam_agent_prompt() -> str:
    """Main's manipulation agent prompt, followed by what differs on this rig."""
    drops = {
        f"{side}_drop": " ".join(str(v) for v in pose) + f" {side}_manipulator"
        for side, pose in DUAL_OPENYAM_DROP_POSES.items()
    }
    return MANIPULATION_AGENT_SYSTEM_PROMPT + _DUAL_ARM_ADDENDUM.format(
        table_z=f"{DUAL_OPENYAM_TABLE_TOP_Z:.3f}", **drops
    )


# Assigned through autoconnect so the registry generator sees it.
dual_openyam_grasp_agent = autoconnect(
    dual_openyam_grasp,
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=dual_openyam_agent_prompt()),
)

dual_openyam_grasp_graspgenx_agent = autoconnect(
    dual_openyam_grasp_graspgenx,
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=dual_openyam_agent_prompt()),
)
