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

"""The navigation stack, as ports: a stream in, a stream out, and tf.

A local planner is blind or map-aware; both hand the follower the same ``path``.

A ``path`` with one pose means "hold, no safe route"; with none, "stop".

tf is ``IO`` where the implementers are dimos-module natives (``#[tf]`` both
subscribes and publishes) and their python twins; ``In`` on the global planner,
whose native only reads it.
"""

import sys
from typing import Protocol

from dimos_lcm.actionlib_msgs import GoalStatus

from dimos.core.stream import IO, In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.types.timestamped import Timestamped

if sys.version_info >= (3, 13):
    from typing import is_protocol
else:
    from typing_extensions import is_protocol

PREEMPT_REPLACED = "replaced"
PREEMPT_CANCELED = "canceled"
_NAVIGATION_SKILLS = ("go_to", "stop_navigation")


def goal_id(goal: Timestamped) -> str:
    """The id nav_status reports carry for a goal, which is its stamp."""
    sec, nsec = goal.ros_timestamp()
    return f"{sec}.{nsec:09d}"


class GlobalPlanner(Protocol):
    """Plans to goals and is the agent's navigation interface.

    nav_status reports each goal under goal_id of the message that set it.
    A PREEMPTED report's text is PREEMPT_REPLACED or PREEMPT_CANCELED.
    """

    tf: In[TFMessage]
    goal: In[PointStamped]

    path: Out[Path]
    nav_status: Out[GoalStatus]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if is_protocol(cls):
            return
        # nice way to verify that each planner has the functions specifically as skills
        missing = [
            n for n in _NAVIGATION_SKILLS if not getattr(getattr(cls, n, None), "__skill__", False)
        ]
        if missing:
            raise TypeError(f"{cls.__name__} must implement {', '.join(missing)} as @skill")

    async def go_to(self, x: float, y: float) -> str:
        """Drive to a point in the world frame, replacing any goal under way."""

    async def stop_navigation(self) -> str:
        """Cancel the current goal and stop moving."""


class BlindLocalPlanner(Protocol):
    """Shapes the global route without looking: no map."""

    planner_path: In[Path]
    tf: In[TFMessage]

    path: Out[Path]


class MapLocalPlanner(Protocol):
    """Routes around what the local map says is there."""

    planner_path: In[Path]
    local_map: In[PointCloud2]
    tf: IO[TFMessage]

    path: Out[Path]


class TrajectoryFollower(Protocol):
    tf: IO[TFMessage]
    path: In[Path]

    nav_cmd_vel: Out[Twist]
