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

"""Episode recording for Dual OpenYAM teleoperation.

`DualOpenYamCollectionRecorder` captures the three headset camera streams,
measured joint state, commanded joint targets, and episode status into a session database.
`LEROBOT_DATAPREP_CONFIG` is the matching `dimos dataprep` configuration that
exports saved episodes as a LeRobot dataset.
"""

from pathlib import Path

from dimos.core.stream import In
from dimos.imitation.collection.recorder import EpisodeRecorder
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState

# The DataPrep config that turns a session recorded by the recorder below into
# a LeRobot dataset: `dimos dataprep build --config <this file> --source <db>`.
LEROBOT_DATAPREP_CONFIG = Path(__file__).with_name("lerobot_dataprep.json")

# Camera ports, named after the teleop blueprint's panel streams so they
# autoconnect to the TeleopCamera instances.
DUAL_OPENYAM_CAMERA_STREAMS = ("egocentric_image", "left_wrist_image", "right_wrist_image")
# Everything the recorder captures; none of it is anchored to a world pose.
DUAL_OPENYAM_COLLECTION_STREAMS = (
    *DUAL_OPENYAM_CAMERA_STREAMS,
    "coordinator_joint_state",
    "coordinator_joint_command",
    "status",
)


class DualOpenYamCollectionRecorder(EpisodeRecorder):
    """Record cameras, measured state, commanded targets, and episode status."""

    # Own process: three JPEG streams plus SQLite writes must not share a GIL
    # with the control coordinator.
    dedicated_worker = True

    egocentric_image: In[Image]
    left_wrist_image: In[Image]
    right_wrist_image: In[Image]
    coordinator_joint_command: In[JointState]
