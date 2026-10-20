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

from dimos_cdrgen.definitions import Definitions
from dimos_cdrgen.registry import decode as cdr_decode, encode as cdr_encode
from mcap.reader import make_reader
import numpy as np
from dimos.messages.builtin_interfaces.msg import Time
from dimos.messages.nav_msgs.msg import OccupancyGrid
from dimos.messages.std_msgs.msg import Header
from dimos.msgs.occupancy import occupancy_from_array, occupancy_view
from dimos.protocol.cdr_mcap import CdrMcapWriter


def test_mcap_contains_generated_grid_and_full_viewer_schema_closure(tmp_path):
    message = occupancy_from_array(
        np.array([[-1, 0, 100]], dtype=np.int8),
        header=Header(frame_id="map", stamp=Time(sec=1700000000, nanosec=123456789)),
        resolution=0.25,
    )
    name = "nav_msgs/msg/OccupancyGrid"
    path = tmp_path / "map.mcap"
    with CdrMcapWriter(path) as writer:
        writer.write(
            "/map",
            cdr_encode(message),
            schema_name=name,
            schema=Definitions([], installed=True).schema(name),
            log_time_ns=1700000000123456790,
            publish_time_ns=1700000000123456789,
        )

    with path.open("rb") as source:
        records = list(make_reader(source).iter_messages())

    assert len(records) == 1
    schema, channel, record = records[0]
    assert channel.message_encoding == "cdr"
    assert schema.encoding == "ros2msg"
    assert schema.name == name
    assert "MSG: geometry_msgs/Pose" in schema.data.decode()
    assert "MSG: builtin_interfaces/Time" in schema.data.decode()
    assert record.publish_time == 1700000000123456789
    decoded = cdr_decode(record.data, OccupancyGrid)
    assert decoded.header == message.header
    np.testing.assert_array_equal(occupancy_view(decoded), [[-1, 0, 100]])
