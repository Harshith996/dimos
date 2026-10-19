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
import socket
import threading

import zenoh

from dimos.evals.sandbox.robot import await_key, router_config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_await_key_sees_a_key_publish_and_times_out_on_one_that_doesnt() -> None:
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    assert not await_key(None, endpoint, 1.0)  # no router yet
    with zenoh.open(router_config([endpoint], [])):
        assert await_key(None, endpoint, 5.0)
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        with zenoh.open(config) as publisher:
            stop = threading.Event()

            def publish() -> None:
                while not stop.wait(0.1):
                    publisher.put("robot/camera/jpeg", b"frame")

            thread = threading.Thread(target=publish, daemon=True)
            thread.start()
            try:
                assert await_key("robot/camera/**", endpoint, 5.0)
                assert not await_key("robot/lidar/**", endpoint, 1.0)
            finally:
                stop.set()
                thread.join()
