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

"""One explicitly connected, process-local bus for the Desktop experiment."""

from collections.abc import Iterator
from contextlib import contextmanager
import socket
from uuid import uuid4

from dimos.core.global_config import global_config
from dimos.protocol.service.zenohservice import (
    LOOPBACK_LISTEN,
    ZenohConfig,
    ZenohService,
    ZenohSessionPool,
    default_session_pool,
)


@contextmanager
def learning_bus() -> Iterator[None]:
    """Link host, native recorder and policy without network discovery."""
    previous = global_config.model_dump()
    pool = ZenohSessionPool()
    client_config: ZenohConfig | None = None
    host = LOOPBACK_LISTEN.removeprefix("tcp/").rsplit(":", 1)[0]
    with socket.socket() as listener:
        listener.bind((host, 0))
        endpoint = f"tcp/{host}:{listener.getsockname()[1]}"
    router = ZenohService(
        session_pool=pool,
        mode="router",
        listen=[endpoint],
        connect=[],
        multicast=False,
        gossip=False,
        scouting=False,
    )
    try:
        router.start()
        global_config.update(
            transport="zenoh",
            robot_ip=None,
            robot_ips=None,
            zenoh_mode="client",
            zenoh_connect=endpoint,
            zenoh_scouting=False,
            zenoh_interface="",
            zenoh_multicast=False,
            zenoh_gossip=False,
            zenoh_domain=f"yam-learning-{uuid4().hex}",
            listen_host="127.0.0.1",
        )
        client_config = ZenohConfig()
        yield
    finally:
        if client_config is not None:
            default_session_pool.release(client_config)
        pool.close_all()
        global_config.update(**previous)
