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

"""A finite built-in EpisodeStatus exchange over typed Python workers and Zenoh."""

from contextlib import ExitStack
from functools import partial
import math
from pathlib import Path
import socket
import threading
from typing import Literal
from unittest.mock import patch
import uuid

from dimos_generated.dimos_msgs.msg import EpisodeStatus

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.protocol.cdr_mcap import CdrMcapWriter
from dimos.protocol.service.system_configurator.base import configure_system
from dimos.protocol.service.zenohservice import ZenohConfig, ZenohSessionPool


def validate_status(status: EpisodeStatus) -> None:
    """Apply the source model's finite timestamp and Literal-value constraints."""
    if not math.isfinite(status.ts):
        raise ValueError("EpisodeStatus.ts must be finite")
    if status.state not in ("idle", "recording"):
        raise ValueError("EpisodeStatus.state must be idle or recording")
    if status.last_event not in ("start", "save", "discard", "init"):
        raise ValueError("EpisodeStatus.last_event must be start, save, discard or init")
    status.validate()


def episode_status(
    *,
    ts: float,
    state: Literal["idle", "recording"],
    episodes_saved: int,
    episodes_discarded: int,
    last_event: Literal["start", "save", "discard", "init"] = "init",
    task_label: str | None = None,
) -> EpisodeStatus:
    """Construct the wire value, preserving required fields and nullable labels."""
    status = EpisodeStatus(
        ts=ts,
        state=state,
        episodes_saved=episodes_saved,
        episodes_discarded=episodes_discarded,
        last_event=last_event,
        task_label=[] if task_label is None else [task_label],
    )
    validate_status(status)
    return status


class EpisodeExchange(Module):
    status: Out[EpisodeStatus]
    observed: In[EpisodeStatus]

    @rpc
    def start(self) -> None:
        self.received: EpisodeStatus | None = None
        self.done = threading.Event()
        super().start()

    async def handle_observed(self, status: EpisodeStatus) -> None:
        self.received = status
        self.done.set()

    @rpc
    def exchange(self) -> EpisodeStatus:
        status = episode_status(
            ts=17.25,
            state="recording",
            episodes_saved=2,
            episodes_discarded=1,
            last_event="start",
            task_label="",
        )
        for _ in range(60):
            self.status.publish(status)
            if self.done.wait(0.25):
                break
        if self.received is None:
            raise TimeoutError("No typed EpisodeStatus reply")
        if self.received.encode() != status.encode():
            raise ValueError("EpisodeStatus fields changed across the typed transport")
        return self.received


class EpisodeInspector(Module):
    status: In[EpisodeStatus]
    observed: Out[EpisodeStatus]

    async def handle_status(self, status: EpisodeStatus) -> None:
        validate_status(status)
        self.observed.publish(status)


def record_status(path: Path, status: EpisodeStatus) -> None:
    validate_status(status)
    source_ns = round(status.ts * 1_000_000_000)
    with CdrMcapWriter(path) as writer:
        writer.write(
            "/episode/status",
            status.encode(),
            schema_name=EpisodeStatus.msg_name,
            schema=EpisodeStatus.schema,
            log_time_ns=source_ns + 1,
            publish_time_ns=source_ns,
        )


def main() -> None:
    with ExitStack() as stack:
        # Report tuning needs without applying host/system/network configuration.
        stack.enter_context(
            patch(
                "dimos.protocol.service.system_configurator.base.configure_system",
                partial(configure_system, check_only=True),
            )
        )
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{reservation.getsockname()[1]}"
        pool = ZenohSessionPool()
        stack.callback(pool.close_all)
        pool.acquire(
            ZenohConfig(mode="router", listen=[endpoint], connect=[], multicast=False, gossip=False)
        )
        blueprint = autoconnect(EpisodeExchange.blueprint(), EpisodeInspector.blueprint())
        configured = blueprint.namespace("episode" + uuid.uuid4().hex[:8]).global_config(
            viewer="none",
            n_workers=2,
            transport="zenoh",
            zenoh_mode="client",
            zenoh_connect=endpoint,
            zenoh_multicast=False,
        )
        coordinator = ModuleCoordinator.build(configured)
        stack.callback(coordinator.stop)
        status = coordinator.get_instance(EpisodeExchange).exchange()
        path = Path("build/episode-status/status.mcap")
        path.parent.mkdir(parents=True, exist_ok=True)
        record_status(path, status)
        print(f"PASS: typed EpisodeStatus state={status.state}, empty label preserved; {path}")


if __name__ == "__main__":
    main()
