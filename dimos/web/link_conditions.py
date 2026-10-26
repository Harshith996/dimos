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

"""Apply communication conditions at an ASGI boundary, without blocking its app."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
import threading
from typing import Any

from anyio import CancelScope
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dimos.protocol.testing.link import (
    DelayTimeline,
    Delivery,
    DeliveryQueue,
    Direction,
    LinkConditions,
    LinkOverflowError,
)


def message_size(message: Message) -> int:
    return (
        len(message.get("body", b""))
        + len(message.get("bytes") or b"")
        + len((message.get("text") or "").encode("utf-8"))
        + sum(len(key) + len(value) for key, value in message.get("headers", ()))
    )


@dataclass(eq=False)
class _LinkSession:
    reset: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)


class LinkRuntime:
    """One shared directional timeline and budget; sessions retain their own order.

    Scheduling runs on the server loop. Status is also readable from module RPC
    threads. Recent delivery records are bounded and contain timing, never data.
    """

    def __init__(self, conditions: LinkConditions) -> None:
        self.conditions = conditions
        self.loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._origin: float | None = None
        self._closed = False
        self._resetting = False
        self._timelines = {
            "to_robot": DelayTimeline(conditions, "to_robot"),
            "to_operator": DelayTimeline(conditions, "to_operator"),
        }
        self._sessions: set[_LinkSession] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=128)
        self._counts = dict(
            delivered=0, cancelled=0, overflows=0, pending_messages=0, pending_bytes=0
        )

    def register(self) -> _LinkSession | None:
        loop = asyncio.get_running_loop()
        session = _LinkSession()
        with self._lock:
            if self._closed or self._resetting:
                return None
            if self.loop is not None and self.loop is not loop:
                raise RuntimeError("a link runtime belongs to one server event loop")
            self.loop = loop
            if self._origin is None:
                self._origin = loop.time()
            self._sessions.add(session)
        return session

    def unregister(self, session: _LinkSession) -> None:
        with self._lock:
            self._sessions.discard(session)
            session.closed.set()

    def enqueue(
        self,
        queue: DeliveryQueue[Message],
        message: Message,
        direction: Direction,
        *,
        immediate: bool,
    ) -> None:
        assert self.loop is not None and self._origin is not None
        now, size = self.loop.time(), message_size(message)
        with self._lock:
            if (
                self._counts["pending_messages"] >= self.conditions.max_pending_messages
                or self._counts["pending_bytes"] + size > self.conditions.max_pending_bytes
            ):
                self._counts["overflows"] += 1
                raise LinkOverflowError("link delivery buffer limit exceeded")
            delay = 0.0 if immediate else self._timelines[direction].delay_at(now - self._origin)
            queue.push(dict(message), now=now, delay_s=delay, size=size)
            self._counts["pending_messages"] += 1
            self._counts["pending_bytes"] += size

    def delivered(self, delivery: Delivery[Message], direction: Direction, path: str) -> None:
        assert self.loop is not None and self._origin is not None
        now = self.loop.time()
        with self._lock:
            self._counts["pending_messages"] -= 1
            self._counts["pending_bytes"] -= delivery.size
            self._counts["delivered"] += 1
            self._recent.append(
                dict(
                    direction=direction,
                    path=path,
                    message_type=delivery.value["type"],
                    arrival_s=delivery.arrival - self._origin,
                    due_s=delivery.due - self._origin,
                    delivered_s=now - self._origin,
                    requested_delay_s=delivery.requested_delay_s,
                    ordering_wait_s=delivery.ordering_wait_s,
                    lateness_s=max(0.0, now - delivery.due),
                    bytes=delivery.size,
                )
            )

    def cancelled(self, messages: int, size: int) -> None:
        with self._lock:
            self._counts["pending_messages"] -= messages
            self._counts["pending_bytes"] -= size
            self._counts["cancelled"] += messages

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "conditions": self.conditions.model_dump(mode="json"),
                "elapsed_s": self.loop.time() - self._origin
                if self.loop is not None and self._origin is not None
                else 0.0,
                "active_connections": len(self._sessions),
                "closed": self._closed,
                "resetting": self._resetting,
                **self._counts,
                "recent_deliveries": list(self._recent),
            }

    async def reset(self) -> dict[str, Any]:
        """Disconnect current sessions before restarting the seeded timeline."""
        with self._lock:
            if self._closed or self._resetting:
                raise RuntimeError("link is closed or already resetting")
            self._resetting = True
            sessions = tuple(self._sessions)
        for session in sessions:
            session.reset.set()
        await asyncio.gather(*(session.closed.wait() for session in sessions))
        with self._lock:
            self._origin = asyncio.get_running_loop().time()
            self._timelines = {
                "to_robot": DelayTimeline(self.conditions, "to_robot"),
                "to_operator": DelayTimeline(self.conditions, "to_operator"),
            }
            self._recent.clear()
            self._resetting = False
        return self.status()

    def close(self) -> None:
        """Stop admitting clients and wake pending sessions before server shutdown."""
        with self._lock:
            self._closed = True
            sessions, loop = tuple(self._sessions), self.loop
        if loop is not None and loop.is_running():
            for session in sessions:
                loop.call_soon_threadsafe(session.reset.set)


class _DeliveryChannel:
    def __init__(self, runtime: LinkRuntime, direction: Direction, path: str) -> None:
        self.runtime, self.direction, self.path = runtime, direction, path
        self.queue: DeliveryQueue[Message] = DeliveryQueue(
            max_messages=runtime.conditions.max_pending_messages,
            max_bytes=runtime.conditions.max_pending_bytes,
        )
        self.changed = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
        self._inflight: Delivery[Message] | None = None

    def put(self, message: Message, *, immediate: bool = False) -> None:
        self.runtime.enqueue(self.queue, message, self.direction, immediate=immediate)
        self.idle.clear()
        self.changed.set()

    async def take(self) -> Delivery[Message]:
        loop = asyncio.get_running_loop()
        while True:
            self.changed.clear()
            delivery = self.queue.pop_ready(loop.time())
            if delivery is not None:
                self._inflight = delivery
                return delivery
            timeout = (
                None if self.queue.next_due is None else max(0, self.queue.next_due - loop.time())
            )
            with suppress(TimeoutError):
                await asyncio.wait_for(self.changed.wait(), timeout)

    def acknowledge(self) -> None:
        assert self._inflight is not None
        self.runtime.delivered(self._inflight, self.direction, self.path)
        self._inflight = None
        if self.queue.next_due is None:
            self.idle.set()

    async def receive(self) -> Message:
        delivery = await self.take()
        self.acknowledge()
        return delivery.value

    async def send(self, message: Message) -> None:
        self.put(message, immediate=message["type"] == "websocket.accept")

    def cancel(self) -> None:
        count, size = self.queue.clear()
        if self._inflight is not None:
            count += 1
            size += self._inflight.size
            self._inflight = None
        self.runtime.cancelled(count, size)
        self.idle.set()


class LinkConditionsMiddleware:
    """Delay HTTP delivery and WebSocket data; disconnect always cancels old work.

    Connection setup is unshaped for WebSockets. HTTP scope arrival is shaped as
    well as request bodies, since GET handlers need not call receive(). Outgoing
    bytes are captured before delay, including streaming response chunks.
    """

    def __init__(self, app: ASGIApp, runtime: LinkRuntime) -> None:
        self.app, self.runtime = app, runtime

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or scope["path"] == "/link-conditions":
            await self.app(scope, receive, send)
            return
        session = self.runtime.register()
        if session is None:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013, "reason": "Link unavailable"})
            else:
                await send({"type": "http.response.start", "status": 503, "headers": []})
                await send({"type": "http.response.body", "body": b"Link unavailable"})
            return
        incoming = _DeliveryChannel(self.runtime, "to_robot", scope["path"])
        outgoing = _DeliveryChannel(self.runtime, "to_operator", scope["path"])
        http = scope["type"] == "http"
        response_started = False

        async def read() -> None:
            if http:
                incoming.put({"type": "link.request"})
            while True:
                message = await receive()
                if message["type"] in ("http.disconnect", "websocket.disconnect"):
                    return
                incoming.put(message, immediate=message["type"] == "websocket.connect")

        async def write() -> None:
            nonlocal response_started
            while True:
                delivery = await outgoing.take()
                await send(delivery.value)
                if delivery.value["type"] in ("http.response.start", "websocket.accept"):
                    response_started = True
                outgoing.acknowledge()

        async def dispatch() -> None:
            if http:
                await incoming.receive()  # Scope/header arrival precedes request body delivery.
            await self.app(scope, incoming.receive, outgoing.send)
            await outgoing.idle.wait()

        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(run()) for run in (read, write, dispatch)
        ]
        resetting = asyncio.create_task(session.reset.wait())
        tasks.append(resetting)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if resetting in done and not http:
                await send({"type": "websocket.close", "code": 1012, "reason": "Link reset"})
        except LinkOverflowError:
            if not http:
                await send({"type": "websocket.close", "code": 1013, "reason": "Link buffer full"})
            elif not response_started:
                await send({"type": "http.response.start", "status": 503, "headers": []})
                await send({"type": "http.response.body", "body": b"Link buffer full"})
            else:
                raise
        finally:
            for task in tasks:
                task.cancel()
            # Complete application finally blocks even when an ASGI parent's
            # cancellation scope repeatedly cancels awaits during shutdown.
            with CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)
                incoming.cancel()
                outgoing.cancel()
                self.runtime.unregister(session)
