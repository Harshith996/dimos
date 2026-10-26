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

import asyncio
import json

import httpx
import pytest

from dimos.protocol.testing.link import Delay, LinkConditions, Stall
from dimos.web.link_conditions import LinkConditionsMiddleware, LinkRuntime
from dimos.web.robot_web_interface import RobotWebInterface


@pytest.fixture
async def connections():
    tasks = []

    def start(app, conditions, *, kind="websocket", runtime=None):
        source, output = asyncio.Queue(), asyncio.Queue()
        runtime = runtime or LinkRuntime(conditions)

        async def receive():
            value = await source.get()
            source.task_done()
            return value

        middleware = LinkConditionsMiddleware(app, runtime)
        task = asyncio.create_task(middleware({"type": kind, "path": "/test"}, receive, output.put))
        tasks.append(task)
        return runtime, source, output, task

    yield start
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def next_message(queue):
    return await asyncio.wait_for(queue.get(), timeout=3)


async def echo(scope, receive, send):
    assert (await receive())["type"] == "websocket.connect"
    await send({"type": "websocket.accept"})
    while True:
        message = await receive()
        await send({"type": "websocket.send", "text": message["text"]})


async def test_http_delays_request_scope_and_preserves_captured_response(connections):
    captured = []

    async def app(scope, receive, send):
        captured.append(asyncio.get_running_loop().time())
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": json.dumps(captured[0]).encode()})

    runtime, source, output, task = connections(
        app,
        LinkConditions(to_robot=Delay(base_s=0.03), to_operator=Delay(base_s=0.04)),
        kind="http",
    )
    before = asyncio.get_running_loop().time()
    await source.put({"type": "http.request", "body": b""})
    assert (await next_message(output))["status"] == 200
    body = await next_message(output)
    received = asyncio.get_running_loop().time()
    await asyncio.wait_for(task, 3)
    assert json.loads(body["body"]) == captured[0]
    assert captured[0] - before >= 0.03
    assert received - captured[0] >= 0.04
    assert runtime.status()["pending_messages"] == 0


async def test_streaming_bytes_are_delivered_in_order_without_blocking_producer(connections):
    produced = asyncio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        for body in (b"frame1", b"frame2", b"frame3"):
            await send({"type": "http.response.body", "body": body, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        produced.set()

    runtime, source, output, task = connections(
        app, LinkConditions(to_operator=Delay(base_s=0.1)), kind="http"
    )
    await source.put({"type": "http.request", "body": b""})
    await asyncio.wait_for(produced.wait(), 3)
    assert output.empty()
    assert runtime.status()["pending_bytes"] == 18
    assert (await next_message(output))["status"] == 200
    assert [(await next_message(output))["body"] for _ in range(4)] == [
        b"frame1",
        b"frame2",
        b"frame3",
        b"",
    ]
    await asyncio.wait_for(task, 3)


async def test_disconnect_cancels_delayed_grab_and_reconnect_has_no_old_messages(connections):
    conditions = LinkConditions(to_robot=Delay(base_s=0.05))
    runtime, source, output, task = connections(echo, conditions)
    await source.put({"type": "websocket.connect"})
    assert (await next_message(output))["type"] == "websocket.accept"
    await source.put({"type": "websocket.receive", "text": "old grab"})
    await source.join()
    await source.put({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(task, 3)
    assert output.empty()
    assert runtime.status()["cancelled"] == 1
    assert runtime.status()["pending_bytes"] == 0

    _, source2, output2, _ = connections(echo, conditions, runtime=runtime)
    await source2.put({"type": "websocket.connect"})
    assert (await next_message(output2))["type"] == "websocket.accept"
    await source2.put({"type": "websocket.receive", "text": "new release"})
    assert (await next_message(output2))["text"] == "new release"


async def test_reset_closes_control_session_and_clears_stalled_commands(connections):
    runtime, source, output, task = connections(
        echo, LinkConditions(stalls=(Stall(at_s=0, duration_s=10),))
    )
    await source.put({"type": "websocket.connect"})
    assert (await next_message(output))["type"] == "websocket.accept"
    await source.put({"type": "websocket.receive", "text": "grab"})
    await source.join()
    assert runtime.status()["pending_messages"] == 1
    status = await runtime.reset()
    assert (await next_message(output))["code"] == 1012
    await asyncio.wait_for(task, 3)
    assert not task.cancelled()
    assert (
        status["active_connections"] == status["pending_messages"] == status["pending_bytes"] == 0
    )


async def test_overflow_closes_connection_instead_of_dropping_a_release(connections):
    runtime, source, output, task = connections(
        echo, LinkConditions(to_robot=Delay(base_s=1), max_pending_bytes=4)
    )
    await source.put({"type": "websocket.connect"})
    assert (await next_message(output))["type"] == "websocket.accept"
    await source.put({"type": "websocket.receive", "text": "grab"})
    await source.put({"type": "websocket.receive", "text": "release"})
    assert (await next_message(output))["code"] == 1013
    await asyncio.wait_for(task, 3)
    assert runtime.status()["overflows"] == 1
    assert runtime.status()["pending_messages"] == 0


async def test_shutdown_cancels_long_stall_and_rejects_new_sessions(connections):
    conditions = LinkConditions(stalls=(Stall(at_s=0, duration_s=3600),))
    runtime, source, output, task = connections(echo, conditions)
    await source.put({"type": "websocket.connect"})
    assert (await next_message(output))["type"] == "websocket.accept"
    await source.put({"type": "websocket.receive", "text": "grab"})
    await source.join()
    runtime.close()
    assert (await next_message(output))["code"] == 1012
    await asyncio.wait_for(task, 3)
    assert runtime.status()["pending_messages"] == 0
    _, _, rejected, rejected_task = connections(echo, conditions, runtime=runtime)
    assert (await next_message(rejected))["code"] == 1013
    await asyncio.wait_for(rejected_task, 3)


async def test_robot_web_interface_installs_adapter_and_exposes_unshaped_diagnostics():
    server = RobotWebInterface(link_conditions=LinkConditions(to_robot=Delay(base_s=0.02)))

    @server.app.get("/observation")
    def observation():
        return {"ts": 123, "opening": 0.5}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(server.app), base_url="http://test"
    ) as client:
        result = await client.get("/observation")
        assert result.json() == {"ts": 123, "opening": 0.5}
        status = (await client.get("/link-conditions")).json()
    assert status["conditions"]["to_robot"]["base_s"] == 0.02
    incoming = [r for r in status["recent_deliveries"] if r["message_type"] == "link.request"]
    assert incoming[0]["requested_delay_s"] == pytest.approx(0.02)
    assert status["active_connections"] == 0
    assert {r["path"] for r in status["recent_deliveries"]} == {"/observation"}
