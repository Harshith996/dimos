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

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Path as PathParam

from experimental.gateway import logs, responses, store
from experimental.gateway.state import ApiError, GatewayState, State

router = APIRouter(prefix="/history")

RunId = Annotated[str, PathParam(alias="runId")]


async def read_log(
    state: State,
    run_id: str,
    after: int | None,
    level: str | None,
    q: str | None,
    limit: int | None,
) -> dict[str, Any]:
    file = await asyncio.to_thread(logs.find, state.dimos_dir, run_id)
    if file is None:
        raise ApiError(
            404, "no run has a log yet" if run_id == "latest" else f"no log for run {run_id}"
        )
    return await asyncio.to_thread(
        logs.read, file, run_id, after, limit or 1000, q or None, level or None
    )


@router.get("", response_model=responses.History)
async def history(state: GatewayState) -> dict[str, Any]:
    """Every run with a log, newest first, whether or not it still runs."""
    return {"runs": await asyncio.to_thread(logs.history, state.dimos_dir)}


@router.get("/recordings", response_model=responses.Recordings)
async def recordings() -> dict[str, Any]:
    """The standard directory dimos writes recordings (mcap and db files) to."""
    return {"path": str(store.recordings_dir())}


@router.get("/{runId}", response_model=responses.PastRunDetail)
async def past_run(state: GatewayState, run_id: RunId) -> dict[str, Any]:
    """One run's entry and its log directory; `latest` is the newest."""
    runs = await asyncio.to_thread(logs.history, state.dimos_dir)
    found = (
        runs[0]
        if run_id == "latest" and runs
        else next((r for r in runs if r["runId"] == run_id), None)
    )
    file = await asyncio.to_thread(logs.find, state.dimos_dir, found["runId"]) if found else None
    if found is None or file is None:
        raise ApiError(404, f"no run {run_id}")
    return {**found, "logDir": str(file.parent)}


@router.get("/{runId}/logs", response_model=responses.RunLog)
async def past_log(
    state: GatewayState,
    run_id: RunId,
    after: int | None = None,
    level: str | None = None,
    q: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """A run's log (`latest`: the newest run's)."""
    return await read_log(state, run_id, after, level, q, limit)
