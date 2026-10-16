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
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, JsonValue

from experimental.gateway import responses
from experimental.gateway.skills import SkillError
from experimental.gateway.state import ApiError, GatewayState

router = APIRouter()


class SkillCallRequest(BaseModel):
    skill: str
    args: dict[str, JsonValue] = {}
    module: str | None = None
    runId: str | None = None


@router.get("/skills", response_model=responses.Skills)
async def skill_list(state: GatewayState) -> dict[str, Any]:
    return await asyncio.to_thread(state.skills.listed)


@router.post("/skills/call", response_model=responses.SkillResult)
async def skill_call(state: GatewayState, request: SkillCallRequest) -> dict[str, Any]:
    """Call one of the running blueprint's skills, through its MCP server or over RPC."""
    try:
        return await asyncio.to_thread(
            state.skills.call, request.skill, request.args, request.module, request.runId
        )
    except SkillError as error:
        raise ApiError(error.status, str(error))
