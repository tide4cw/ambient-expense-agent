# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and integration tests for the ambient expense agent workflow."""

from typing import Any

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import (
    app as adk_app,
    auto_approve,
    route_by_amount,
)


def test_route_by_amount_threshold() -> None:
    class DummyContext:
        def __init__(self) -> None:
            self.state: dict[str, Any] = {}

    ctx = DummyContext()
    event_low = route_by_amount({"amount": 99.99}, ctx)
    assert event_low.actions.route == "AUTO_APPROVE"

    event_high = route_by_amount({"amount": 100.00}, ctx)
    assert event_high.actions.route == "NEEDS_REVIEW"

    event_higher = route_by_amount({"amount": 500.00}, ctx)
    assert event_higher.actions.route == "NEEDS_REVIEW"


def test_auto_approve_node() -> None:
    result = auto_approve({"amount": 50.00, "submitter": "alice@example.com", "category": "meals"})
    assert result.output["status"] == "approved"
    assert result.output["amount"] == 50.00


@pytest.mark.asyncio
async def test_workflow_auto_approve_flow() -> None:
    """Expenses < $100 run to completion without any RequestInput pause."""
    session_service = InMemorySessionService()
    runner = Runner(app=adk_app, session_service=session_service)
    session = await session_service.create_session(app_name="app", user_id="user_low")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Expense forty-five dollars for team lunch")],
    )

    events = []
    async for event in runner.run_async(user_id="user_low", session_id=session.id, new_message=message):
        events.append(event)

    # Verify no interrupt occurred
    interrupts = [
        part.function_call
        for e in events
        if e.content and e.content.parts
        for part in e.content.parts
        if getattr(part, "function_call", None) and part.function_call.name == "adk_request_input"
    ]
    assert len(interrupts) == 0, "Expense under $100 must not trigger RequestInput"

    outputs = [e.output for e in events if getattr(e, "output", None)]
    assert any(o.get("status") == "approved" for o in outputs)


@pytest.mark.asyncio
async def test_workflow_verbal_amount_five_hundred_bucks_triggers_hitl() -> None:
    """Natural language 'five hundred buck flight ticket' parses to 500 and pauses via RequestInput."""
    session_service = InMemorySessionService()
    runner = Runner(app=adk_app, session_service=session_service)
    session = await session_service.create_session(app_name="app", user_id="user_high")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Expense five hundred buck flight ticket for conference")],
    )

    events = []
    async for event in runner.run_async(user_id="user_high", session_id=session.id, new_message=message):
        events.append(event)

    # Verify RequestInput pause triggered
    interrupt_call = None
    for e in events:
        if e.content and e.content.parts:
            for part in e.content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == "adk_request_input":
                    interrupt_call = fc
                    break

    assert interrupt_call is not None, "Expense of five hundred bucks must trigger adk_request_input"
    interrupt_id = interrupt_call.id
    assert interrupt_call.args["payload"]["amount"] == 500.00
    assert interrupt_call.args["payload"]["category"] == "travel"

    # Resume with approval
    resume_msg = types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            name="adk_request_input",
            response={"result": '{"decision": "approve"}'},
            id=interrupt_id,
        ))],
    )

    resume_events = []
    async for event in runner.run_async(user_id="user_high", session_id=session.id, new_message=resume_msg):
        resume_events.append(event)

    outputs = [e.output for e in resume_events if getattr(e, "output", None)]
    assert any(o.get("status") == "approved" for o in outputs)
