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

"""Ambient expense agent that streamlines employee expense reporting.

Leverages an LLM parse_agent to intelligently understand and extract expense
data from natural language or structured input, instantly approves claims under
$100 via auto_approve, and routes claims of $100 or more to review_agent,
triggering a human-in-the-loop pause (RequestInput) before final decision.
"""

import datetime
import json
import logging
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events import Event, RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import Client, types
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
REVIEW_THRESHOLD = 100.0


# ---------------------------------------------------------------------------
# Structured schema for expense data flow between nodes
# ---------------------------------------------------------------------------


class ExpenseData(BaseModel):
    """Structured expense data extracted by parse_agent."""

    amount: float = Field(
        description="Total expense amount in USD as a floating point number. Convert verbal amounts (e.g. 'five hundred bucks') into numbers (500.0).",
    )
    submitter: str = Field(
        default="employee@example.com",
        description="Email address of the submitter if mentioned, otherwise employee@example.com.",
    )
    category: str = Field(
        default="general",
        description="Expense category (e.g., travel, meals, supplies, equipment, software, lodging).",
    )
    description: str = Field(
        default="expense claim",
        description="Clear concise description of what the expense was for.",
    )
    date: str = Field(
        default_factory=lambda: datetime.date.today().isoformat(),
        description="Date of expense in YYYY-MM-DD format.",
    )


# ---------------------------------------------------------------------------
# 1. Parsing node: Intelligent LLM agent to parse natural language or JSON
# ---------------------------------------------------------------------------

parse_agent = Agent(
    name="parse_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are an intelligent expense extraction agent.
Analyze the user's input (which may be informal natural language, verbal amounts like 'five hundred buck flight ticket', or structured JSON).
Extract the expense details accurately and populate the output schema:
- Convert verbal numbers (e.g. 'five hundred bucks' -> 500.0, 'forty five dollars' -> 45.0) into float amounts.
- Infer the appropriate category (travel, meals, supplies, equipment, etc.).
- Extract the submitter email if provided, otherwise default to employee@example.com.
- Summarize the expense description cleanly.""",
    output_schema=ExpenseData,
)


# ---------------------------------------------------------------------------
# 2. Routing node: Deterministic business logic based on $100 threshold
# ---------------------------------------------------------------------------


def route_by_amount(node_input: dict, ctx: Context) -> Event:
    """Route expenses based on the $100 review threshold."""
    ctx.state["expense_data"] = node_input
    amount = float(node_input.get("amount", 0.0))
    if amount >= REVIEW_THRESHOLD:
        return Event(route="NEEDS_REVIEW", output=node_input)
    return Event(route="AUTO_APPROVE", output=node_input)


# ---------------------------------------------------------------------------
# 3. Auto-approval node: Instantly approves claims under $100
# ---------------------------------------------------------------------------


def auto_approve(node_input: dict) -> Event:
    """Auto-approve expenses under $100 and log the confirmation."""
    amount = float(node_input.get("amount", 0.0))
    submitter = node_input.get("submitter", "employee@example.com")
    category = node_input.get("category", "general")
    description = node_input.get("description", "standard claim")

    log_entry = {
        "severity": "INFO",
        "message": f"Expense auto-approved: ${amount:.2f} from {submitter}",
        "decision": "approved",
        "amount": amount,
        "submitter": submitter,
        "category": category,
    }
    print(json.dumps(log_entry), flush=True)

    summary_text = (
        f"✅ Expense auto-approved: ${amount:.2f} for '{description}' "
        f"({category}) from {submitter}. Decision: approved."
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=summary_text)],
        ),
        output={"status": "approved", **node_input},
    )


# ---------------------------------------------------------------------------
# 4. Review node: Flags expenses >= $100 and pauses via RequestInput (HITL)
# ---------------------------------------------------------------------------


def emit_expense_alert(
    submitter: str,
    amount: float,
    category: str,
    risk_summary: str,
) -> dict:
    """Emit a structured warning log alerting to high-value expense review."""
    log_entry = {
        "severity": "WARNING",
        "alert_type": "expense_review",
        "message": f"Expense review alert: ${amount:.2f} from {submitter} — {risk_summary}",
        "submitter": submitter,
        "amount": amount,
        "category": category,
        "risk_summary": risk_summary,
    }
    print(json.dumps(log_entry), flush=True)
    return {"status": "alert_emitted", "submitter": submitter, "amount": amount}


async def review_agent(node_input: dict, ctx: Context):
    """Review expense of $100 or more, evaluate risks, and trigger RequestInput HITL pause."""
    amount = float(node_input.get("amount", 0.0))
    submitter = node_input.get("submitter", "employee@example.com")
    category = node_input.get("category", "general")
    description = node_input.get("description", "high-value claim")

    # Evaluate risk factors via Gemini model
    risk_summary = f"Claim of ${amount:.2f} exceeds standard $100 approval limit."
    try:
        client = Client()
        prompt = (
            f"Provide a 1-sentence risk summary for an expense claim of ${amount:.2f} "
            f"in category '{category}' submitted by {submitter}: '{description}'."
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        if response and response.text:
            risk_summary = response.text.strip()
    except Exception as e:
        logger.warning(f"Could not reach LLM for review analysis: {e}")

    # Emit structured review alert
    emit_expense_alert(submitter, amount, category, risk_summary)

    alert_message = (
        f"⚠️ Expense of ${amount:.2f} from {submitter} requires human review.\n"
        f"Category: {category}\n"
        f"Description: {description}\n"
        f"Risk Summary: {risk_summary}\n\n"
        f"Paused for manager review (RequestInput triggered)."
    )
    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=alert_message)],
        )
    )

    # Pause execution awaiting human manager input
    yield RequestInput(
        message=f"Expense of ${amount:.2f} from {submitter} requires manager approval. Please approve or reject.",
        payload={
            "amount": amount,
            "submitter": submitter,
            "category": category,
            "description": description,
            "risk_summary": risk_summary,
        },
    )


# ---------------------------------------------------------------------------
# 5. Decision node: Resumes upon human approval or rejection
# ---------------------------------------------------------------------------


def process_decision(node_input: Any, ctx: Context) -> Event:
    """Process human approval decision when the workflow is resumed."""
    decision = "unknown"
    if isinstance(node_input, dict):
        decision = node_input.get("decision", "unknown")
    elif isinstance(node_input, str):
        decision = "approve" if "approve" in node_input.lower() else "reject"

    approved = decision.lower() == "approve"
    status = "approved" if approved else "rejected"
    expense = ctx.state.get("expense_data", {})
    amount = float(expense.get("amount", 0.0))
    submitter = expense.get("submitter", "employee@example.com")

    log_entry = {
        "severity": "INFO" if approved else "WARNING",
        "message": f"Expense {status} by manager",
        "decision": status,
        "amount": amount,
        "submitter": submitter,
    }
    print(json.dumps(log_entry), flush=True)

    summary = (
        f"{'✅' if approved else '❌'} Expense of ${amount:.2f} from {submitter} "
        f"has been {status} by manager."
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=summary)],
        ),
        output={"status": status, "decision": decision, **expense},
    )


# ---------------------------------------------------------------------------
# Root Workflow Graph
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="ambient_expense_agent",
    edges=[
        ("START", parse_agent),
        (parse_agent, route_by_amount),
        (
            route_by_amount,
            {
                "AUTO_APPROVE": auto_approve,
                "NEEDS_REVIEW": review_agent,
            },
        ),
        (review_agent, process_decision),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
