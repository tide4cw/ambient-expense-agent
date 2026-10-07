"""Manager Dashboard Service for Ambient Expense Agent.

A standalone FastAPI service providing Human-in-the-Loop review for paused
Agent Runtime sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import google.auth
from google.auth.transport.requests import Request
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from google.adk.sessions import VertexAiSessionService
from pydantic import BaseModel
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manager_dashboard")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCP_PROJECT = (
    os.getenv("GCP_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or "joshslee-l200-sandbox-323214"
)
AGENT_RUNTIME_ID = (
    os.getenv("AGENT_RUNTIME_ID")
    or os.getenv("REASONING_ENGINE_ID")
    or "projects/1094633924648/locations/us-east1/reasoningEngines/5119476772018061312"
)
LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("LOCATION")
    or "us-east1"
)

# Extract short engine ID if full resource name was provided
def extract_engine_id(resource_or_id: str) -> str:
    return resource_or_id.strip("/").split("/")[-1]

SHORT_ENGINE_ID = extract_engine_id(AGENT_RUNTIME_ID)

# Ensure full resource name format
if not AGENT_RUNTIME_ID.startswith("projects/"):
    FULL_RESOURCE_NAME = f"projects/{GCP_PROJECT}/locations/{LOCATION}/reasoningEngines/{SHORT_ENGINE_ID}"
else:
    FULL_RESOURCE_NAME = AGENT_RUNTIME_ID
    # Parse location from resource name if available
    loc_match = re.search(r"/locations/([a-zA-Z0-9-_]+)/", AGENT_RUNTIME_ID)
    if loc_match:
        LOCATION = loc_match.group(1)

# Initialize Session Service
session_service = VertexAiSessionService(
    project=GCP_PROJECT,
    location=LOCATION,
    agent_engine_id=SHORT_ENGINE_ID,
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ambient Expense Governance Dashboard",
    description="Human-in-the-Loop Management Dashboard for Agent Runtime",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ActionRequest(BaseModel):
    interrupt_id: str
    approved: bool


# ---------------------------------------------------------------------------
# Helpers for Agent Runtime Stream Execution
# ---------------------------------------------------------------------------
def _get_auth_token() -> str:
    """Retrieve ADC OAuth2 token."""
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds.token


def call_agent_runtime_stream(
    session_id: str,
    resume_payload: dict[str, Any],
    user_id: str = "default-user",
) -> list[dict[str, Any]]:
    """Resumes the session on Agent Runtime via HTTP :streamQuery."""
    token = _get_auth_token()
    stream_url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/{FULL_RESOURCE_NAME}:streamQuery"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "class_method": "async_stream_query",
        "input": {
            "message": resume_payload,
            "user_id": user_id,
            "session_id": session_id,
        },
    }

    logger.info("Calling Agent Runtime streamQuery for session %s (user: %s)", session_id, user_id)
    resp = requests.post(stream_url, headers=headers, json=payload, stream=True, timeout=120)

    if not resp.ok:
        logger.error("Agent Runtime returned %d: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Agent Runtime execution failed: {resp.text}",
        )

    events = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            event_dict = json.loads(line)
            events.append(event_dict)
        except json.JSONDecodeError:
            continue

    return events


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/pending")
async def get_pending_approvals():
    """Queries VertexAiSessionService to list all sessions, fetches full

    history for each session, and identifies unresolved adk_request_input
    function calls.
    """
    try:
        resp = await session_service.list_sessions(app_name=SHORT_ENGINE_ID)
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query sessions from Vertex AI: {e}",
        )

    pending_items = []

    for s in resp.sessions:
        try:
            # Pass session's actual owner to avoid ownership mismatch when reading history
            session = await session_service.get_session(
                app_name=SHORT_ENGINE_ID,
                user_id=s.user_id,
                session_id=s.id,
            )
        except Exception as e:
            logger.warning("Could not read history for session %s: %s", s.id, e)
            continue

        if not session or not session.events:
            continue

        calls = {}
        responses = set()

        for ev in session.events:
            if not ev.content or not ev.content.parts:
                continue
            for part in ev.content.parts:
                # 1. Look for adk_request_input function call
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    if fc.name == "adk_request_input":
                        args = fc.args if isinstance(fc.args, dict) else {}
                        call_id = fc.id or args.get("interruptId")
                        payload = args.get("payload", {})
                        message = args.get("message", "")
                        calls[call_id] = {
                            "session_id": s.id,
                            "session_user": s.user_id,
                            "interrupt_id": call_id,
                            "payload": payload,
                            "message": message,
                            "timestamp": ev.timestamp,
                        }

                # 2. Look for adk_request_input function response
                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    if fr.name == "adk_request_input":
                        responses.add(fr.id)

        # Unresolved are calls without matching response
        for call_id, info in calls.items():
            if call_id and call_id not in responses:
                pending_items.append(info)

    # Sort descending by timestamp
    pending_items.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
    return JSONResponse(content={"pending": pending_items, "count": len(pending_items)})


@app.post("/api/action/{session_id}")
async def handle_manager_action(session_id: str, body: ActionRequest):
    """Resumes the paused session on Agent Runtime.

    Passes resume payload directly as dict value of `message` argument
    with user_id strictly set to 'default-user'.
    """
    # Build resume payload
    resume_payload = {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": body.interrupt_id,
                    "name": "adk_request_input",
                    "response": {
                        "approved": body.approved,
                        "decision": "approve" if body.approved else "reject",
                    },
                }
            }
        ],
    }

    # Strict requirement: set user_id to "default-user"
    user_id = "default-user"

    try:
        events = call_agent_runtime_stream(
            session_id=session_id,
            resume_payload=resume_payload,
            user_id=user_id,
        )

        # Fallback if session ownership check in Agent Runtime requires original session owner
        if not events:
            logger.info("Empty events returned with default-user. Checking session owner...")
            try:
                session_resp = await session_service.list_sessions(app_name=SHORT_ENGINE_ID)
                matching = next((s for s in session_resp.sessions if s.id == session_id), None)
                if matching and matching.user_id and matching.user_id != "default-user":
                    logger.info("Retrying with original session user %s", matching.user_id)
                    events = call_agent_runtime_stream(
                        session_id=session_id,
                        resume_payload=resume_payload,
                        user_id=matching.user_id,
                    )
            except Exception as e:
                logger.warning("Session owner lookup failed: %s", e)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume session %s: %s", session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume session on Agent Runtime: {e}",
        )

    # Extract final compliance review text and decision from events
    review_text = ""
    decision_status = "approved" if body.approved else "rejected"

    for ev in events:
        content = ev.get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            if "text" in p and p["text"]:
                review_text = p["text"]
        output = ev.get("output") or {}
        if "status" in output:
            decision_status = output["status"]

    if not review_text:
        review_text = (
            f"✅ Expense has been approved by manager."
            if body.approved
            else f"❌ Expense has been rejected by manager."
        )

    return JSONResponse(
        content={
            "status": "success",
            "session_id": session_id,
            "interrupt_id": body.interrupt_id,
            "decision": decision_status,
            "review": review_text,
            "events_count": len(events),
        }
    )


# ---------------------------------------------------------------------------
# Dashboard UI HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Expense Governance • Manager Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #07090e;
      --card-bg: rgba(18, 22, 34, 0.65);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-hover-border: rgba(99, 102, 241, 0.4);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.35);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.3);
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.3);
      --amber: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(circle at 10% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 85% 20%, rgba(16, 185, 129, 0.09) 0%, transparent 40%),
        radial-gradient(circle at 50% 85%, rgba(244, 63, 94, 0.08) 0%, transparent 50%);
      background-attachment: fixed;
      color: var(--text);
      font-family: var(--font-body);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }

    /* Ambient glass container */
    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem;
    }

    /* Header */
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1.5rem;
      margin-bottom: 2.5rem;
      padding-bottom: 1.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .brand-icon {
      width: 48px;
      height: 48px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 24px var(--primary-glow);
    }

    .brand-icon svg {
      width: 26px;
      height: 26px;
      color: #fff;
    }

    .brand-text h1 {
      font-family: var(--font-display);
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-text p {
      font-size: 0.85rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .stats-pill {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      padding: 0.5rem 1rem;
      border-radius: 12px;
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .stats-count {
      font-family: var(--font-display);
      font-weight: 700;
      color: var(--primary);
      font-size: 1.1rem;
    }

    .refresh-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 0.5rem 0.9rem;
      border-radius: 12px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      backdrop-filter: blur(12px);
    }

    .refresh-btn:hover {
      background: rgba(255, 255, 255, 0.09);
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }

    .refresh-btn svg.spin {
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      100% { transform: rotate(360deg); }
    }

    /* Grid layout */
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.75rem;
    }

    /* Glassmorphic card */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 1.5rem;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--primary), transparent);
      opacity: 0;
      transition: opacity 0.3s;
    }

    .card:hover {
      border-color: var(--card-hover-border);
      transform: translateY(-4px);
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
    }

    .card:hover::before {
      opacity: 1;
    }

    .card-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1.25rem;
    }

    .category-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.25);
      color: #a5b4fc;
    }

    .category-badge.meals {
      background: rgba(245, 158, 11, 0.12);
      border-color: rgba(245, 158, 11, 0.3);
      color: #fcd34d;
    }

    .category-badge.travel {
      background: rgba(14, 165, 233, 0.12);
      border-color: rgba(14, 165, 233, 0.3);
      color: #7dd3fc;
    }

    .amount-tag {
      font-family: var(--font-display);
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 600;
      color: #fff;
      margin-bottom: 0.35rem;
    }

    .submitter-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 1.25rem;
    }

    .submitter-row svg {
      width: 14px;
      height: 14px;
      color: #64748b;
    }

    /* Risk analysis callout */
    .risk-box {
      background: rgba(244, 63, 94, 0.06);
      border: 1px solid rgba(244, 63, 94, 0.2);
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 1.5rem;
    }

    .risk-header {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: #fb7185;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.4rem;
    }

    .risk-text {
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.45;
    }

    .meta-details {
      margin-top: auto;
      padding-top: 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: #64748b;
      margin-bottom: 1.25rem;
    }

    .session-chip {
      background: rgba(255, 255, 255, 0.04);
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      font-family: monospace;
      color: #94a3b8;
    }

    /* Action buttons */
    .card-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }

    .btn {
      padding: 0.75rem 1rem;
      border-radius: 12px;
      font-family: var(--font-body);
      font-size: 0.85rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
    }

    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn-approve {
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ffffff;
      box-shadow: 0 4px 15px var(--emerald-glow);
    }

    .btn-approve:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
    }

    .btn-reject {
      background: rgba(244, 63, 94, 0.12);
      border: 1px solid rgba(244, 63, 94, 0.35);
      color: #fb7185;
    }

    .btn-reject:hover:not(:disabled) {
      background: rgba(244, 63, 94, 0.22);
      border-color: rgba(244, 63, 94, 0.55);
      transform: translateY(-2px);
    }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 5rem 2rem;
      background: var(--card-bg);
      border: 1px dashed var(--card-border);
      border-radius: 24px;
      backdrop-filter: blur(16px);
      max-width: 600px;
      margin: 2rem auto;
    }

    .empty-icon {
      width: 64px;
      height: 64px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: 20px;
      margin: 0 auto 1.5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #10b981;
      box-shadow: 0 0 24px var(--emerald-glow);
    }

    .empty-title {
      font-family: var(--font-display);
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 0.5rem;
    }

    .empty-desc {
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    /* Modal Backdrop and Drawer */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(8px);
      z-index: 100;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }

    .modal-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }

    .slide-modal {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: 520px;
      max-width: 92vw;
      background: #0f1322;
      border-left: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: -10px 0 40px rgba(0, 0, 0, 0.7);
      z-index: 101;
      padding: 2.25rem;
      display: flex;
      flex-direction: column;
      transform: translateX(100%);
      transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1);
      overflow-y: auto;
    }

    .slide-modal.open {
      transform: translateX(0);
    }

    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
    }

    .modal-decision-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-display);
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.35rem 0.9rem;
      border-radius: 9999px;
    }

    .modal-decision-badge.approved {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }

    .modal-decision-badge.rejected {
      background: rgba(244, 63, 94, 0.15);
      border: 1px solid rgba(244, 63, 94, 0.4);
      color: #fb7185;
    }

    .close-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
      width: 36px;
      height: 36px;
      border-radius: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }

    .close-btn:hover {
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
    }

    .modal-section-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      font-weight: 700;
      margin-bottom: 0.6rem;
    }

    .review-terminal {
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 16px;
      padding: 1.25rem;
      font-family: 'Inter', sans-serif;
      font-size: 0.95rem;
      line-height: 1.6;
      color: #e2e8f0;
      box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
      margin-bottom: 2rem;
    }

    .metadata-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1rem;
      font-size: 0.8rem;
      color: #94a3b8;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      margin-bottom: 2rem;
    }

    .metadata-item {
      display: flex;
      justify-content: space-between;
    }

    .metadata-value {
      font-family: monospace;
      color: #e2e8f0;
    }

    .modal-btn-done {
      margin-top: auto;
      width: 100%;
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #ffffff;
      padding: 0.9rem;
      border-radius: 14px;
      border: none;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      box-shadow: 0 4px 20px var(--primary-glow);
      transition: all 0.2s;
    }

    .modal-btn-done:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(99, 102, 241, 0.5);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">
          <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
          </svg>
        </div>
        <div class="brand-text">
          <h1>Expense Governance Hub</h1>
          <p>
            <span>Agent Runtime Active</span>
            <span class="status-badge"><span class="status-dot"></span> HITL Ready</span>
          </p>
        </div>
      </div>

      <div class="header-actions">
        <div class="stats-pill">
          <span>Pending Actions:</span>
          <span class="stats-count" id="pendingCount">0</span>
        </div>
        <button class="refresh-btn" id="refreshBtn" onclick="loadPending()">
          <svg id="refreshIcon" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          <span>Refresh</span>
        </button>
      </div>
    </header>

    <div id="contentArea">
      <!-- Injected cards -->
    </div>
  </div>

  <!-- Slide-out Modal -->
  <div class="modal-backdrop" id="modalBackdrop" onclick="closeModal()"></div>
  <div class="slide-modal" id="slideModal">
    <div class="modal-header">
      <span class="modal-decision-badge approved" id="modalDecisionBadge">Approved</span>
      <button class="close-btn" onclick="closeModal()">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>

    <div class="modal-section-title">Agent Compliance Review</div>
    <div class="review-terminal" id="modalReviewText">
      <!-- Agent compliance text -->
    </div>

    <div class="modal-section-title">Execution Context</div>
    <div class="metadata-card">
      <div class="metadata-item">
        <span>Session ID</span>
        <span class="metadata-value" id="modalSessionId">-</span>
      </div>
      <div class="metadata-item">
        <span>Interrupt ID</span>
        <span class="metadata-value" id="modalInterruptId">-</span>
      </div>
      <div class="metadata-item">
        <span>Workflow Step</span>
        <span class="metadata-value">process_decision</span>
      </div>
    </div>

    <button class="modal-btn-done" onclick="closeModal()">Acknowledge & Close</button>
  </div>

  <script>
    let isFetching = false;
    let actionInProgress = false;
    let lastRenderedJson = '';

    async function loadPending() {
      if (isFetching || actionInProgress) return;
      isFetching = true;
      const refreshIcon = document.getElementById('refreshIcon');
      refreshIcon.classList.add('spin');
      
      try {
        const res = await fetch('/api/pending');
        const data = await res.json();
        const pending = data.pending || [];
        const currentJson = JSON.stringify(pending.map(p => ({ s: p.session_id, i: p.interrupt_id })));
        if (currentJson !== lastRenderedJson) {
          lastRenderedJson = currentJson;
          renderPending(pending);
        }
      } catch (err) {
        console.error('Error fetching pending approvals:', err);
      } finally {
        isFetching = false;
        setTimeout(() => refreshIcon.classList.remove('spin'), 500);
      }
    }

    function renderPending(items) {
      document.getElementById('pendingCount').innerText = items.length;
      const content = document.getElementById('contentArea');

      if (items.length === 0) {
        content.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">
              <svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
              </svg>
            </div>
            <h2 class="empty-title">All Approvals Clear</h2>
            <p class="empty-desc">There are currently no expense submissions paused for manager review.</p>
          </div>
        `;
        return;
      }

      let html = '<div class="cards-grid">';
      for (const item of items) {
        const p = item.payload || {};
        const amount = p.amount ? Number(p.amount).toFixed(2) : '0.00';
        const category = p.category || 'general';
        const desc = p.description || 'Expense request';
        const submitter = p.submitter || 'employee@example.com';
        const risk = p.risk_summary || 'Review required by enterprise policy threshold.';
        const sessionId = item.session_id;
        const interruptId = item.interrupt_id;

        html += `
          <div class="card" id="card-${sessionId}">
            <div class="card-top">
              <span class="category-badge ${category}">${category}</span>
              <div class="amount-tag">$${amount}</div>
            </div>
            <h3 class="card-title">${desc}</h3>
            <div class="submitter-row">
              <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207"/>
              </svg>
              <span>${submitter}</span>
            </div>

            <div class="risk-box">
              <div class="risk-header">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>
                <span>AI Risk Assessment</span>
              </div>
              <p class="risk-text">${risk}</p>
            </div>

            <div class="meta-details">
              <span>Session</span>
              <span class="session-chip">${sessionId.slice(0, 8)}...</span>
            </div>

            <div class="card-actions">
              <button class="btn btn-approve" id="btn-approve-${sessionId}" onclick="executeAction('${sessionId}', '${interruptId}', true)">
                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                </svg>
                <span>Approve</span>
              </button>
              <button class="btn btn-reject" id="btn-reject-${sessionId}" onclick="executeAction('${sessionId}', '${interruptId}', false)">
                <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
                </svg>
                <span>Reject</span>
              </button>
            </div>
          </div>
        `;
      }
      html += '</div>';
      content.innerHTML = html;
    }

    async function executeAction(sessionId, interruptId, approved) {
      actionInProgress = true;
      const btnApprove = document.getElementById(`btn-approve-${sessionId}`);
      const btnReject = document.getElementById(`btn-reject-${sessionId}`);
      
      btnApprove.disabled = true;
      btnReject.disabled = true;
      
      const activeBtn = approved ? btnApprove : btnReject;
      const originalText = activeBtn.innerHTML;
      activeBtn.innerHTML = `
        <svg class="spin" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" stroke-dasharray="32" stroke-linecap="round"/>
        </svg>
        <span>Resuming...</span>
      `;

      try {
        const res = await fetch(`/api/action/${sessionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            interrupt_id: interruptId,
            approved: approved
          })
        });

        if (!res.ok) {
          const err = await res.json();
          alert(`Error resuming agent: ${err.detail || 'Unknown error'}`);
          return;
        }

        const data = await res.json();
        showModal(data.decision, data.review, sessionId, interruptId);

        // Remove card from UI
        const card = document.getElementById(`card-${sessionId}`);
        if (card) {
          card.style.opacity = '0';
          card.style.transform = 'scale(0.9)';
          setTimeout(() => {
            card.remove();
            lastRenderedJson = '';
            const count = document.querySelectorAll('.card').length;
            document.getElementById('pendingCount').innerText = count;
            if (count === 0) loadPending();
          }, 300);
        }
      } catch (err) {
        console.error('Action failed:', err);
        alert('Failed to send manager action.');
      } finally {
        activeBtn.innerHTML = originalText;
        btnApprove.disabled = false;
        btnReject.disabled = false;
        actionInProgress = false;
      }
    }

    function showModal(decision, reviewText, sessionId, interruptId) {
      const badge = document.getElementById('modalDecisionBadge');
      badge.className = `modal-decision-badge ${decision.toLowerCase()}`;
      badge.innerText = decision.toUpperCase();

      document.getElementById('modalReviewText').innerText = reviewText;
      document.getElementById('modalSessionId').innerText = sessionId;
      document.getElementById('modalInterruptId').innerText = interruptId;

      document.getElementById('modalBackdrop').classList.add('open');
      document.getElementById('slideModal').classList.add('open');
    }

    function closeModal() {
      document.getElementById('modalBackdrop').classList.remove('open');
      document.getElementById('slideModal').classList.remove('open');
    }

    // Initial load and periodic polling every 5 seconds
    document.addEventListener('DOMContentLoaded', () => {
      loadPending();
      setInterval(loadPending, 5000);
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive manager dashboard HTML page."""
    return HTMLResponse(content=DASHBOARD_HTML)
