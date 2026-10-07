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

"""Serve the reasoning_engine ``{class_method, input}`` contract over HTTP.

Exists to guarantee support for the Vertex AI Console Playground and Gemini
Enterprise (via ADK registration), which both invoke the engine through this
contract. Agent Engine forwards calls to ``/api/reasoning_engine`` (sync) and
``/api/stream_reasoning_engine`` (streaming); dispatch is limited to the
:class:`AdkApp` ``register_operations()`` methods so the wire output matches a
packaged Agent Engine.
"""

import inspect
import json
from typing import Any

from agentplatform.agent_engines.templates.adk import AdkApp
from fastapi import FastAPI, HTTPException, Request, encoders, responses
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.app_utils import services


def _no_op_instrumentor_builder(project_id: str) -> None:
    """No-op so set_up() keeps the startup instrumentor and generate_content spans."""
    return None


def attach_reasoning_engine_routes(app: FastAPI) -> None:
    """Register reasoning_engine routes that dispatch to an AdkApp."""
    runtime: AdkApp | None = None
    streaming_methods: set[str] = set()
    sync_methods: set[str] = set()

    def get_runtime() -> AdkApp:
        nonlocal runtime, streaming_methods, sync_methods
        if runtime is None:
            from app.agent import app as adk_app

            # Reuse the process-wide services so sessions created here are
            # visible to the adk_api and A2A paths, and vice versa (see services.py).
            runtime = AdkApp(
                app=adk_app,
                session_service_builder=services.get_session_service,
                artifact_service_builder=services.get_artifact_service,
                instrumentor_builder=_no_op_instrumentor_builder,
            )
            runtime.set_up()
            operations = runtime.register_operations()
            streaming_methods = set(operations.get("stream", [])) | set(
                operations.get("async_stream", [])
            )
            sync_methods = set(operations.get("", [])) | set(
                operations.get("async", [])
            )
        return runtime

    def _extract_query_params(body: Any) -> tuple[str, dict[str, Any]]:
        """Normalize raw or enwrapped payloads into (method_name, kwargs)."""
        if isinstance(body, dict) and "class_method" in body:
            return body["class_method"], body.get("input") or {}

        # Raw payload (e.g. from Pub/Sub with --push-no-wrapper)
        raw_dict = body if isinstance(body, dict) else {}
        user_id = raw_dict.get("user_id") or "pubsub-user"
        session_id = raw_dict.get("session_id")

        if isinstance(body, str):
            msg = body
        else:
            msg = json.dumps(body)

        kwargs: dict[str, Any] = {
            "message": msg,
            "user_id": user_id,
        }
        if session_id:
            kwargs["session_id"] = session_id

        return "async_stream_query", kwargs

    @app.post("/api/stream_reasoning_engine")
    async def stream_query(request: Request) -> responses.StreamingResponse:
        try:
            body = await request.json()
        except Exception:
            raw_bytes = await request.body()
            body = raw_bytes.decode("utf-8", errors="replace")

        method_name, kwargs = _extract_query_params(body)
        rt = get_runtime()

        if method_name in streaming_methods:
            method = getattr(rt, method_name)
        elif method_name in ("query", "async_query"):
            method = getattr(rt, "async_stream_query")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupported reasoning_engine streaming method: {method_name!r}",
            )

        stream = (
            await method(**kwargs)
            if inspect.iscoroutinefunction(method)
            else method(**kwargs)
        )

        async def generator():
            if hasattr(stream, "__aiter__"):
                async for event in stream:
                    yield json.dumps(encoders.jsonable_encoder(event)) + "\n"
            else:
                async for event in iterate_in_threadpool(stream):
                    yield json.dumps(encoders.jsonable_encoder(event)) + "\n"

        return responses.StreamingResponse(
            content=generator(), media_type="application/json"
        )

    @app.post("/api/reasoning_engine")
    async def query(request: Request) -> responses.JSONResponse:
        try:
            body = await request.json()
        except Exception:
            raw_bytes = await request.body()
            body = raw_bytes.decode("utf-8", errors="replace")

        method_name, kwargs = _extract_query_params(body)
        rt = get_runtime()

        if method_name in sync_methods:
            method = getattr(rt, method_name)
            if inspect.iscoroutinefunction(method):
                output = await method(**kwargs)
            else:
                output = await run_in_threadpool(method, **kwargs)
            return responses.JSONResponse(
                content=encoders.jsonable_encoder({"output": output})
            )

        if method_name in ("query", "async_query", "stream_query", "async_stream_query"):
            # Consume the stream and return all events for synchronous callers (e.g. Pub/Sub push)
            events = []
            async for event in rt.async_stream_query(**kwargs):
                events.append(event)
            return responses.JSONResponse(
                content=encoders.jsonable_encoder({"output": events})
            )

        raise HTTPException(
            status_code=404,
            detail=f"Unsupported reasoning_engine method: {method_name!r}",
        )
