"""FastAPI service layer: REST + SSE, exposing multi-session, model selection and
live inference streaming to the frontend.

**Boundary note**: this directory is the **service layer** and may depend on
FastAPI; the core ``src/orange`` still keeps zero web-framework dependencies
(enforced by ``scripts/check_constraints.py``).

Start::

    uv run uvicorn apps.server.main:app --reload --port 8000

Endpoints:

=========================== ====== ====================================
Method                      Path   Description
=========================== ====== ====================================
GET                         /           frontend page
GET                         /api/health service health
GET                         /api/models available models
GET                         /api/tools  tool catalogue (path + description + schema)
GET                         /api/sessions           list sessions
POST                        /api/sessions           create a session
GET                         /api/sessions/{id}      session detail (may include messages)
PATCH                       /api/sessions/{id}      change the model
DELETE                      /api/sessions/{id}      delete the session
GET                         /api/sessions/{id}/messages session history
POST                        /api/sessions/{id}/messages send a message (triggers inference)
POST                        /api/sessions/{id}/cancel   abort this turn
GET                         /api/sessions/{id}/events   SSE event stream
=========================== ====== ====================================
"""

from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.server.sessions import SessionManager, SessionRunner
from orange.capabilities import manifest_path as capability_manifest_path
from orange.config import ConfigError, build_proxy, load_llm_config
from orange.defaults import DEFAULT_CONTEXT
from orange.rate import RateWindow

APP_DIR = Path(__file__).resolve().parent


def _static_dir() -> Path:
    """Locate the frontend assets, preferring the copy shipped inside the package.

    The frontend lives in ``orange/web`` so an installed wheel carries it: a
    directory beside this module only exists in a source checkout. Preferring the
    package copy means a pip-installed console and a dev checkout render the same
    UI rather than drifting apart.
    """
    import orange

    packaged = Path(orange.__file__).resolve().parent / "web"
    if packaged.is_dir():
        return packaged
    return APP_DIR / "static"


STATIC_DIR = _static_dir()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class NewSessionBody(BaseModel):
    """Create-session request body."""

    model: str | None = Field(default=None, description="模型名，省略用默认")
    capacity: int | None = Field(default=None, gt=0, description="上下文容量")


class SendMessageBody(BaseModel):
    """Send-message request body."""

    text: str = Field(min_length=1, description="用户消息")
    channel: str = Field(
        default="chat", pattern="^(chat|control)$", description="chat=等待插入，control=立即插入"
    )


class UpdateSessionBody(BaseModel):
    """Update-session request body."""

    model: str | None = None


class UpdateSettingsBody(BaseModel):
    """Settings write request body.

    ``values`` maps ``"group.name"`` to the new value, e.g.
    ``{"loop.max_steps": 60}``. Types are validated against the shipped schema by
    :mod:`orange.settings`, not here, so the two can never drift apart.
    """

    values: dict[str, Any] = Field(default_factory=dict, description="键值对，如 loop.max_steps")


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


class _State:
    """Process-level state: the model proxy and the session manager."""

    def __init__(self) -> None:
        self.manager: SessionManager | None = None
        self.error: str | None = None
        self.workdir: Path = Path.cwd()


def _build_manager(workdir: Path) -> tuple[SessionManager | None, str | None]:
    """Build the session manager.

    Returns:
        ``(manager, error)``; ``manager`` is ``None`` when configuration is
        missing, with ``error`` explaining why.
    """
    try:
        config = load_llm_config(dotenv_path=workdir / ".env")
    except ConfigError as exc:
        return None, str(exc)

    proxy = build_proxy(
        config,
        max_requests=120,
        max_tokens_per_window=400_000,
    )
    # Ensure the rate window exists (build_proxy already created it; shown here as
    # an override point)
    assert isinstance(proxy.rate_window, RateWindow)

    manager = SessionManager(
        workdir=str(workdir),
        proxy=proxy,
        default_model=config.model,
        capacity=DEFAULT_CONTEXT.capacity,
    )
    # P1: bring previously persisted sessions back into memory so a restart no
    # longer wipes the conversation list.
    manager.restore()
    return manager, None


def create_app(workdir: str | Path | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        workdir: Working directory (both ``.env`` and the tool root are relative
            to it); defaults to the current directory.

    Returns:
        A ``FastAPI`` instance ready for uvicorn.
    """
    state = _State()
    state.workdir = Path(workdir).resolve() if workdir else Path.cwd().resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.manager, state.error = _build_manager(state.workdir)
        if state.manager is not None:
            # Start the stall watchdog: a turn blocked on a dead connection holds
            # its runner slot, and the idle sweeper cannot reclaim an ``active``
            # session. Without this the service degrades quietly, one slot at a
            # time, until no session can start.
            state.manager.start_watchdog()
        yield
        if state.manager is not None:
            state.manager.stop_watchdog()
        state.manager = None

    app = FastAPI(
        title="Orange Web",
        version="0.1.0",
        description="Orange 服务前端：多会话、模型选择、上下文与实时推理",
        lifespan=lifespan,
    )
    app.state.orange = state

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def manager_or_503() -> SessionManager:
        """Return the manager; 503 with the reason when it is not ready."""
        if state.manager is None:
            raise HTTPException(
                status_code=503,
                detail=f"服务未就绪：{state.error or '配置缺失'}",
            )
        return state.manager

    # ---- pages and health ----

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Return the single-page frontend."""
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>前端资源缺失</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Service health status."""
        manager = state.manager
        return {
            "ok": manager is not None,
            "workdir": str(state.workdir),
            "sessions": len(manager.list()) if manager else 0,
            "error": state.error,
            "tools": len(manager.tools.store.active_paths()) if manager else 0,
            "skills": len(manager.capabilities.skills) if manager and manager.capabilities else 0,
            "capability_error": manager.capability_error if manager else None,
            "compress_available": manager.compress_available if manager else False,
            "capability_file": str(
                capability_manifest_path(state.workdir)
            ),
        }

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        """Available model list."""
        manager = manager_or_503()
        return {"models": manager.models(), "default": manager.default_model}

    @app.get("/api/tools")
    async def list_tools() -> dict[str, Any]:
        """Tool catalogue (so the frontend can show the capability surface)."""
        manager = manager_or_503()
        entries = manager.tools.store.active_paths()
        return {
            "tools": [
                {
                    "path": entry.path,
                    "description": entry.description,
                    "concurrency": entry.concurrency,
                    "idempotent": entry.idempotent,
                }
                for entry in sorted(entries.values(), key=lambda e: e.path)
            ]
        }

    # ---- settings ----

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """Effective hyperparameters grouped by area, with override provenance.

        ``overridden`` tells the UI whether a value came from ``settings.json`` or
        is still the shipped default, so it can offer a per-item reset.
        """
        from orange.settings import effective_config, settings_path

        return {
            "groups": effective_config(),
            "file": str(settings_path()),
        }

    @app.patch("/api/settings")
    async def update_settings(body: UpdateSettingsBody) -> dict[str, Any]:
        """Persist hyperparameter overrides.

        Only whitelisted keys are accepted; secrets are deliberately absent from
        the schema, so this endpoint cannot be used to write an API key to disk.
        A value marked ``restart_required`` still needs a restart -- the response
        says which ones so the UI can prompt instead of silently lying.
        """
        from orange.settings import (
            RESTART_REQUIRED,
            SettingsError,
            effective_config,
            write_overrides,
        )

        try:
            # ``only_keys`` keeps the UI inside its documented surface; anything
            # else must go through the CLI, where the change is reviewable.
            write_overrides(body.values, only_keys=True)
        except SettingsError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        needs_restart = sorted(k for k in body.values if k in RESTART_REQUIRED)
        return {
            "groups": effective_config(),
            "applied": sorted(k for k in body.values if k not in RESTART_REQUIRED),
            "restart_required": needs_restart,
        }

    @app.delete("/api/settings")
    async def reset_settings(keys: str | None = None) -> dict[str, Any]:
        """Drop overrides so the shipped defaults apply again.

        Args:
            keys: Comma-separated list; omit to reset everything.
        """
        from orange.settings import clear_overrides, effective_config

        targets = [k.strip() for k in keys.split(",")] if keys else None
        clear_overrides(targets)
        return {"groups": effective_config()}

    # ---- sessions ----

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        """Session list."""
        manager = manager_or_503()
        return {"sessions": [s.snapshot() for s in manager.list()]}

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: NewSessionBody | None = None) -> dict[str, Any]:
        """Create a session."""
        manager = manager_or_503()
        payload = body or NewSessionBody()
        try:
            runner = manager.create(model=payload.model, capacity=payload.capacity)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return runner.snapshot(include_messages=True)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, include_messages: bool = True) -> dict[str, Any]:
        """Session detail."""
        runner = _require(manager_or_503(), session_id)
        return runner.snapshot(include_messages=include_messages)

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, body: UpdateSessionBody) -> dict[str, Any]:
        """Change the session's model."""
        runner = _require(manager_or_503(), session_id)
        if body.model:
            runner.model = body.model
            runner.emit("state", {"state": runner.state, "detail": f"model={body.model}"})
        return runner.snapshot()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        """Delete a session."""
        ok = manager_or_503().delete(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": session_id}

    @app.get("/api/sessions/{session_id}/messages")
    async def get_messages(session_id: str) -> dict[str, Any]:
        """Session history (for the context panel to render)."""
        runner = _require(manager_or_503(), session_id)
        snapshot = runner.snapshot(include_messages=True)
        return {
            "session_id": session_id,
            "messages": snapshot["messages"],
            "tokens": snapshot["tokens"],
            "capacity": snapshot["capacity"],
        }

    @app.post("/api/sessions/{session_id}/messages", status_code=202)
    async def send_message(session_id: str, body: SendMessageBody) -> dict[str, Any]:
        """Send a message and trigger one inference turn."""
        runner = _require(manager_or_503(), session_id)
        if body.channel == "control":
            from orange.inbox import InterruptRequest, UserInput

            runner.inbox.put_control(
                InterruptRequest(reason="user_abort", text=body.text)
            )
            runner.inbox.put_chat(UserInput(text=body.text, channel="ws"))
            return {"accepted": True, "channel": "control"}

        accepted = runner.start(body.text)
        if not accepted:
            raise HTTPException(status_code=409, detail="该会话正在推理中")
        return {"accepted": True, "channel": "chat"}

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> dict[str, Any]:
        """Request abort of this turn (takes effect at the next safe point)."""
        runner = _require(manager_or_503(), session_id)
        runner.cancel()
        return {"cancelling": True}

    @app.get("/api/sessions/{session_id}/trace")
    async def get_trace(session_id: str) -> dict[str, Any]:
        """Persisted audit trail (tool activity, routing, tree descents).

        Kept apart from the message history on purpose: the conversation is the
        model's context and prunes discovery scaffolding once it has served its
        purpose, while the trace is what the user reads to see how a tool was
        found -- including after a restart.

        Reads the table rather than the replay buffer: the buffer is bounded (it
        only serves as SSE backlog), while the audit trail must not be truncated.
        """
        runner = _require(manager_or_503(), session_id)
        events: list[dict[str, Any]] = []
        if runner.store is not None:
            # Flush first: events emitted since the last safe point are still in
            # the runner's buffer, and a page load must not miss them.
            runner.persist_trace()
            try:
                for row in runner.store.load_trace_events(session_id):
                    try:
                        payload = json.loads(row.payload)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    events.append(
                        {
                            "seq": row.seq,
                            "ts": row.created_at,
                            "kind": row.kind,
                            "payload": payload,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - a damaged trace is not fatal
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"session_id": session_id, "events": events}

    # ---- SSE ----

    @app.get("/api/sessions/{session_id}/events")
    async def stream_events(
        session_id: str, request: Request, after_seq: int = 0
    ) -> StreamingResponse:
        """SSE event stream: assistant / tool / route / compress / usage / state / error.

        Each connection subscribes independently (broadcast fan-out), so opening a
        second tab cannot steal events from the first. ``Last-Event-ID`` replays
        whatever the client missed while disconnected; ``after_seq`` does the same
        for a client that cannot set headers (``EventSource``) and has already
        loaded the persisted trace, which would otherwise be replayed twice.
        """
        runner = _require(manager_or_503(), session_id)
        try:
            header_seq = int(request.headers.get("last-event-id") or 0)
        except ValueError:
            header_seq = 0
        resume = max(header_seq, after_seq)
        return StreamingResponse(
            _event_stream(runner, request, resume),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _require(manager: SessionManager, session_id: str) -> SessionRunner:
    """Look up a session; 404 when absent."""
    runner = manager.get(session_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return runner


async def _event_stream(
    runner: SessionRunner,
    request: Request,
    after_seq: int = 0,
) -> AsyncIterator[bytes]:
    """Convert one subscriber's event queue into SSE frames.

    The queue is private to this connection (see ``SessionRunner.subscribe``) and
    unsubscribed on exit, so a closed tab releases its buffer instead of leaking
    it. A heartbeat comment is sent every 15 seconds so intermediary proxies do
    not drop an idle connection.
    """
    loop = asyncio.get_running_loop()
    subscription = runner.subscribe(after_seq)
    last_heartbeat = loop.time()

    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, _poll, subscription, 1.0),
                    timeout=1.5,
                )
            except TimeoutError:
                event = None

            if event is not None:
                data = json.dumps(event.to_dict(), ensure_ascii=False)
                yield f"id: {event.seq}\ndata: {data}\n\n".encode()
                last_heartbeat = loop.time()
                continue

            if loop.time() - last_heartbeat > 15:
                yield b": ping\n\n"
                last_heartbeat = loop.time()
    finally:
        runner.unsubscribe(subscription)


def _poll(q: Any, timeout: float) -> Any:
    """Wait for one event on a blocking queue (runs in a thread pool)."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual startup entry
    import uvicorn

    uvicorn.run("apps.server.main:app", host="127.0.0.1", port=8000, reload=False)
