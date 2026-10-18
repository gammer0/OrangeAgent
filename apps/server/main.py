"""FastAPI 服务层：REST + SSE，为前端提供多会话、模型选择与实时推理流。

**边界说明**：本目录是**服务层**，可以依赖 FastAPI；
核心 ``src/harness`` 仍保持零 Web 框架（由 ``scripts/check_constraints.py`` 守护）。

启动::

    uv run uvicorn apps.server.main:app --reload --port 8000

接口一览：

=========================== ====== ====================================
方法                         路径    说明
=========================== ====== ====================================
GET                         /           前端页面
GET                         /api/health 服务健康
GET                         /api/models 可用模型列表
GET                         /api/tools  工具目录（路径 + 描述 + schema）
GET                         /api/sessions           会话列表
POST                        /api/sessions           新建会话
GET                         /api/sessions/{id}      会话详情（可含消息）
PATCH                       /api/sessions/{id}      修改模型
DELETE                      /api/sessions/{id}      删除会话
GET                         /api/sessions/{id}/messages 会话历史
POST                        /api/sessions/{id}/messages 发送消息（触发推理）
POST                        /api/sessions/{id}/cancel    中止本轮
GET                         /api/sessions/{id}/events    SSE 事件流
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
from harness.config import ConfigError, build_proxy, load_llm_config
from harness.defaults import DEFAULT_CONTEXT
from harness.rate import RateWindow

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class NewSessionBody(BaseModel):
    """新建会话请求体。"""

    model: str | None = Field(default=None, description="模型名，省略用默认")
    capacity: int | None = Field(default=None, gt=0, description="上下文容量")


class SendMessageBody(BaseModel):
    """发送消息请求体。"""

    text: str = Field(min_length=1, description="用户消息")
    channel: str = Field(
        default="chat", pattern="^(chat|control)$", description="chat=等待插入，control=立即插入"
    )


class UpdateSessionBody(BaseModel):
    """修改会话请求体。"""

    model: str | None = None


# ---------------------------------------------------------------------------
# 应用生命周期
# ---------------------------------------------------------------------------


class _State:
    """进程级状态：模型代理与会话管理器。"""

    def __init__(self) -> None:
        self.manager: SessionManager | None = None
        self.error: str | None = None
        self.workdir: Path = Path.cwd()


def _build_manager(workdir: Path) -> tuple[SessionManager | None, str | None]:
    """构建会话管理器。

    Returns:
        ``(manager, error)``；配置缺失时 manager 为 ``None``，error 说明原因。
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
    # 确保限流窗口存在（build_proxy 已创建，此处仅示意可覆盖）
    assert isinstance(proxy.rate_window, RateWindow)

    manager = SessionManager(
        workdir=str(workdir),
        proxy=proxy,
        default_model=config.model,
        capacity=DEFAULT_CONTEXT.capacity,
    )
    return manager, None


def create_app(workdir: str | Path | None = None) -> FastAPI:
    """构建 FastAPI 应用。

    Args:
        workdir: 工作目录（``.env`` 与工具根都相对它）；省略时取当前目录。

    Returns:
        可直接交给 uvicorn 的 ``FastAPI`` 实例。
    """
    state = _State()
    state.workdir = Path(workdir).resolve() if workdir else Path.cwd().resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.manager, state.error = _build_manager(state.workdir)
        yield
        state.manager = None

    app = FastAPI(
        title="Harness Web",
        version="0.1.0",
        description="Harness 服务前端：多会话、模型选择、上下文与实时推理",
        lifespan=lifespan,
    )
    app.state.harness = state

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def manager_or_503() -> SessionManager:
        """取管理器；未就绪时返回 503 并说明原因。"""
        if state.manager is None:
            raise HTTPException(
                status_code=503,
                detail=f"服务未就绪：{state.error or '配置缺失'}",
            )
        return state.manager

    # ---- 页面与健康 ----

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """返回单页前端。"""
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>前端资源缺失</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """服务健康状态。"""
        manager = state.manager
        return {
            "ok": manager is not None,
            "workdir": str(state.workdir),
            "sessions": len(manager.list()) if manager else 0,
            "error": state.error,
            "tools": len(manager.tools.store.active_paths()) if manager else 0,
            "skills": len(manager.capabilities.skills) if manager and manager.capabilities else 0,
            "capability_error": manager.capability_error if manager else None,
            "capability_file": str(
                state.workdir / ".agent" / "capabilities.json"
            ),
        }

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        """可用模型列表。"""
        manager = manager_or_503()
        return {"models": manager.models(), "default": manager.default_model}

    @app.get("/api/tools")
    async def list_tools() -> dict[str, Any]:
        """工具目录（供前端展示能力面）。"""
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

    # ---- 会话 ----

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        """会话列表。"""
        manager = manager_or_503()
        return {"sessions": [s.snapshot() for s in manager.list()]}

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: NewSessionBody | None = None) -> dict[str, Any]:
        """新建会话。"""
        manager = manager_or_503()
        payload = body or NewSessionBody()
        try:
            runner = manager.create(model=payload.model, capacity=payload.capacity)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return runner.snapshot(include_messages=True)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, include_messages: bool = True) -> dict[str, Any]:
        """会话详情。"""
        runner = _require(manager_or_503(), session_id)
        return runner.snapshot(include_messages=include_messages)

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, body: UpdateSessionBody) -> dict[str, Any]:
        """修改会话模型。"""
        runner = _require(manager_or_503(), session_id)
        if body.model:
            runner.model = body.model
            runner.emit("state", {"state": runner.state, "detail": f"model={body.model}"})
        return runner.snapshot()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        """删除会话。"""
        ok = manager_or_503().delete(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": session_id}

    @app.get("/api/sessions/{session_id}/messages")
    async def get_messages(session_id: str) -> dict[str, Any]:
        """会话历史（供上下文面板渲染）。"""
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
        """发送消息并触发一轮推理。"""
        runner = _require(manager_or_503(), session_id)
        if body.channel == "control":
            from harness.inbox import InterruptRequest, UserInput

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
        """请求中止本轮推理（在下一个安全点生效）。"""
        runner = _require(manager_or_503(), session_id)
        runner.cancel()
        return {"cancelling": True}

    # ---- SSE ----

    @app.get("/api/sessions/{session_id}/events")
    async def stream_events(session_id: str, request: Request) -> StreamingResponse:
        """SSE 事件流：assistant / tool / route / compress / usage / state / error。"""
        runner = _require(manager_or_503(), session_id)
        return StreamingResponse(
            _event_stream(runner, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _require(manager: SessionManager, session_id: str) -> SessionRunner:
    """取会话，不存在时 404。"""
    runner = manager.get(session_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return runner


async def _event_stream(runner: SessionRunner, request: Request) -> AsyncIterator[bytes]:
    """把会话事件队列转成 SSE 帧。

    每 15 秒发送一次心跳注释，避免中间代理断开空闲连接。
    """
    loop = asyncio.get_running_loop()
    last_heartbeat = loop.time()

    while True:
        if await request.is_disconnected():
            break

        try:
            event = await asyncio.wait_for(
                loop.run_in_executor(None, _poll, runner, 1.0), timeout=1.5
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


def _poll(runner: SessionRunner, timeout: float) -> Any:
    """在阻塞队列上等待一条事件（在线程池中执行）。"""
    try:
        return runner.events.get(timeout=timeout)
    except queue.Empty:
        return None


app = create_app()


if __name__ == "__main__":  # pragma: no cover - 手动启动入口
    import uvicorn

    uvicorn.run("apps.server.main:app", host="127.0.0.1", port=8000, reload=False)
