"""Session runner: wraps AgentLoop into a run unit the web layer can subscribe to.

Design notes:

* **Do not modify the core**: ``src/harness`` keeps zero web dependencies; this
  module is adaptation only.
* **Event stream, not polling**: each session owns an event queue that the web
  layer consumes over SSE.
* **Safe-point injection**: external messages go through ``Inbox`` and are
  consumed by ``AgentLoop`` at safe points, so an uncommitted tool group is never
  interrupted (I6).
* **Cancellation**: implemented via ``user_abort`` on ``Inbox.control``,
  following G1 (stop remaining groups + fill 3e).

Event kinds (for frontend rendering):

============ ==========================================================
kind         Meaning
============ ==========================================================
``state``    session state change (idle / running / done / aborted / error)
``assistant``model reply (content / tool_calls / is_final)
``tool``     tool result (path / ok / error_type / duration_ms / summary)
``route``    Jev routing decision (branch / fallback / confidence)
``compress`` C3 compression happened (action / token change)
``inject``   user injection (channel / text)
``usage``    token usage and watermark ratio
``error``    error (message)
============ ==========================================================
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.capabilities import CapabilityError, CapabilityManifest, load_manifest
from harness.context import ContextAssembler, SystemPromptParts
from harness.discovery import DiscoveredTools, ToolDiscovery
from harness.executor import execute_batch
from harness.inbox import Inbox
from harness.jev import JevRouter
from harness.loop import AgentLoop, LoopHooks, StepController
from harness.memory import ensure_working_memory
from harness.message import AssistantMsg, Message, SystemMsg, ToolCall, ToolMsg, UserMsg
from harness.model import ReverseProxy
from harness.permission import AgentPermissions
from harness.permission_store import WEB_AGENT, load_permissions
from harness.session import Session, SessionRecord
from harness.session_store import SessionStore
from harness.tool_tree import EMPTY_SCHEMA, FlatStore, ToolDef, build_tree

__all__ = [
    "SessionEvent",
    "SessionRunner",
    "SessionManager",
    "DemoToolRegistry",
]

#: Web session state -> persisted ``sessions.status`` (the core status vocabulary
#: is only ``active | idle | ended``). ``running`` maps to ``active``; the
#: terminal web states (done/aborted/error) map to ``idle`` because the session is
#: still resumable.
_PERSIST_STATUS: dict[str, str] = {
    "running": "active",
    "idle": "idle",
    "done": "idle",
    "aborted": "idle",
    "error": "idle",
    "ended": "ended",
}

#: Per-subscriber queue cap; the oldest events are dropped beyond it so a slow
#: client cannot drag the service down.
EVENT_QUEUE_MAX = 2_000

#: Number of recent events retained for replay to a late or reconnecting client.
EVENT_REPLAY_MAX = 500

#: How deep the tool index goes when advertised to the model. ``1`` exposes only
#: the resource roots (``/tool``, ``/mcp``, ``/skill``), forcing a descent.
INDEX_DEPTH = 1


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One event that can be pushed to the frontend.

    Attributes:
        seq: Monotonically increasing per session, for deduplication and
            reconnection replay in the frontend.
        ts: Event timestamp.
        kind: Event kind (see the module docstring).
        payload: Event payload; must be JSON-serializable.
    """

    seq: int
    ts: float
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Export as a JSON-friendly dict."""
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "payload": self.payload,
        }


class DemoToolRegistry:
    """Sample tool set: read a file, list a directory, write a file, echo.

    A real deployment should provide these via config/plugins; they live inside
    the service layer here so the core package stays clean.
    """

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir
        self.store = FlatStore()
        for entry in self._definitions():
            self.store.put(entry)
        self.permissions = AgentPermissions(agent_id="web")
        self.tree = build_tree(self.store, self.permissions)

    def _definitions(self) -> list[ToolDef]:
        return [
            ToolDef(
                tool_id="t_read_file",
                path="/tool/os/file/read",
                fn=self._read_file,
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                caps=["read"],
                description="读取文件内容",
            ),
            ToolDef(
                tool_id="t_list_dir",
                path="/tool/os/file/list",
                fn=self._list_dir,
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                caps=["read"],
                description="列出目录条目",
            ),
            ToolDef(
                tool_id="t_write_file",
                path="/tool/os/file/write",
                fn=self._write_file,
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                caps=["write"],
                description="写入文件（覆盖）",
            ),
            ToolDef(
                tool_id="t_echo",
                path="/tool/os/echo",
                fn=self._echo,
                schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                caps=["read"],
                description="原样回显文本，用于连通性测试",
            ),
        ]

    # ---- tool implementations (all confined to the working directory) ----

    def _resolve(self, raw: str) -> str:
        from pathlib import Path

        base = Path(self.workdir).resolve()
        target = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"路径越界: {raw}")
        return str(target)

    def _read_file(self, path: str) -> str:
        from pathlib import Path

        target = Path(self._resolve(path))
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:20_000]

    def _list_dir(self, path: str = ".") -> str:
        from pathlib import Path

        target = Path(self._resolve(path))
        if not target.is_dir():
            raise NotADirectoryError(f"不是目录: {path}")
        items = sorted(target.iterdir(), key=lambda p: p.name)[:200]
        return "\n".join(f"{'D' if p.is_dir() else 'F'} {p.name}" for p in items)

    def _write_file(self, path: str, content: str) -> str:
        from pathlib import Path

        target = Path(self._resolve(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 字符到 {path}"

    def _echo(self, text: str) -> str:
        return text

    # ---- resolver used by AgentLoop ----

    def resolve(self, call: ToolCall) -> Any:
        """Resolve a ToolCall into an executable definition.

        Delegates to the tree rather than the flat store so that intermediate
        paths come back with their descent function bound; calling ``/tool/os``
        then lists its children instead of failing.
        """
        return self.tree.resolve(call.name)

    def tool_specs(
        self,
        discovered: tuple[set[str], object] | None = None,
    ) -> list[dict[str, Any]]:
        """Export OpenAI tool declarations for model requests.

        The shallow index (resource roots) is always advertised, and so is
        **everything discovered so far**. Declaring only the roots is not enough:
        after descending to ``/tool`` the model learns ``/tool/os`` exists, but an
        undeclared name cannot be called -- the model then falls back to repeating
        ``/tool`` and never reaches a leaf (observed: nine descents, zero tools
        called).

        Args:
            discovered: Optional ``(expanded_paths, inventory)`` pair. Intermediate
                paths found by descent stay callable, and leaf tools found by
                descent become callable; both are layered on top of the roots.
        """
        specs: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(entry: Any) -> None:
            if entry.path in seen:
                return
            seen.add(entry.path)
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.path,
                        "description": entry.description,
                        "parameters": entry.schema or dict(EMPTY_SCHEMA),
                    },
                }
            )

        for entry in self.tree.index_specs(depth=INDEX_DEPTH):
            add(entry)

        if discovered is not None:
            expanded, inventory = discovered
            paths = sorted(expanded)
            paths += getattr(inventory, "paths", lambda: [])()
            for path in paths:
                try:
                    add(self.tree.resolve(path))
                except Exception:  # noqa: BLE001 - a stale path must not break the request
                    continue

        return specs


@dataclass
class SessionRunner:
    """Runtime container for a single session.

    Attributes:
        session_id: Session identifier.
        model: Currently selected model name (switchable from the frontend).
        runner: Background thread.
    """

    session_id: str
    workdir: str
    proxy: ReverseProxy
    tools: DemoToolRegistry
    model: str
    capacity: int = 32_768

    inbox: Inbox = field(default_factory=Inbox)
    messages: list[Message] = field(default_factory=list)
    state: str = "idle"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    #: Core-layer session object (S4): lets the registry's eviction policy observe
    #: the real busy/idle state.
    core_session: Any = None
    #: SQLite persistence (P1). ``None`` degrades to memory-only operation.
    store: Any = None
    #: Discovered-tools inventory: children returned by descents. Injected as a
    #: trailing temporary section, never written into the message list.
    discovered: DiscoveredTools = field(default_factory=DiscoveredTools)
    #: Tool-call ids belonging to tree descents. They stay in the live message
    #: list (the model needs the children) but are skipped when persisting.
    _descent_call_ids: set[str] = field(default_factory=set)
    #: Intermediate paths the model has opened. These are not capabilities and are
    #: not cached, but they must be declared as callable so the model can descend
    #: further -- an undeclared name cannot be invoked.
    _expanded_paths: set[str] = field(default_factory=set)
    #: Number of messages already written to SQLite. Everything before this index
    #: is durable; everything from it on is still memory-only. This cursor is what
    #: makes persistence incremental instead of rewriting the whole history.
    persisted_count: int = 0

    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _cancelled: bool = False
    #: Message index -> timestamp (for the frontend to render history times
    #: without polluting core message types)
    _timestamps: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Broadcast fan-out. A single shared queue would hand every event to
        # exactly one reader, so a second tab (or a reconnect while the old
        # connection is still draining) silently steals events from the first --
        # which leaves the UI stuck on "running". Instead every SSE client gets
        # its own queue, and recent events are retained so a late subscriber can
        # be replayed up to the point it connects.
        self._subscribers: list[queue.Queue[SessionEvent]] = []
        self._recent: list[SessionEvent] = []
        #: Recently issued tool calls by id, so the trace can show the arguments
        #: when the matching result arrives (see _on_assistant / _on_group_done).
        self._calls_by_id: dict[str, ToolCall] = {}
        #: Messages already reflected in the trace; anything after this index is
        #: new and still needs a tool event.
        self._traced_count: int = 0
        self.assembler = ContextAssembler(
            capacity=self.capacity,
            summarizer=self._summarize,
        )
        self.system = self.assembler.build_system(
            SystemPromptParts(
                # Shallow index: only the resource roots. The model discovers
                # leaves by calling an intermediate path, which returns its
                # children (see ToolDef.with_query).
                tool_index=self.tools.tree.tree_index(depth=INDEX_DEPTH),
                working_dir=self.workdir,
                goal=None,
            )
        )
        self.messages = [self.system]
        # Trace cursor: everything already in ``messages`` counts as announced, so
        # a fresh or restored session never replays its history into the panel.
        self._traced_count = len(self.messages)

    # ----------------------------------------------------------- summary

    #: Input cap for a summarization request (characters). Beyond it, the history
    #: is chunked by message type; see :meth:`_summarize`.
    SUMMARY_INPUT_CHARS = 24_000

    #: Compression rules (dictated by the system; the model must not improvise).
    #: Note: this text is model-visible, so it intentionally stays in Chinese.
    SUMMARY_SYSTEM_PROMPT = """\
你是上下文压缩器。输入是一段带类型标记的对话历史，你必须按**每种标记**的专属规则压缩，
不能把所有内容一视同仁地"总结成要点"。

## 各类型标记的处理规则

[user] 用户指令
- **逐条保留**原始意图，不得改写措辞。若有多条，按时间顺序合并成编号列表。
- 用户给的约束、路径、命名、禁止项，逐字保留。

[assistant:call <工具路径>] 模型发起的工具调用
- 只保留最终有效的调用及其意图（例："写 textstat/core.py"）。
- 同一路径被反复调用的，合并成一条并注明次数（例："read main.py ×3"）。
- 参数内容一律丢弃，除非参数里包含用户要求的字面量（如文件名、常量值）。

[tool:<工具路径>] 工具执行结果
- **只保留结论，不保留原文**：
  - 写文件类：`已写入 <路径>（N 字符）`
  - 读文件类：`读取 <路径>（M 行）`，仅在后续有依赖时才提及其中的关键符号名
  - 执行/测试类：`退出码 N` + 一行关键输出；失败时必须保留错误类型与出错位置
  - 列表/查询类：只保留条目数量与最相关的 3 条
- 被截断的结果（含"字符已省略"）标注为 `（原输出已截断）`。
- 严禁复述大段代码、JSON、日志。

[assistant] 无工具调用的模型文本
- 只保留其中的**决定**与**对用户的承诺**；过程性描述、自我解释、寒暄全删。

## 输出格式（严格遵守）

```
## 目标
<一句话，来自 [user]>

## 已完成
- <动作> → <产物路径 / 结果>
（按时间顺序，每条一行，不写细节）

## 关键决定
- <决定> — <一句话理由>

## 待处理
- <未完成事项或已知问题>

## 环境坑
- <踩过的错误与正确做法>
（没有则写"无"）
```

## 硬约束
- 只输出上述五个小节，不要额外解释、不要复述规则。
- 总长度不超过输入字符数的 15%。
- 信息不足的小节写"无"，不要编造。
"""

    def _summarize(self, text: str) -> str:
        """Context-free single-turn summarization: fold history per the system's per-type rules.

        Chunking strategy: when the input exceeds :attr:`SUMMARY_INPUT_CHARS` it is
        split into equal chunks, each compressed separately, and the partials are
        merged once at the end.
        **Nothing is discarded** -- hard truncation would silently lose history,
        which is the least acceptable failure mode for compression.

        ``proxy`` is shared with the main loop, but a distinct ``session_id``
        suffix is used so the main session's usage accounting is not polluted;
        **no tools are passed**, so the model can only reply with text and cannot
        trigger further tool calls.
        """

        # Note: the chunk budget must subtract the header marker length (e.g.
        # "[this is chunk 1/3 ...]"), otherwise concatenation slightly exceeds
        # SUMMARY_INPUT_CHARS.
        budget = self.SUMMARY_INPUT_CHARS - 80
        chunks = [
            text[start : start + budget] for start in range(0, len(text), budget)
        ] or [text]

        if len(chunks) == 1:
            return self._summarize_once(chunks[0])

        # Multiple chunks: compress each, then merge the partials (the merge input
        # is usually far smaller than the original).
        partials = [
            self._summarize_once(chunk, index=i + 1, total=len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        merged = "\n\n".join(partials)
        # The merge stage is not hard-truncated either: if still over the limit,
        # recurse with the same strategy.
        if len(merged) > self.SUMMARY_INPUT_CHARS:
            return self._summarize(merged)
        return self._summarize_once(merged, index=1, total=1, merge=True)

    def _summarize_once(
        self,
        text: str,
        *,
        index: int = 1,
        total: int = 1,
        merge: bool = False,
    ) -> str:
        """Perform a single summarization call."""
        from harness.message import SystemMsg, UserMsg

        header = ""
        if total > 1 and not merge:
            header = f"[这是第 {index}/{total} 段，请只压缩本段内容]\n\n"
        elif merge:
            header = "[以下是同一段历史的多份分段摘要，请合并去重为一份]\n\n"

        response = self.proxy.chat(
            [
                SystemMsg(content=self.SUMMARY_SYSTEM_PROMPT),
                UserMsg(content=header + text),
            ],
            session_id=f"{self.session_id}:summary",
            model=self.model,
        )
        summary = (response.message.content or "").strip()
        return summary or "[摘要生成失败，原样保留]"

    # -------------------------------------------------------------- events

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        """Publish one event to every current subscriber (dropping on overflow)."""
        with self._lock:
            self._seq += 1
            event = SessionEvent(self._seq, time.time(), kind, payload)
            # Retain a bounded history for late subscribers.
            self._recent.append(event)
            if len(self._recent) > EVENT_REPLAY_MAX:
                del self._recent[: len(self._recent) - EVENT_REPLAY_MAX]
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.put_nowait(event)
            except queue.Full:
                # A slow reader must not stall the producer: drop its oldest.
                try:
                    sub.get_nowait()
                    sub.put_nowait(event)
                except (queue.Empty, queue.Full):  # pragma: no cover - race fallback
                    pass
        self.updated_at = time.time()

    # ---------------------------------------------------------- streaming

    def subscribe(self, after_seq: int = 0) -> queue.Queue[SessionEvent]:
        """Register a new SSE consumer and return its private queue.

        Args:
            after_seq: Only replay retained events with ``seq > after_seq``. Pass
                ``0`` for "everything still retained", or the last seq the client
                processed when reconnecting (``Last-Event-ID``) to avoid
                duplicates.

        Returns:
            A queue that receives every future event plus the replay backlog.
        """
        q: queue.Queue[SessionEvent] = queue.Queue(maxsize=EVENT_QUEUE_MAX)
        with self._lock:
            backlog = [e for e in self._recent if e.seq > after_seq]
            # Register before filling so no event can slip between the two steps.
            self._subscribers.append(q)
        for event in backlog:
            try:
                q.put_nowait(event)
            except queue.Full:  # pragma: no cover - backlog never exceeds capacity
                break
        return q

    def unsubscribe(self, q: queue.Queue[SessionEvent]) -> None:
        """Remove a consumer's queue (idempotent)."""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        # S4: sync web state into the core Session so the eviction policy
        # observes busy/idle.
        core = self.core_session
        if core is not None:
            if state == "running":
                core.busy()
            elif state in {"done", "aborted", "error", "idle"}:
                core.release()
        # P1: durable status must follow the in-memory state machine, otherwise a
        # restart would resurrect sessions in the wrong state.
        self.persist_status(state)
        self.emit("state", {"state": state, "detail": detail})

    # --------------------------------------------------------- persistence

    def persist_status(self, status: str | None = None) -> None:
        """Write this session's row to SQLite (status defaults to ``self.state``).

        Failures are swallowed on purpose: persistence is a durability feature,
        so a disk problem must never take down a live conversation.
        """
        if self.store is None:
            return
        mapped = _PERSIST_STATUS.get(status or self.state, "idle")
        try:
            self.store.update_status(self.session_id, mapped)
        except Exception as exc:  # noqa: BLE001 - persistence must not break the turn
            self.emit("error", {"message": f"persistence: {exc}"})

    def persist_pending(self) -> int:
        """Flush every not-yet-persisted message to SQLite in one transaction.

        The transaction unit is the batch produced between two safe points — one
        ``ExecGroup``'s assistant reply plus its backfilled tool results — which
        satisfies H7. ``SystemMsg`` is skipped (H6): the context layer rebuilds it
        from the current tool tree on restore.

        Tree descents **are** persisted: the trace the user reads must show the
        full discovery path (``/tool`` → ``/tool/os`` → ... → leaf), and after a
        restart that history is the only remaining evidence of how the tool was
        found. Keeping them out of the *model's* context is a separate concern,
        handled when a request is assembled.

        Returns:
            The number of messages written by this call.
        """
        if self.store is None:
            return 0
        anchor = self.persisted_count
        if anchor >= len(self.messages):
            return 0

        pending: list[Message] = [
            message
            for message in self.messages[anchor:]
            if not isinstance(message, SystemMsg)
        ]

        # Advance the cursor even when only a SystemMsg was pending, otherwise the
        # next call would rescan the same slice forever.
        self.persisted_count = len(self.messages)
        if not pending:
            return 0
        try:
            self.store.append_messages(self.session_id, pending)
        except Exception as exc:  # noqa: BLE001 - persistence must not break the turn
            # Restore the anchor so the next safe point retries this batch.
            self.persisted_count = anchor
            self.emit("error", {"message": f"persistence: {exc}"})
            return 0
        return len(pending)

    # --------------------------------------------------------------- run

    @property
    def running(self) -> bool:
        """Whether inference is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, text: str) -> bool:
        """Start one inference turn; returns ``False`` when a turn is already running."""
        if self.running:
            return False
        self._cancelled = False
        self.messages.append(UserMsg(content=text))
        self._timestamps[len(self.messages) - 1] = time.time()
        # P1: persist the user turn immediately -- if the process dies mid-reasoning,
        # the request the user typed must survive.
        self.persist_pending()
        # Note: do not emit inject here -- when AgentLoop inserts at a safe point
        # it fires the ``on_inject`` hook, avoiding duplicate user messages in
        # the frontend.
        self._thread = threading.Thread(target=self._run_once, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Request abort of this turn (takes effect at the next safe point)."""
        from harness.inbox import InterruptRequest

        self._cancelled = True
        self.inbox.put_control(InterruptRequest(reason="user_abort", text=None))

    def _run_once(self) -> None:
        """Run one complete ReAct loop and translate the process into events."""
        self._set_state("running")
        # The trace cursor is maintained by _on_group_done (it always points just
        # past what has already been announced), so it must not be reset here:
        # start() has already appended this turn's UserMsg.
        self._calls_by_id.clear()
        # Note: _descent_call_ids is intentionally NOT cleared. It holds descents
        # from earlier turns and from restored history, which must stay known so
        # they are omitted from this request's projection.

        hooks = LoopHooks(
            on_assistant=self._on_assistant,
            on_group_done=self._on_group_done,
            on_batch_done=self._on_batch_done,
            on_inject=self._on_inject,
            on_abort=self._on_abort,
            on_summary=self._on_summary,
            on_descent=self._on_descent,
        )

        controller = StepController(max_steps=20)
        loop = AgentLoop(
            model=self._model_client(),
            inbox=self.inbox,
            executor=self._executor,
            resolver=self.tools.resolve,
            system_msg=self.system,
            controller=controller,
            hooks=hooks,
            assembler=self.assembler,
            router=JevRouter(chat=self._router_chat),
            branch_provider=self._branches,
            tool_count_provider=lambda: len(self.tools.store.active_paths()),
            history_tokens_provider=lambda n: self.assembler.estimate(self.messages),
        )
        # Tell the loop which paths are descents, so their turns can be omitted
        # from the model's context while staying in the user's trace.
        loop.descent_paths = self.tools.tree.abstract_paths()
        loop.descent_call_ids = self._descent_call_ids

        try:
            # Pass the full history (including the UserMsg just appended);
            # AgentLoop uses it directly as its working copy.
            result = loop.run(self.messages)
            self.messages = list(result.messages)
            # Stamp newly produced messages with the current time (approximate;
            # does not affect semantics).
            now = time.time()
            for index in range(len(self.messages)):
                self._timestamps.setdefault(index, now)
            if result.route is not None:
                self.emit(
                    "route",
                    {
                        "branch": result.route.branch,
                        "confidence": result.route.confidence,
                        "fallback": result.route.fallback,
                    },
                )
            self._emit_usage()
            if result.aborted:
                self._set_state("aborted", result.stop_reason or "aborted")
            else:
                self._set_state("done", result.stop_reason or "final")
        except Exception as exc:  # noqa: BLE001 - session isolation: one failure must not affect the service
            self.emit("error", {"message": str(exc)})
            self._set_state("error", str(exc))
        finally:
            # P1: the loop mutates a working copy, so the runner's own list is only
            # authoritative once results are back. Flush whatever is still pending
            # so a crash right after this turn cannot lose the answer.
            self.persist_pending()

    def _model_client(self) -> Any:
        """Adapt ReverseProxy to the orchestration layer's ``chat(messages)`` protocol."""
        session_id = self.session_id

        class _Client:
            def __init__(self, owner: SessionRunner) -> None:
                self.owner = owner

            def chat(self, messages: list[Message]) -> AssistantMsg:
                response = self.owner.proxy.chat(
                    messages,
                    session_id=session_id,
                    # Declare the roots plus everything discovered so far, so a
                    # node the model just learned about is actually callable.
                    tools=self.owner.tools.tool_specs(
                        (self.owner._expanded_paths, self.owner.discovered)
                    ),
                    model=self.owner.model,
                    discovered=self.owner.discovered_rows(),
                )
                return response.message

        return _Client(self)

    def _router_chat(self, messages: list[Message]) -> AssistantMsg:
        """Single-turn call for Jev routing.

        Does not reuse the main session id, keeping usage accounting clean.
        """
        response = self.proxy.chat(
            messages,
            session_id=f"{self.session_id}:route",
            model=self.model,
        )
        return response.message

    def _branches(self) -> list[str]:
        """Candidate top-level branches: abstract nodes at L1 + L2."""
        return [p for p in self.tools.tree.paths if p.count("/") <= 2]

    def _executor(
        self,
        plan: Any,
        resolver: Any,
        on_group_done: Any = None,
        **kw: Any,
    ) -> dict[str, Any]:
        """Wrap the core executor to measure duration and emit events."""
        return execute_batch(plan, resolver, on_group_done, **kw)

    # -------------------------------------------------------------- hooks

    def _on_assistant(self, message: AssistantMsg) -> None:
        # Remember this reply's calls so the trace can show their arguments when
        # the matching tool results arrive later (see _on_group_done).
        for call in message.tool_calls:
            self._calls_by_id[call.id] = call
        self.emit(
            "assistant",
            {
                "content": message.content or "",
                "final": message.is_final,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in message.tool_calls
                ],
            },
        )

    def _on_group_done(self, index: int) -> None:
        # P1 / H7: one ExecGroup has just been committed, which is exactly the
        # transaction boundary the design mandates. This fires before the results
        # are backfilled, so it flushes the assistant reply; the tool events are
        # emitted later by _on_batch_done.
        self.persist_pending()

    def _on_descent(self, payload: str) -> None:
        """Record a descent's **leaves** (with schema) in the discovered-tools cache.

        The descent turn stays in the live message list (the model needs the
        children it just asked for), but :meth:`persist_pending` skips it so the
        stored transcript only contains real work.

        Each leaf is stored **together with its parameter schema**: a discovered
        tool must be immediately callable, otherwise the model would have to
        descend again merely to learn the arguments.

        Abstract children are ignored. They are locations in the tree, not
        capabilities, so caching them would consume slots that real tools need
        and would replay paths the model must still expand. Whether a node is a
        location or a capability is decided by ``kind`` alone.
        """
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return
        children = data.get("children")
        if not isinstance(children, list):
            return

        step = len(self.messages)
        parent = data.get("path")
        if isinstance(parent, str):
            # Remember the parent so it stays callable for further descents.
            self._expanded_paths.add(parent)

        for child in children:
            path = child.get("path")
            if not isinstance(path, str):
                continue
            if child.get("kind") != "leaf":
                # Intermediate nodes are not capabilities, so they never enter the
                # cache -- but they must be callable for the next descent.
                self._expanded_paths.add(path)
                continue
            self.discovered.upsert(
                ToolDiscovery(
                    path=path,
                    description=str(child.get("description") or ""),
                    kind="leaf",
                    first_seen_step=step,
                    last_used_step=step,
                    schema=dict(child.get("schema") or {}),
                )
            )

    def discovered_rows(self) -> list[dict[str, Any]]:
        """Inventory projection for the request's trailing temporary injection."""
        return self.discovered.rows()

    def mark_restored_descents(self) -> None:
        """Rebuild the descent-id set from restored history.

        The stored transcript keeps descent turns (the user's trace must show how
        a tool was found), but the model does not need them again: the tools they
        revealed are already in the inventory. Re-deriving the ids here lets the
        next request omit them, and it also lets :attr:`discovered` be repopulated
        so the model still sees those tools after a restart.
        """
        descents = self.tools.tree.abstract_paths()
        calls: dict[str, str] = {}
        for message in self.messages:
            if isinstance(message, AssistantMsg):
                for call in message.tool_calls:
                    if call.name in descents:
                        calls[call.id] = call.name

        for message in self.messages:
            if not isinstance(message, ToolMsg):
                continue
            parent = calls.get(message.tool_call_id)
            if parent is None:
                continue
            # Reuse the normal descent handler so the cache and the id set stay
            # consistent with the live path.
            self._on_descent(message.content)
            self._descent_call_ids.add(message.tool_call_id)
            self._expanded_paths.add(parent)

    def _on_batch_done(self, messages: list[Message]) -> None:
        """Announce the tool results of a completed batch to the trace panel.

        The conversation pane no longer renders tool cards, so this is the only
        surface where a call is visible. Results are paired with the arguments
        cached in :meth:`_on_assistant` (keyed by ``tool_call_id``), which works
        regardless of how far back the producing ``AssistantMsg`` now sits.
        """
        # Only results this turn produced may be emitted, otherwise earlier rounds
        # would be re-announced as tool events with empty arguments.
        for message in messages[self._traced_count :]:
            if not isinstance(message, ToolMsg):
                continue
            call = self._calls_by_id.pop(message.tool_call_id, None)
            self.emit(
                "tool",
                {
                    "path": message.name,
                    "preview": message.content[:400],
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                    "arguments": call.arguments if call is not None else "",
                },
            )
        self._traced_count = len(messages)
        self.persist_pending()

    def _on_inject(self, message: UserMsg, channel: str) -> None:
        self.emit("inject", {"channel": channel, "text": message.content})

    def _on_abort(self, reason: str) -> None:
        self.emit("state", {"state": "aborted", "detail": reason})

    def _on_summary(self, action: str) -> None:
        self.emit(
            "compress",
            {"action": action, "messages": len(self.messages)},
        )

    def _emit_usage(self) -> None:
        estimate = self.assembler.estimate(self.messages)
        self.emit(
            "usage",
            {
                "tokens": estimate,
                "capacity": self.capacity,
                "ratio": round(estimate / self.capacity, 4) if self.capacity else 0.0,
            },
        )

    # ----------------------------------------------------------- snapshot

    def snapshot(self, *, include_messages: bool = False) -> dict[str, Any]:
        """Return the session's current state (for REST queries)."""
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "state": self.state,
            "model": self.model,
            "capacity": self.capacity,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "tokens": self.assembler.estimate(self.messages),
            "seq": self._seq,
        }
        if include_messages:
            payload["messages"] = [
                _message_to_dict(m, self._message_ts(i))
                for i, m in enumerate(self.messages)
            ]
        return payload

    def _message_ts(self, index: int) -> float | None:
        """Return the timestamp of message ``index``; ``None`` when unknown.

        The session runner only keeps a lightweight "received message count ->
        time" map, so core message types need no timestamp field (which would
        break group-A structure definitions).
        """
        return self._timestamps.get(index)

def _message_to_dict(message: Message, ts: float | None = None) -> dict[str, Any]:
    """Project a message into a frontend-readable structure.

    Args:
        message: The message to project.
        ts: Message timestamp; omitted for historical messages without their own
            timestamp (the frontend shows no time then).
    """
    base: dict[str, Any] = {"ts": ts} if ts is not None else {}
    if isinstance(message, UserMsg):
        return {"role": "user", "content": message.content, **base}
    if isinstance(message, AssistantMsg):
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in message.tool_calls
            ],
            **base,
        }
    if isinstance(message, ToolMsg):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
            **base,
        }
    if isinstance(message, SystemMsg):
        return {"role": "system", "content": message.content, **base}
    raise TypeError(f"unknown message type: {type(message).__name__}")  # pragma: no cover


class SessionManager:
    """Multi-session management: create, query, delete and capacity control.

    Differs from the core ``SessionRegistry``: this class serves web requests and
    holds ``SessionRunner`` objects (with threads and event queues), so its
    lifetime follows the HTTP service.
    """

    def __init__(
        self,
        workdir: str,
        proxy: ReverseProxy,
        *,
        default_model: str,
        capacity: int = 32_768,
        max_sessions: int = 20,
    ) -> None:
        self.workdir = workdir
        self.proxy = proxy
        self.default_model = default_model
        self.capacity = capacity
        self.max_sessions = max_sessions
        self.tools = DemoToolRegistry(workdir)
        self.capabilities: CapabilityManifest | None = None
        self.capability_error: str | None = None
        #: Path of the working-memory file and whether this run created it (diagnostics).
        self.work_memory: Path | None = None
        self.work_memory_created = False
        self.work_memory_error: str | None = None
        self.load_capabilities(workdir)
        self._sessions: dict[str, SessionRunner] = {}
        self._lock = threading.Lock()
        # Core session registry (S4): web sessions are also covered by idle-timeout
        # and capacity eviction.
        # A construction failure must not block service startup; degrade to relying
        # on max_sessions alone.
        try:
            from harness.session import EvictionPolicy, SessionRegistry

            self.registry: Any = SessionRegistry(EvictionPolicy())
            self.registry.start_sweeper()
        except Exception:  # noqa: BLE001
            self.registry = None
        # SQLite persistence (P1): None means memory-only (e.g. read-only workdir).
        self.store = self._open_store()

    def _open_store(self) -> SessionStore | None:
        """Open the session database; ``None`` degrades to memory-only operation.

        Persistence is a durability enhancement, so an unusable database (read-only
        directory, locked file) must not prevent the service from starting.
        """
        try:
            store = SessionStore(self.workdir)
            store.list_sessions()  # triggers connection + schema creation
            return store
        except Exception:  # noqa: BLE001 - degrade instead of crashing
            return None

    # ------------------------------------------------------- restore on boot

    #: How many historical sessions to bring back into memory at startup.
    RESTORE_LIMIT = 10

    def restore(self) -> list[SessionRunner]:
        """Rebuild previously persisted sessions from SQLite.

        Only the most recent :attr:`RESTORE_LIMIT` sessions are resurrected, and
        none are started (no thread), so a restart is cheap. The persisted
        ``SystemMsg`` does not exist (H6); ``SessionRunner.__post_init__`` rebuilds
        it from the *current* tool tree, which is why restore step 4 assigns that
        job to the context layer.

        Returns:
            The runners that were restored, newest first.
        """
        if self.store is None:
            return []
        try:
            records = self.store.list_sessions()
        except Exception:  # noqa: BLE001 - a corrupt DB must not block startup
            return []

        restored: list[SessionRunner] = []
        for record in reversed(records):
            if len(restored) >= self.RESTORE_LIMIT:
                break
            if record.status == "ended":
                continue
            runner = self._restore_one(record)
            if runner is not None:
                restored.append(runner)
        return restored

    def _restore_one(self, record: SessionRecord) -> SessionRunner | None:
        """Rebuild a single ``SessionRunner`` from its persisted row and messages."""
        try:
            restored = self.store.restore(record.session_id) if self.store else None
            history = restored.messages if restored is not None else []
        except Exception:  # noqa: BLE001 - skip a session with a damaged history
            history = []

        try:
            runner = SessionRunner(
                session_id=record.session_id,
                workdir=self.workdir,
                proxy=self.proxy,
                tools=self.tools,
                model=self.default_model,
                capacity=self.capacity,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
        except Exception:  # noqa: BLE001
            return None

        if history:
            # Append after the freshly assembled SystemMsg (H6) and move the
            # persistence cursor past it, otherwise the next flush would duplicate
            # the whole history.
            runner.messages = [runner.system, *history]
            runner.persisted_count = len(runner.messages)
            # Re-identify descent calls (their tool results echo the intermediate
            # path), so the next model request can omit them even though the
            # stored transcript keeps them for the user's trace.
            runner.mark_restored_descents()
        # Restored history counts as already announced; the trace panel only shows
        # live events.
        runner._traced_count = len(runner.messages)
        runner.state = "idle"
        runner.store = self.store
        # A restored session never resumes by itself; it waits for the user.
        if self.registry is not None:
            core = Session(session_id=record.session_id, working_dir=self.workdir)
            runner.core_session = core
            self.registry.add(core)
        self._sessions[record.session_id] = runner
        return runner

    @property
    def compress_available(self) -> bool:
        """Whether C3 compression is available.

        The summarizer is provided by ``SessionRunner`` (a context-free single-turn
        call), so this is always True; the property is kept so ``/api/health`` has
        a stable observation point.
        """
        return True

    def load_capabilities(self, workdir: str) -> None:
        """Override the tool tree from the capability manifest; keep demo tools on failure.

        The manifest lives in the system data directory (``harness.paths``), with a
        project-local ``.agent/capabilities.json`` honoured when the repository
        commits one. It is an **operations surface**: a load failure must not
        prevent startup, so the error is only recorded in ``capability_error`` and
        exposed to the frontend by ``/api/health``.

        The working-memory file is created as a side effect, outside the working
        directory; failure is likewise only recorded and does not block startup.
        """
        try:
            path, created = ensure_working_memory(workdir)
            self.work_memory = path
            self.work_memory_created = created
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
            self.work_memory = None
            self.work_memory_created = False
            self.work_memory_error = str(exc)

        try:
            manifest = load_manifest(workdir)
        except CapabilityError as exc:
            self.capabilities = None
            self.capability_error = str(exc)
            return

        store = FlatStore()
        for entry in manifest.to_tooldefs(workdir):
            store.put(entry)
        self.tools = DemoToolRegistry(workdir)
        self.tools.store = store
        self.permissions = load_permissions(workdir, WEB_AGENT)
        self.tools.tree = build_tree(
            store,
            self.permissions,
            group_descriptions={
                g.group_id: g.description for g in manifest.groups
            },
        )
        self.capabilities = manifest
        self.capability_error = None

    def create(self, *, model: str | None = None, capacity: int | None = None) -> SessionRunner:
        """Create a new session."""
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                # Evict the least recently updated non-running session
                idle = [
                    s
                    for s in self._sessions.values()
                    if not s.running and s.state in {"idle", "done", "aborted", "error"}
                ]
                if not idle:
                    raise RuntimeError("会话数已达上限，且没有可淘汰的空闲会话")
                victim = min(idle, key=lambda s: s.updated_at)
                self._sessions.pop(victim.session_id, None)
                # P1: eviction from memory must not lose data -- the victim is
                # flushed and then left resumable in SQLite.
                victim.persist_pending()
                if self.store is not None:
                    with contextlib.suppress(Exception):
                        self.store.update_status(victim.session_id, "idle")
                if self.registry is not None:
                    self.registry.remove(victim.session_id)

            session_id = uuid.uuid4().hex[:12]
            runner = SessionRunner(
                session_id=session_id,
                workdir=self.workdir,
                proxy=self.proxy,
                tools=self.tools,
                model=model or self.default_model,
                capacity=capacity or self.capacity,
            )
            # P1: the sessions row must exist before any message row (the messages
            # table has a foreign key on it).
            runner.store = self.store
            if self.store is not None:
                try:
                    self.store.upsert_session(
                        SessionRecord(
                            session_id=session_id,
                            working_dir=runner.workdir,
                            created_at=runner.created_at,
                            updated_at=runner.updated_at,
                            status="idle",
                        )
                    )
                    runner.persisted_count = len(runner.messages)
                except Exception:  # noqa: BLE001 - keep the session usable anyway
                    runner.store = None
            # S4: register with the core registry so idle-timeout/capacity eviction
            # actually covers web sessions.
            if self.registry is not None:
                core = Session(session_id=session_id, working_dir=self.workdir)
                runner.core_session = core
                self.registry.add(core)
            self._sessions[session_id] = runner
            return runner

    def get(self, session_id: str) -> SessionRunner | None:
        """Look up a session by id."""
        return self._sessions.get(session_id)

    def list(self) -> list[SessionRunner]:
        """List all sessions (newest created first)."""
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def delete(self, session_id: str) -> bool:
        """Delete a session; a running session is aborted first.

        The row is marked ``ended`` rather than deleted (H13): the history stays on
        disk for audit, and ``restore()`` skips ended sessions so they do not come
        back after a restart.
        """
        with self._lock:
            runner = self._sessions.pop(session_id, None)
        if runner is None:
            return False
        if runner.running:
            runner.cancel()
        # P1: flush what is still in memory, then mark the session ended (H13).
        runner.persist_pending()
        if self.store is not None:
            # A failed status update must still let the in-memory delete succeed.
            with contextlib.suppress(Exception):
                self.store.update_status(session_id, "ended")
        # S4: remove the core session in sync so no ghost entry is left in the registry
        if self.registry is not None:
            self.registry.end(session_id)
        return True

    def models(self) -> list[str]:
        """Available models (from proxy config plus common alternatives)."""
        names = {p.model for p in self.proxy.providers if p.model}
        extras = {"deepseek-chat", "deepseek-reasoner"}
        return sorted(names | extras)
