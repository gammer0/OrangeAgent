"""会话运行器：把 AgentLoop 包装成可被 Web 层订阅的运行单元。

设计要点：

* **不修改核心**：``src/harness`` 保持零 Web 依赖；本模块只做适配。
* **事件流而非轮询**：每个会话持有一个事件队列，Web 层用 SSE 消费。
* **安全点插话**：外部发消息走 ``Inbox``，由 ``AgentLoop`` 在安全点消费，
  保证不打断未提交的工具组（I6）。
* **取消**：通过 ``Inbox.control`` 的 ``user_abort`` 实现，遵循 G1（停剩余组 + 补 3e）。

事件类型（供前端渲染）：

============ ==========================================================
kind         说明
============ ==========================================================
``state``    会话状态变化（idle / running / done / aborted / error）
``assistant``模型回复（content / tool_calls / is_final）
``tool``     工具执行结果（path / ok / error_type / duration_ms / 摘要）
``route``    Jev 路由决策（branch / fallback / confidence）
``compress`` C3 压缩发生（action / token 变化）
``inject``   用户插话（channel / 文本）
``usage``    token 占用与水位比例
``error``    错误（message）
============ ==========================================================
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from harness.context import ContextAssembler, SystemPromptParts
from harness.capabilities import CapabilityError, CapabilityManifest, load_manifest
from harness.executor import execute_batch
from harness.inbox import Inbox
from harness.jev import JevRouter
from harness.loop import AgentLoop, LoopHooks, StepController
from harness.message import AssistantMsg, Message, SystemMsg, ToolCall, ToolMsg, UserMsg
from harness.model import ReverseProxy
from harness.permission import AgentPermissions
from harness.permission_store import WEB_AGENT, load_permissions
from harness.tool_tree import EMPTY_SCHEMA, FlatStore, ToolDef, build_tree

__all__ = [
    "SessionEvent",
    "SessionRunner",
    "SessionManager",
    "DemoToolRegistry",
]

#: 事件队列上限；超出时丢弃最旧事件，避免慢客户端拖垮服务。
EVENT_QUEUE_MAX = 2_000


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """一条可推送给前端的事件。

    Attributes:
        seq: 会话内单调递增序号，便于前端去重与断线续传。
        ts: 事件时间戳。
        kind: 事件类型（见模块文档）。
        payload: 事件负载，必须可 JSON 序列化。
    """

    seq: int
    ts: float
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """导出为 JSON 友好的字典。"""
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "payload": self.payload,
        }


class DemoToolRegistry:
    """示例工具集：可读文件、列目录、写文件、回显。

    真实场景应由配置/插件提供；这里内置于服务层，避免污染核心包。
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

    # ---- 工具实现（全部限定在工作目录内）----

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

    # ---- 给 AgentLoop 用的解析器 ----

    def resolve(self, call: ToolCall) -> Any:
        """把 ToolCall 解析为可执行定义。"""
        return self.store.by_path(call.name)

    def tool_specs(self) -> list[dict[str, Any]]:
        """导出 OpenAI tools 声明，供模型请求使用。"""
        specs: list[dict[str, Any]] = []
        for entry in self.store.active_paths().values():
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
        return specs


@dataclass
class SessionRunner:
    """单个会话的运行时容器。

    Attributes:
        session_id: 会话标识。
        model: 当前选中的模型名（可被前端切换）。
        runner: 后台线程。
    """

    session_id: str
    workdir: str
    proxy: ReverseProxy
    tools: DemoToolRegistry
    model: str
    capacity: int = 32_768

    inbox: Inbox = field(default_factory=Inbox)
    messages: list[Message] = field(default_factory=list)
    events: queue.Queue[SessionEvent] = field(
        default_factory=lambda: queue.Queue(maxsize=EVENT_QUEUE_MAX)
    )
    state: str = "idle"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _cancelled: bool = False
    #: 消息索引 → 时间戳（供前端渲染历史时间，不污染核心消息类型）
    _timestamps: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.assembler = ContextAssembler(
            capacity=self.capacity,
            summarizer=self._summarize,
        )
        self.system = self.assembler.build_system(
            SystemPromptParts(
                tool_index=self.tools.tree.tree_index(depth=0),
                working_dir=self.workdir,
                goal=None,
            )
        )
        self.messages = [self.system]

    # ------------------------------------------------------------ 摘要

    #: 摘要请求的输入上限（字符）。超出时按消息类型分块压缩，见 :meth:`_summarize`。
    SUMMARY_INPUT_CHARS = 24_000

    #: 压缩规则（系统侧规定，模型不得自由发挥）。
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
        """无上下文单轮摘要：按系统规定的类型规则折叠历史。

        分块策略：输入超过 :attr:`SUMMARY_INPUT_CHARS` 时切成等长块，
        每块单独压缩，最后再把各块摘要合并一次。
        **不丢弃任何内容**——直接截断会静默丢失历史，这是压缩最不可接受的失败方式。

        与主循环共享 ``proxy``，但用独立 ``session_id`` 后缀，避免污染主会话的
        用量归集；**不传 tools**，模型只能回文本，不会再次触发工具调用。
        """
        from harness.message import SystemMsg, UserMsg

        # ponytail: 分块预算要扣掉头部标记的长度（如"[这是第 1/3 段...]"），
        # 否则拼接后会略超 SUMMARY_INPUT_CHARS。
        budget = self.SUMMARY_INPUT_CHARS - 80
        chunks = [
            text[start : start + budget] for start in range(0, len(text), budget)
        ] or [text]

        if len(chunks) == 1:
            return self._summarize_once(chunks[0])

        # 多块：逐块压缩，再把各块摘要合并（合并输入通常远小于原文）
        partials = [
            self._summarize_once(chunk, index=i + 1, total=len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        merged = "\n\n".join(partials)
        # 合并阶段也不硬截断：如果仍然超限，递归复用同一策略
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
        """执行单次摘要调用。"""
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

    # ------------------------------------------------------------ 事件

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        """追加一条事件（满队列时丢弃最旧）。"""
        with self._lock:
            self._seq += 1
            event = SessionEvent(self._seq, time.time(), kind, payload)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
                self.events.put_nowait(event)
            except queue.Empty:  # pragma: no cover - 竞争兜底
                pass
        self.updated_at = time.time()

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        self.emit("state", {"state": state, "detail": detail})

    # ------------------------------------------------------------ 运行

    @property
    def running(self) -> bool:
        """是否正在推理。"""
        return self._thread is not None and self._thread.is_alive()

    def start(self, text: str) -> bool:
        """开始一轮推理；已有轮次在跑时返回 ``False``。"""
        if self.running:
            return False
        self._cancelled = False
        self.messages.append(UserMsg(content=text))
        self._timestamps[len(self.messages) - 1] = time.time()
        # 注意：不在此处 emit inject —— AgentLoop 在安全点插入时通过
        # ``on_inject`` 钩子发出，避免前端收到重复的用户消息。
        self._thread = threading.Thread(target=self._run_once, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        """请求中止本轮（在下一个安全点生效）。"""
        from harness.inbox import InterruptRequest

        self._cancelled = True
        self.inbox.put_control(InterruptRequest(reason="user_abort", text=None))

    def _run_once(self) -> None:
        """执行一次完整的 ReAct 循环，并把过程翻译成事件。"""
        self._set_state("running")

        hooks = LoopHooks(
            on_assistant=self._on_assistant,
            on_group_done=self._on_group_done,
            on_inject=self._on_inject,
            on_abort=self._on_abort,
            on_summary=self._on_summary,
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

        try:
            # 传完整历史（含刚追加的 UserMsg），AgentLoop 会直接以它为工作副本
            result = loop.run(self.messages)
            self.messages = list(result.messages)
            # 新产生的消息统一打上当前时间（近似，不影响语义）
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
        except Exception as exc:  # noqa: BLE001 - 会话隔离：单会话出错不影响服务
            self.emit("error", {"message": str(exc)})
            self._set_state("error", str(exc))

    def _model_client(self) -> Any:
        """把 ReverseProxy 适配成编排层的 ``chat(messages)`` 协议。"""
        session_id = self.session_id

        class _Client:
            def __init__(self, owner: SessionRunner) -> None:
                self.owner = owner

            def chat(self, messages: list[Message]) -> AssistantMsg:
                response = self.owner.proxy.chat(
                    messages,
                    session_id=session_id,
                    tools=self.owner.tools.tool_specs(),
                    model=self.owner.model,
                )
                return response.message

        return _Client(self)

    def _router_chat(self, messages: list[Message]) -> AssistantMsg:
        """Jev 路由用的单轮调用（不复用主会话 id 以免污染用量归集）。"""
        response = self.proxy.chat(
            messages,
            session_id=f"{self.session_id}:route",
            model=self.model,
        )
        return response.message

    def _branches(self) -> list[str]:
        """候选一级分支：L1 + L2 的 abstract 节点。"""
        return [p for p in self.tools.tree.paths if p.count("/") <= 2]

    def _executor(
        self,
        plan: Any,
        resolver: Any,
        on_group_done: Any = None,
        **kw: Any,
    ) -> dict[str, Any]:
        """把核心执行器包装一层，用于统计耗时与产出事件。"""
        return execute_batch(plan, resolver, on_group_done, **kw)

    # ------------------------------------------------------------ 钩子

    def _on_assistant(self, message: AssistantMsg) -> None:
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
        for message in self.messages[-8:]:
            if isinstance(message, ToolMsg):
                self.emit(
                    "tool",
                    {
                        "group": index,
                        "path": message.name,
                        "preview": message.content[:400],
                    },
                )

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

    # ------------------------------------------------------------ 快照

    def snapshot(self, *, include_messages: bool = False) -> dict[str, Any]:
        """返回会话的当前状态（供 REST 查询）。"""
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
        """返回第 ``index`` 条消息的时间戳；未知时返回 ``None``。

        会话运行器只记录「收到的消息数 → 时间」的轻量映射，
        避免给核心消息类型增加时间戳字段（会破坏 A 组结构定义）。
        """
        return self._timestamps.get(index)

def _message_to_dict(message: Message, ts: float | None = None) -> dict[str, Any]:
    """把消息投影为前端可读结构。

    Args:
        message: 待投影消息。
        ts: 消息时间戳；历史消息无独立时间戳时省略（前端不显示时间）。
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
    raise TypeError(f"未知消息类型: {type(message).__name__}")  # pragma: no cover


class SessionManager:
    """多会话管理：创建、查询、删除与容量控制。

    与核心 ``SessionRegistry`` 的区别：本类面向 Web 请求，
    持有 ``SessionRunner``（含线程与事件队列），生命周期跟随 HTTP 服务。
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
        self.load_capabilities(workdir)
        self._sessions: dict[str, SessionRunner] = {}
        self._lock = threading.Lock()

    @property
    def compress_available(self) -> bool:
        """C3 压缩是否可用。

        摘要器由 ``SessionRunner`` 提供（无上下文单轮调用），因此恒为 True；
        保留该属性是为了让 ``/api/health`` 有一个稳定的观测点。
        """
        return True

    def load_capabilities(self, workdir: str) -> None:
        """按 ``.agent/capabilities.json`` 覆盖工具树；失败时保留演示工具。

        清单是**运维面**：装载失败不应让服务起不来，因此错误只记录在
        ``capability_error``，由 ``/api/health`` 暴露给前端。
        """
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
        """创建新会话。"""
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                # 淘汰最久未更新的非运行会话
                idle = [
                    s
                    for s in self._sessions.values()
                    if not s.running and s.state in {"idle", "done", "aborted", "error"}
                ]
                if not idle:
                    raise RuntimeError("会话数已达上限，且没有可淘汰的空闲会话")
                victim = min(idle, key=lambda s: s.updated_at)
                self._sessions.pop(victim.session_id, None)

            session_id = uuid.uuid4().hex[:12]
            runner = SessionRunner(
                session_id=session_id,
                workdir=self.workdir,
                proxy=self.proxy,
                tools=self.tools,
                model=model or self.default_model,
                capacity=capacity or self.capacity,
            )
            self._sessions[session_id] = runner
            return runner

    def get(self, session_id: str) -> SessionRunner | None:
        """按 id 取会话。"""
        return self._sessions.get(session_id)

    def list(self) -> list[SessionRunner]:
        """列出全部会话（按创建时间倒序）。"""
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def delete(self, session_id: str) -> bool:
        """删除会话；运行中的会话先请求中止。"""
        with self._lock:
            runner = self._sessions.pop(session_id, None)
        if runner is None:
            return False
        if runner.running:
            runner.cancel()
        return True

    def models(self) -> list[str]:
        """可用模型列表（来自代理配置 + 常见备选）。"""
        names = {p.model for p in self.proxy.providers if p.model}
        extras = {"deepseek-chat", "deepseek-reasoner"}
        return sorted(names | extras)
