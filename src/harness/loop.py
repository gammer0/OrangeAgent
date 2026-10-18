"""编排层：ReAct 状态机与步数闸门。

对应文档：docs/01-系统架构设计.md §4.2、§4.6.1、§4.9

本模块只负责**控制流**，不执行工具、不直接构造 ``messages``：

* :class:`StepController` —— 步数闸门、注入计数、终止判定（含死锁判据）。
* :class:`AgentLoop` —— 把「LLM 调用 → 工具批次 → 安全点插话 → 下一步」串起来，
  依赖全部通过构造参数注入，便于用假对象做单元测试。

安全点语义（G1–G4 / D7 / D8）：

========================  =================  ====================
安全点                       chat                control
========================  =================  ====================
① LLM 返回后                全部 drain          取首个并清空
② 组间（仍有后续组）         **不动**            取首个并清空 → 停剩余组 + 补 3e
③ 全批完成                  全部 drain          取首个并清空
========================  =================  ====================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.exceptions import MessageInvariantError
from harness.execution import BatchPlan, ExecResult, fill_placeholders, plan_batch
from harness.executor import Committed
from harness.inbox import Inbox, InterruptRequest
from harness.jev import JevRouter
from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.routing import RouteDecision

__all__ = [
    "DeadlockTracker",
    "StepController",
    "AgentLoop",
    "LoopHooks",
    "LoopResult",
    "MAX_STEPS",
    "MAX_INJECTS",
]


#: 设计文档配置项 ``MAX_STEPS``。
#: ponytail: 20 不够用——实测多步工程任务（改代码 + 跑测试 + 修错）常在收尾前耗尽。
MAX_STEPS = 40
#: 设计文档配置项 ``MAX_INJECTS``。
MAX_INJECTS = 10
#: 连续失败次数达到该值且未换工具路径即判定死锁（文档 §4.9）。
DEADLOCK_THRESHOLD = 3


class ModelClient(Protocol):
    """模型调用协议：编排层只依赖这一个方法。"""

    def chat(self, messages: Sequence[Message]) -> AssistantMsg:
        """返回下一轮 ``AssistantMsg``（含 ``tool_calls`` 或最终文本）。"""
        ...


class ToolResolver(Protocol):
    """工具解析协议：``ToolCall → 可执行定义``。"""

    def __call__(self, call: ToolCall) -> Any:
        """返回带 ``fn`` / ``concurrency`` / ``idempotent`` 的对象。"""
        ...


class BatchExecutor(Protocol):
    """批次执行协议：与 :func:`harness.executor.execute_batch` 同签名。"""

    def __call__(
        self,
        plan: BatchPlan,
        resolver: Callable[[ToolCall], Any],
        on_group_done: Callable[[int], Any] | None = None,
        *,
        max_workers: int = 4,
    ) -> dict[str, ExecResult]:
        ...


@dataclass
class DeadlockTracker:
    """死锁判据：连续 3 次失败 + 结果 hash 未变 + 未换工具路径。

    三条同时满足才判死锁；任一条被打破即重置计数。
    """

    threshold: int = DEADLOCK_THRESHOLD
    _count: int = 0
    _last_hash: str = ""
    _last_path: str = ""

    def record(self, *, ok: bool, content: str, path: str) -> bool:
        """记录一次调用结果，返回是否判定死锁。"""
        if ok:
            self.reset()
            return False

        digest = f"{path}|{hash(content)}"
        if digest == self._last_hash and path == self._last_path:
            self._count += 1
        else:
            # 换了工具路径或结果变化：重新计数
            self._count = 1
            self._last_hash = digest
            self._last_path = path
        return self._count >= self.threshold

    def reset(self) -> None:
        """重置计数（成功调用、换工具、新批次时调用）。"""
        self._count = 0
        self._last_hash = ""
        self._last_path = ""


class StepController:
    """步数闸门、注入计数、终止判定（§4.2）。

    三个终止条件相互独立，任一满足即要求结束本轮会话：

    * 步数达到 ``MAX_STEPS``；
    * 注入次数达到 ``MAX_INJECTS``；
    * 死锁判据命中。
    """

    def __init__(self, max_steps: int = MAX_STEPS, max_injects: int = MAX_INJECTS) -> None:
        self.max_steps = max_steps
        self.max_injects = max_injects
        self.step = 0
        self.injects = 0
        self.deadlock = DeadlockTracker()
        self._stop_reason: str | None = None

    # ---------------------------------------------------------- 闸门

    def begin_step(self) -> bool:
        """开始新一步；返回 ``False`` 表示已达步数上限，不得继续。"""
        if self.step >= self.max_steps:
            self._stop_reason = self._stop_reason or "max_steps"
            return False
        self.step += 1
        return True

    def can_inject(self) -> bool:
        """本步是否还允许注入用户消息。"""
        return self.injects < self.max_injects

    def record_inject(self) -> bool:
        """登记一次注入；返回 ``False`` 表示本次注入后达到上限。"""
        self.injects += 1
        if self.injects >= self.max_injects:
            self._stop_reason = self._stop_reason or "max_injects"
            return False
        return True

    def record_tool_result(self, *, ok: bool, content: str, path: str) -> bool:
        """登记一次工具结果，返回是否命中死锁判据。"""
        if self.deadlock.record(ok=ok, content=content, path=path):
            self._stop_reason = self._stop_reason or "deadlock"
            return True
        return False

    # ---------------------------------------------------------- 终止

    @property
    def stop_reason(self) -> str | None:
        """本会话的终止原因：``max_steps`` / ``max_injects`` / ``deadlock`` / ``None``。"""
        return self._stop_reason

    @property
    def should_stop(self) -> bool:
        """是否应当结束会话。"""
        return self._stop_reason is not None

    def mark_stop(self, reason: str) -> None:
        """外部条件（上下文收尾等）要求结束时登记原因。"""
        self._stop_reason = self._stop_reason or reason


@dataclass
class LoopHooks:
    """编排过程中的可选回调，全部默认为空操作。

    Attributes:
        on_assistant: 收到模型回复后调用（用于落库 / 写轨迹）。
        on_group_done: 每个 ``ExecGroup`` 完成并回填后调用。
        on_inject: 每注入一条用户消息后调用。
        on_abort: 发生中止（control 信号）时调用，参数为原因。
        on_summary: 上下文压缩触发时调用。
    """

    on_assistant: Callable[[AssistantMsg], None] | None = None
    on_group_done: Callable[[int], None] | None = None
    on_inject: Callable[[UserMsg, str], None] | None = None
    on_abort: Callable[[str], None] | None = None
    on_summary: Callable[[str], None] | None = None


@dataclass
class LoopResult:
    """一轮会话的最终结果。

    Attributes:
        messages: 会话消息（未持久化，由调用方决定落库）。
        final_text: 最终答案文本；因中止结束时为 ``None``。
        steps: 实际推进的步数。
        stop_reason: 终止原因。
        aborted: 是否因 ``control`` 信号中止。
    """

    messages: list[Message]
    final_text: str | None
    steps: int
    stop_reason: str | None
    aborted: bool = False
    trace: list[str] = field(default_factory=list)
    route: RouteDecision | None = None


class AgentLoop:
    """ReAct 主循环（§4.2）。

    职责边界：

    * **是**控制流唯一持有者：决定何时调模型、何时执行工具、何时检查安全点。
    * **不是**工具执行器：批次执行委托给注入的 ``executor``。
    * **不是**消息构造器：``SystemMsg`` 与投影由上下文层负责，本类只追加
      ``UserMsg`` / ``AssistantMsg`` / ``ToolMsg``。

    构造参数全部注入，因此单元测试可用假模型 / 假执行器完整覆盖安全点语义。
    """

    def __init__(
        self,
        *,
        model: ModelClient,
        inbox: Inbox,
        executor: BatchExecutor,
        resolver: Callable[[ToolCall], Any],
        system_msg: SystemMsg,
        controller: StepController | None = None,
        hooks: LoopHooks | None = None,
        max_workers: int = 4,
        assembler: Any | None = None,
        router: JevRouter | None = None,
        branch_provider: Callable[[], Sequence[str]] | None = None,
        tool_count_provider: Callable[[], int] | None = None,
        history_tokens_provider: Callable[[int], int] | None = None,
    ) -> None:
        self.model = model
        self.inbox = inbox
        self.executor = executor
        self.resolver = resolver
        self.system_msg = system_msg
        self.controller = controller or StepController()
        self.hooks = hooks or LoopHooks()
        self.max_workers = max_workers
        #: 可选上下文装配器；不传则按朴素实现直接调用模型（I4 无损退化）。
        self.assembler = assembler
        #: 可选 Jev 路由器；不传则不路由（等价全量暴露）。
        self.router = router
        #: 候选一级分支获取器；与 ``router`` 配对使用。
        self.branch_provider = branch_provider
        #: 工具总数获取器，用于路由启用条件判定。
        self.tool_count_provider = tool_count_provider
        #: 历史 token 获取器，用于路由启用条件判定。
        self.history_tokens_provider = history_tokens_provider
        #: 最近一次路由决策（便于外部写轨迹）。
        self.last_route: RouteDecision | None = None

    # ------------------------------------------------------------ 主循环

    def run(self, history: Sequence[Message] | None = None) -> LoopResult:
        """运行到最终答案、步数耗尽或中止。

        Args:
            history: 既有会话消息；``None`` 表示新会话（自动装配 ``SystemMsg``）。

        Returns:
            含最终文本与终止原因的 :class:`LoopResult`。
        """
        messages: list[Message] = list(history) if history else [self.system_msg]
        if not isinstance(messages[0], SystemMsg):
            raise MessageInvariantError("A1", "messages[0] 必须是 SystemMsg")

        aborted = False
        final_text: str | None = None

        while self.controller.begin_step():
            # ---- 安全点①：LLM 返回前先消费上一轮遗留的插话 ----
            if not self._drain_at_safe_point(messages, "safe_point_1"):
                aborted = True
                break

            validate_messages(messages)

            outgoing = self._assemble(messages)
            assistant = self.model.chat(outgoing)
            messages.append(assistant)
            if self.hooks.on_assistant is not None:
                self.hooks.on_assistant(assistant)

            if assistant.is_final:
                final_text = assistant.content
                break

            # ---- Jev 路由（优化路径，失败即全量，I4）----
            self._maybe_route(messages)

            # ---- 工具批次 ----
            plan = plan_batch(list(assistant.tool_calls), self._concurrency_of)
            stop_after = self._execute(plan, messages)

            if stop_after is not None:
                # D7 / G1：停止剩余组后补占位，再按 G3 插入附带文本
                self._fill_and_append(messages, plan, stop_after)
                aborted = True
                if self.hooks.on_abort is not None:
                    self.hooks.on_abort(stop_after.reason)
                break

            # ---- 安全点③：全批完成，两通道都可消费 ----
            if not self._drain_at_safe_point(messages, "safe_point_3"):
                aborted = True
                break

            if self.controller.should_stop:
                break

        if self.controller.stop_reason is not None and self.controller.stop_reason != "max_steps":
            aborted = aborted or self.controller.stop_reason == "deadlock"

        return LoopResult(
            messages=messages,
            final_text=final_text,
            steps=self.controller.step,
            stop_reason=self.controller.stop_reason,
            aborted=aborted,
            route=self.last_route,
        )

    # ------------------------------------------------------------ 上下文装配

    def _assemble(self, messages: list[Message]) -> list[Message]:
        """装配待发送消息（C1/C2/C3）。

        未注入 ``assembler`` 时直接返回原列表——优化缺失必须无损退化（I4）。
        水位达到 ``session_end`` 时登记收尾原因，本轮不再继续推进。
        """
        if self.assembler is None:
            return messages

        result = self.assembler.assemble(messages)
        if result.compressed:
            # 压缩会重建列表，必须用新列表继续，否则后续追加会写到旧对象
            messages[:] = result.messages
            if self.hooks.on_summary is not None:
                self.hooks.on_summary(result.action)

        if result.action == "session_end":
            self.controller.mark_stop("session_end")

        return result.messages

    # ------------------------------------------------------------ 路由

    def _maybe_route(self, messages: list[Message]) -> None:
        """按启用条件执行一次 Jev 路由。

        ``state`` 只放 query（最后一条 UserMsg），不放任何工具输出；
        任何异常都收敛为回退决策（I4）。
        """
        if self.router is None or self.branch_provider is None:
            return

        branches = list(self.branch_provider())
        if not branches:
            return

        query = _last_user_text(messages)
        history_tokens = (
            self.history_tokens_provider(len(messages))
            if self.history_tokens_provider is not None
            else 0
        )
        tool_count = self.tool_count_provider() if self.tool_count_provider is not None else 0

        self.last_route = self.router.route(
            query,
            branches,
            history_tokens=history_tokens,
            tool_count=tool_count,
        )

    # ------------------------------------------------------------ 批次执行

    def _concurrency_of(self, call: ToolCall) -> str:
        """从 ``ToolDef`` 读取并发性；解析失败时保守取 ``exclusive``（I4）。"""
        try:
            tool = self.resolver(call)
        except Exception:
            return "exclusive"
        return getattr(tool, "concurrency", "exclusive")

    def _execute(
        self,
        plan: BatchPlan,
        messages: list[Message],
    ) -> InterruptRequest | None:
        """执行整批；返回中断信号表示需要停止剩余组。

        组完成回调实现安全点②：只消费 ``control``，**不消费** ``chat``（G2）。
        """
        interrupt: InterruptRequest | None = None

        def on_group_done(report: Any) -> Any:
            nonlocal interrupt
            if self.hooks.on_group_done is not None:
                self.hooks.on_group_done(getattr(report, "index", 0))
            signal = self.inbox.take_control_at_sub_batch()
            if signal is not None:
                interrupt = signal
                return Committed(reason=signal.reason)
            return None

        results = self.executor(
            plan,
            self.resolver,
            on_group_done,
            max_workers=self.max_workers,
        )

        # D3：按组序 + 组内序回填，绝不按完成顺序
        for group in plan.groups:
            for call in group.calls:
                result = results.get(call.id)
                if result is None:
                    continue
                messages.append(
                    ToolMsg(
                        tool_call_id=call.id,
                        name=call.name,
                        content=result.content,
                    )
                )
                if self.controller.record_tool_result(
                    ok=result.ok, content=result.content, path=call.name
                ):
                    # 死锁判据命中：不再继续推进
                    self.controller.mark_stop("deadlock")

        return interrupt

    def _fill_and_append(
        self,
        messages: list[Message],
        plan: BatchPlan,
        signal: InterruptRequest,
    ) -> None:
        """G1 / D7：补 3e 占位；G3：``text`` 非空时再插入 ``UserMsg``。"""
        filled = {
            msg.tool_call_id
            for msg in messages
            if isinstance(msg, ToolMsg) and msg.tool_call_id
        }
        placeholders = fill_placeholders(plan, filled, signal.reason)
        messages.extend(placeholders)

        if signal.text:
            self._inject_user_msg(messages, signal.text, channel="control")

    # ------------------------------------------------------------ 安全点

    def _drain_at_safe_point(self, messages: list[Message], label: str) -> bool:
        """完整安全点：先 control 后 chat（G4）。

        Returns:
            ``True`` 表示可继续；``False`` 表示收到 ``control``，本轮应结束。
        """
        signal, chats = self.inbox.drain_at_full_safe_point()
        if signal is not None:
            # ① / ③ 处无剩余组可停，直接作为中止原因
            return False

        for item in chats:
            self._inject_user_msg(messages, item.text, channel="chat")
        del label
        return True

    def _inject_user_msg(self, messages: list[Message], text: str, channel: str) -> None:
        """插入一条 ``UserMsg`` 并登记注入次数。

        未闭合 ``tool_calls`` 序列内部禁止插入（D8 / I6）——安全点保证调用方
        只在合法位置调用本方法；此处做一次防御性校验。
        """
        if _has_unclosed_tool_calls(messages):
            raise MessageInvariantError(
                "I6", f"{channel} 通道消息不得插入未闭合的 tool_calls 序列"
            )

        msg = UserMsg(content=text)
        messages.append(msg)
        if self.hooks.on_inject is not None:
            self.hooks.on_inject(msg, channel)
        if not self.controller.record_inject():
            self.controller.mark_stop("max_injects")


def _has_unclosed_tool_calls(messages: Sequence[Message]) -> bool:
    """是否存在未配对的 ``tool_calls``。"""
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMsg):
            pending.update(call.id for call in msg.tool_calls)
        elif isinstance(msg, ToolMsg):
            pending.discard(msg.tool_call_id)
    return bool(pending)


def _last_user_text(messages: Sequence[Message]) -> str:
    """取最近一条 ``UserMsg`` 的文本，作为路由 query。

    文档 §4.4：``state`` 只放 query，不放工具输出（防 JPDH）。
    """
    for msg in reversed(messages):
        if isinstance(msg, UserMsg):
            return msg.content
    return ""
