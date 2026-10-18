"""上下文装配器（C1/C2/C3 的编排者，``SystemMsg`` 五段装配）。

对应文档：

* docs/02-数据结构设计/03-消息.md §3.3（五段组成、A0a–A0e）
* docs/01-系统架构设计.md §4.3（水位与压缩）、§4.5.1（五段来源）

职责边界：

* **是**唯一构造 ``SystemMsg`` 的地方——五段顺序、分隔符、逐字节稳定性都在这里收口。
* **是** C1/C2/C3 的调用方：装配前先判定水位，必要时压缩，再投影给 provider。
* **不是** 工具执行器，**不是** 会话存储。

关键不变量：

* **A0a / I1**：``SystemMsg.content`` 在会话内逐字节不变。因此装配一次后必须缓存，
  后续 ``assemble`` 直接复用，不重新拼接。
* **A0b**：工作记忆正文不进 ``SystemMsg``，第 1 段只说明读写模式。
* **A0c**：第 3 段只列长期记忆文件名，不列路径与正文。
* **A0d**：第 5 段放末尾，且不含 ``session_id`` 等系统标识。
* **I5**：每个水位只触发一次（由 ``Watermark`` 记账）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from harness.compressor import compress
from harness.discovery import TreeSessionState
from harness.message import AssistantMsg, Message, SystemMsg, ToolMsg, UserMsg
from harness.watermark import JudgeResult, Watermark, estimate_messages_tokens, judge

__all__ = [
    "SEGMENT_SEPARATOR",
    "LONG_TERM_MEMORY_NAME",
    "SystemPromptParts",
    "AssembleResult",
    "ContextAssembler",
    "DEFAULT_SYSTEM_REQUIREMENTS",
    "DEFAULT_ROLE_GUIDE",
]

#: A0a：五段之间的固定分隔符。
SEGMENT_SEPARATOR = "\n\n"

#: A0c：长期记忆文件名（全系统唯一，不含路径与正文）。
LONG_TERM_MEMORY_NAME = "MEMORY.md"

#: 第 1 段：系统要求（代码内常量）。
DEFAULT_SYSTEM_REQUIREMENTS = """你是运行在 Harness 平台上的智能体。

## 响应格式
- 需要调用工具时，通过 tool_calls 发起调用，一次可发起多个。
- 所有工具调用都必须给出完整绝对路径（以 / 开头），不得使用缩写或相对路径。
- 不再需要工具时，直接给出最终答案文本，不要输出空的 tool_calls。

## 工作记忆
- 本项目存在一个共享工作记忆文件 .agent/WORK.md，同工作目录的多个会话共享它。
- 它不是系统提示的一部分：需要回顾时用文件工具读取，需要沉淀时用文件工具写入。
- 读写结果都会以工具结果的形式返回给你。""".strip()

#: 第 4 段：角色与行为准则（代码内常量）。
DEFAULT_ROLE_GUIDE = """## 角色
你是一名严谨的工程助手：先理解目标，再动手；不确定时先查证，不臆测。

## 行为准则
- 只调用与当前目标相关的工具，避免无意义的试探。
- 工具失败时先读错误信息，再决定是否换工具或换参数，不要原地重复同样的调用。
- 长输出先摘要再引用，不要把大段原始内容复述给用户。
- 不隐瞒不确定性：做不到、没验证、有风险，都要明说。""".strip()

#: 摘要器签名：``(text) -> summary``。由调用方注入（复用主模型通道）。
Summarizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class SystemPromptParts:
    """``SystemMsg`` 五段的输入素材。

    Attributes:
        tool_index: 第 2 段，``ToolTree.tree_index()`` 的产物；为空时该段留空。
        memory_name: 第 3 段，长期记忆文件名。
        working_dir: 第 5 段，工作目录绝对路径。
        work_memory_relative_path: 第 5 段，工作记忆相对路径。
        goal: 第 5 段，可选目标。
        system_requirements: 第 1 段，默认取模块常量。
        role_guide: 第 4 段，默认取模块常量。
    """

    tool_index: str = ""
    memory_name: str = LONG_TERM_MEMORY_NAME
    working_dir: str = ""
    work_memory_relative_path: str = ".agent/WORK.md"
    goal: str | None = None
    system_requirements: str = DEFAULT_SYSTEM_REQUIREMENTS
    role_guide: str = DEFAULT_ROLE_GUIDE

    def session_context(self) -> str:
        """第 5 段：会话上下文（工作目录 + 工作记忆相对路径 + 可选目标）。

        A0d：不得出现 ``session_id`` / 租户 / 创建时间等系统标识。
        """
        lines = [
            f"工作目录: {self.working_dir or '(未指定)'}",
            f"工作记忆: {self.work_memory_relative_path}",
        ]
        if self.goal:
            lines.append(f"本次目标: {self.goal}")
        return "\n".join(lines)

    def render(self) -> str:
        """按 A0a 的固定顺序与分隔符拼接五段。

        空的第 2 / 3 段会以空字符串占位，保证**段数恒定**——
        这样无论工具树是否为空，第 5 段的位置都不变，前缀稳定性不被破坏。
        """
        segments = [
            self.system_requirements,
            self.tool_index,
            self.memory_name,
            self.role_guide,
            self.session_context(),
        ]
        return SEGMENT_SEPARATOR.join(segments)


@dataclass
class AssembleResult:
    """一次装配的结果。

    Attributes:
        messages: 可直接发送给模型的消息列表（已压缩、已通过水位判定）。
        system: 会话级 ``SystemMsg``（缓存实例，后续直接复用）。
        judged: 本次水位判定结果；``None`` 表示未判定（如缓存命中且无需判定）。
        compressed: 是否发生压缩。
        action: 水位动作（``normal`` / ``compress_1`` / ``compress_2`` / ``session_end``）。
    """

    messages: list[Message]
    system: SystemMsg
    judged: JudgeResult | None
    compressed: bool
    action: str


@dataclass
class ContextAssembler:
    """扁平 messages 装配 + C1/C2/C3 编排。

    Attributes:
        capacity: 模型上下文上限（token）。
        watermark: 会话级水位记录；由调用方随 ``Session`` 持有。
        summarizer: C3 使用的摘要函数；未注入时压缩会被跳过（保守退化，I4）。
        keep_records_50: 50% 水位保留的 CallRecord 数。
        keep_records_75: 75% 水位保留的 CallRecord 数。
        enable_auto_compress: 关闭后只做水位判定不做压缩（便于测试与降级）。
    """

    capacity: int
    watermark: Watermark | None = None
    summarizer: Summarizer | None = None
    keep_records_50: int | None = None
    keep_records_75: int | None = None
    enable_auto_compress: bool = True
    _system: SystemMsg | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity 必须为正数，实际为 {self.capacity}")
        if self.watermark is None:
            self.watermark = Watermark(capacity=self.capacity)

    # ------------------------------------------------------------ SystemMsg

    def build_system(self, parts: SystemPromptParts) -> SystemMsg:
        """装配 ``SystemMsg``（A0a：只装配一次，之后逐字节复用）。

        若已缓存则直接返回缓存实例——调用方即使在工具树变化后再次调用，
        也不会改变会话前缀（I1）。
        """
        if self._system is None:
            self._system = SystemMsg(content=parts.render())
        return self._system

    @property
    def system(self) -> SystemMsg | None:
        """已缓存的 ``SystemMsg``；尚未装配时为 ``None``。"""
        return self._system

    @property
    def compress_available(self) -> bool:
        """C3 压缩当前是否真的可用。

        ponytail: 没有摘要器时压缩会被静默跳过（I4 保守退化），
        运维面需要这个信号才能发现"压缩从未发生"。
        """
        return self.enable_auto_compress and self.summarizer is not None

    def reset_system(self) -> None:
        """丢弃缓存的 ``SystemMsg``。

        仅用于**新会话**（``Session`` 重建）。会话内调用会破坏 I1，
        因此不作为常规接口暴露给编排层使用。
        """
        self._system = None

    # ------------------------------------------------------------ 装配

    def assemble(
        self,
        messages: Sequence[Message],
        *,
        clear_trace_state: Callable[[], None] | None = None,
    ) -> AssembleResult:
        """判定水位 → 必要时压缩 → 返回可发送的消息列表。

        Args:
            messages: 会话历史（含 ``SystemMsg``）。
            clear_trace_state: 压缩发生后回调，用于清空
                ``TreeQueryState.delivered`` 与 ``DiscoveredTools``（B14 / 文档 §4.6）。

        Returns:
            :class:`AssembleResult`；``action == "session_end"`` 时**不压缩**，
            调用方应触发收尾（写轨迹 + 结算记忆 + 提示开新会话）。
        """
        history = list(messages)
        if not history:
            raise ValueError("messages 不得为空；请先装配 SystemMsg")

        judged = judge(history, self.capacity, watermark=self.watermark)
        action = judged.action

        if action == "session_end":
            # I5：≥85% 只收尾，不压缩
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        if action == "normal" or not self.enable_auto_compress or self.summarizer is None:
            # I4：优化失败必须无损退化——没有摘要器就按朴素实现继续
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        level = 50 if action == "compress_1" else 75
        # ponytail: 两个 keep 字段默认 None，传 None 给 compress 才会走
        # default_keep_records()；显式算默认值会导致"保留全部记录、无可压缩"。
        keep = self.keep_records_50 if level == 50 else self.keep_records_75

        try:
            compressed_messages = compress(
                history,
                level,
                self.summarizer,
                keep_records=keep,
                watermark=self.watermark,
            )
        except ValueError as exc:
            # 水位已记账或记录不足以切分：无损退化，不阻断主流程（I4）
            # ponytail: 保留原因，避免"压缩静默失败"再次不可观测。
            self.last_compress_error = str(exc)
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        if clear_trace_state is not None:
            clear_trace_state()

        # ponytail: compress() 在"记录不足以切分"时会原样返回同一个列表，
        # 此时不能报 compressed=True——否则上层以为压了，实际没压。
        actually_compressed = compressed_messages is not history
        return AssembleResult(
            messages=compressed_messages,
            system=_system_of(compressed_messages),
            judged=judged,
            compressed=actually_compressed,
            action=action,
        )

    # ------------------------------------------------------------ 便捷方法

    def estimate(self, messages: Sequence[Message]) -> int:
        """估算消息列表的 token 数（C2）。"""
        return estimate_messages_tokens(list(messages))

    def ratio(self, messages: Sequence[Message]) -> float:
        """当前占用比例，供界面展示。"""
        return self.estimate(messages) / self.capacity


def _system_of(messages: Sequence[Message]) -> SystemMsg:
    """取 ``messages[0]`` 并校验其为 ``SystemMsg``（A1）。"""
    head = messages[0]
    if not isinstance(head, SystemMsg):
        raise ValueError(f"A1: messages[0] 必须是 SystemMsg，实际为 {type(head).__name__}")
    return head


def clear_after_compress(state: TreeSessionState) -> Callable[[], None]:
    """构造压缩后清空会话级树查询状态的回调（B14 / 文档 §4.6）。

    Args:
        state: 会话级 ``TreeQueryState`` + ``DiscoveredTools`` 门面。

    Returns:
        无参回调；压缩发生后调用它即可同步清空两处缓存。
    """

    def _clear() -> None:
        state.reset()

    return _clear


def has_unclosed_tool_calls(messages: Sequence[Message]) -> bool:
    """是否存在未配对的 ``tool_calls``（I2 / I6 的前置检查）。"""
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMsg):
            pending.update(call.id for call in msg.tool_calls)
        elif isinstance(msg, ToolMsg):
            pending.discard(msg.tool_call_id)
    return bool(pending)


def last_user_text(messages: Sequence[Message]) -> str:
    """取最近一条 ``UserMsg`` 的文本（Jev 路由的 query 来源）。"""
    for msg in reversed(messages):
        if isinstance(msg, UserMsg):
            return msg.content
    return ""
