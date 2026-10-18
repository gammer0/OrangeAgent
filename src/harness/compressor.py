"""C3 GlobalCompressor：按 CallRecord 边界折叠历史。

对应文档：docs/01-系统架构设计.md §4.3（C3 状态机与三条硬约束）

三条硬约束：

1. **边界只落在 CallRecord 之间**——压缩切口不得劈开一个 AssistantMsg 及其
   后续 ToolMsg 序列，否则 A2/A3 立刻失效。
2. **保留段以完整 CallRecord 起始**——切口之后的第一条消息必须是一个
   ``AssistantMsg``（可能是 CallRecord 头，也可能是最终答案），不能是孤立的
   ``ToolMsg`` 或 ``UserMsg`` 夹在调用序列中。
3. **每个水位只触发一次**——由 ``Watermark`` 记账（I5）；本模块只提供
   ``compress``，调用方传 ``level`` 前应先经 ``watermark.judge`` 判定。

摘要注入位置的决策（**必须与 A1 一致**）：

    摘要作为 ``messages[0]``（``SystemMsg``）之后、**保留段之前**的一条
    ``UserMsg``，内容带固定前缀 ``[上下文摘要]``。

    理由：

    * A1 要求 ``messages[0]`` 是 ``SystemMsg`` **且唯一**，因此不能新增
      ``SystemMsg`` 承载摘要；
    * 摘要描述的是"更早的对话历史"，语义上属于用户侧对会话的输入，
      而非系统要求；放在 ``UserMsg`` 中不会污染系统提示词前缀（I1）；
    * ``ToolMsg`` 必须绑定 ``ToolCall.id``（A2），天然不能承载摘要；
    * 该 ``UserMsg`` 内容非空，满足 A9；且位于保留段起始的完整
      ``CallRecord`` 之前，不插入任何未闭合的 ``tool_calls`` 序列内部（I6）。

    若此前已存在摘要（二次压缩），旧摘要会被合并进新摘要文本，不会累积。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.watermark import AUTO_2, Watermark

__all__ = [
    "SUMMARY_PREFIX",
    "CallRecord",
    "SplitResult",
    "default_keep_records",
    "split_call_records",
    "compress",
]

#: 摘要消息的固定前缀，便于识别与合并历史摘要。
SUMMARY_PREFIX = "[上下文摘要]"

#: ``summarizer`` 的签名：``(text) -> summary``。
Summarizer = Callable[[str], str]

#: 未配对尾部：结构上不完整的消息（正常情况下应为空）。
Tail = list[Message]


@dataclass(frozen=True, slots=True)
class CallRecord:
    """一个最小不可分单位。

    Attributes:
        messages: 该记录包含的全部消息，按原序。
        call_ids: 该记录内 ``AssistantMsg.tool_calls`` 的 id，按原序。
    """

    messages: tuple[Message, ...]

    @property
    def call_ids(self) -> tuple[str, ...]:
        """记录内全部 tool_call id。"""
        return tuple(call.id for m in self.messages for call in _calls_of(m))

    def text(self) -> str:
        """把记录渲染成可喂给摘要器的纯文本。"""
        return render_segment(self.messages)


@dataclass(frozen=True, slots=True)
class SplitResult:
    """``split_call_records`` 的结果。

    Attributes:
        system: ``messages[0]`` 的 SystemMsg（A1 保证存在）。
        head: ``system`` 之后、第一个 ``AssistantMsg`` 之前的消息
            （``UserMsg`` / 历史摘要等），可整体归入被压缩段。
        records: 完好的 CallRecord 序列。
        tail: 未配对尾部（末尾没有对应 ``ToolMsg`` 的调用，或悬挂的 ToolMsg）。
    """

    system: SystemMsg
    head: list[Message]
    records: list[CallRecord]
    tail: Tail

    @property
    def records_messages(self) -> list[Message]:
        """把全部记录摊平成消息列表。"""
        return [m for record in self.records for m in record.messages]


def _calls_of(message: Message) -> list[ToolCall]:
    if isinstance(message, AssistantMsg):
        return list(message.tool_calls)
    return []


def split_call_records(messages: Sequence[Message]) -> SplitResult:
    """把一个合法消息序列切成 ``SystemMsg + head + CallRecord* + tail``。

    Args:
        messages: 满足 A1/A2/A3 的消息序列（``SystemMsg`` 必须在首位）。

    Returns:
        ``SplitResult``。

    Raises:
        ValueError: ``messages`` 为空或首条不是 ``SystemMsg``（A1 前置条件）。
    """
    if not messages:
        raise ValueError("A1: messages 不得为空")
    first = messages[0]
    if not isinstance(first, SystemMsg):
        raise ValueError(f"A1: messages[0] 必须是 SystemMsg，实际为 {type(first).__name__}")
    system = first

    rest = list(messages[1:])
    head: list[Message] = []
    records: list[CallRecord] = []
    tail: Tail = []
    index = 0

    while index < len(rest):
        msg = rest[index]
        if isinstance(msg, (UserMsg, SystemMsg)):
            # 位于两条 CallRecord 之间的用户消息（新任务/新追问）本身是可压缩的
            # 前导文本，归入 head；它不会切开任何调用序列，因此不违反硬约束①。
            # SystemMsg 在这里出现意味着 A1 已被破坏，交给 validate_messages 报错。
            head.append(msg)
            index += 1
            continue

        if isinstance(msg, AssistantMsg):
            record: list[Message] = [msg]
            index += 1
            if msg.tool_calls:
                expected = {c.id for c in msg.tool_calls}
                while index < len(rest) and isinstance(rest[index], ToolMsg):
                    tool = rest[index]
                    if tool.tool_call_id not in expected:
                        break
                    expected.discard(tool.tool_call_id)
                    record.append(tool)
                    index += 1
                    if not expected:
                        break
                if expected:
                    # 未配对：整条记录连同其后全部消息归入尾部
                    tail.extend(record)
                    tail.extend(rest[index:])
                    index = len(rest)
                    break
            records.append(CallRecord(messages=tuple(record)))
            continue

        # 悬挂的 ToolMsg（无匹配 ToolCall）：结构非法，归入尾部让校验器报 A2
        tail.append(msg)
        index += 1

    return SplitResult(system=system, head=head, records=records, tail=tail)


def _render_message(message: Message) -> str:
    """把单条消息渲染为摘要输入文本，工具路径用 ``ToolMsg.name`` 标注。"""
    if isinstance(message, SystemMsg):
        return "[system]\n" + message.content
    if isinstance(message, UserMsg):
        return "[user]\n" + message.content
    if isinstance(message, AssistantMsg):
        lines: list[str] = []
        if message.content:
            lines.append(message.content)
        for call in message.tool_calls:
            lines.append(f"[assistant:call {call.name}] {call.arguments}")
        return "\n".join(lines)
    if isinstance(message, ToolMsg):
        # C3 要求：摘要使用 ToolMsg.name 标注工具路径
        return f"[tool:{message.name}]\n{message.content}"
    raise TypeError(f"未知消息类型: {type(message).__name__}")  # pragma: no cover


def render_segment(messages: Sequence[Message]) -> str:
    """渲染一段消息供摘要器使用。"""
    return "\n\n".join(_render_message(m) for m in messages)


def default_keep_records(capacity: int, level: int) -> int:
    """按水位计算保留的完整 CallRecord 数量。

    50% 水位保留 4 个记录，75% 水位保留 2 个。水位越高说明历史越长，
    允许折叠得更激进；该策略与文档"保留最近 K 条"一致。
    """
    if level >= int(round(AUTO_2 * 100)):
        return 2
    return 4


def _summary_user_msg(previous_summary: str, new_summary: str) -> UserMsg:
    body = new_summary.strip() or "（摘要为空）"
    if not previous_summary:
        return UserMsg(content=f"{SUMMARY_PREFIX}\n{body}")
    return UserMsg(content=f"{SUMMARY_PREFIX}\n{previous_summary.strip()}\n{body}")


def _find_previous_summary(messages: Sequence[Message]) -> tuple[int | None, str]:
    for index, message in enumerate(messages):
        if isinstance(message, UserMsg) and message.content.startswith(SUMMARY_PREFIX):
            return index, message.content[len(SUMMARY_PREFIX) :].strip()
    return None, ""


def compress(
    messages: Sequence[Message],
    level: int,
    summarizer: Summarizer,
    *,
    keep_records: int | None = None,
    watermark: Watermark | None = None,
) -> list[Message]:
    """按 CallRecord 边界折叠历史，返回新的消息列表。

    Args:
        messages: 当前消息序列（需满足 A1/A2/A3）。
        level: 压缩水位，``50`` 或 ``75``（与 ``Watermark.compressed_at`` 一致）。
        summarizer: 依赖注入的摘要函数 ``(text) -> summary``；本模块不直接调模型。
        keep_records: 保留的完整 CallRecord 数量；省略时按水位取默认值。
        watermark: 传入时按 I5 校验"该水位只触发一次"并记账。

    Returns:
        重建后的消息列表：``[SystemMsg, (UserMsg 摘要), *保留段, *tail]``，
        已通过 ``validate_messages``（A1/A2/A3）。

    Raises:
        ValueError: 水位重复触发（I5），或输入序列本身非法。

    Note:
        压缩段为空（没有可折叠的记录）时，原样返回浅拷贝，且**不**记账，
        避免"压缩了个寂寞"却占掉一次水位机会。
    """
    if watermark is not None and watermark.is_compressed(level):
        raise ValueError(f"I5: 水位 {level} 已触发过压缩，不得重复触发")

    split = split_call_records(list(messages))
    # 先校验输入，避免在非法输入上做摘要这种昂贵操作
    validate_messages(list(messages))

    keep = default_keep_records(watermark.capacity if watermark else 0, level)
    if keep_records is not None:
        keep = max(0, keep_records)

    compressible = len(split.records) - keep
    if compressible <= 0:
        return list(messages)

    # ① 边界只落在 CallRecord 之间：被压缩段 = head + records[:compressible]
    dropped_records = split.records[:compressible]
    kept_records = split.records[compressible:]
    # ② 保留段以完整 CallRecord 起始：kept_records[0] 的首条消息必为 AssistantMsg
    if kept_records:
        first_kept = kept_records[0].messages[0]
        if not isinstance(first_kept, AssistantMsg):  # pragma: no cover - split 已保证
            raise ValueError("C3: 保留段必须以完整 CallRecord（AssistantMsg）起始")

    prior_index, prior_summary = _find_previous_summary(split.head)
    older = [m for i, m in enumerate(split.head) if i != prior_index]
    to_summarize = [*older, *(m for r in dropped_records for m in r.messages)]
    if not to_summarize:
        return list(messages)

    if not kept_records and not split.tail:
        # 只保留摘要会导致历史里没有任何"最近消息"；此时整体退回不压缩，
        # 由上层（StepController）在 85% 时收尾。
        return list(messages)

    new_summary = summarizer(render_segment(to_summarize))
    summary_msg = _summary_user_msg(prior_summary, new_summary)
    rebuilt: list[Message] = [split.system, summary_msg]
    rebuilt.extend(m for r in kept_records for m in r.messages)
    rebuilt.extend(split.tail)

    validate_messages(rebuilt)

    if watermark is not None:
        watermark.mark_compressed(level)

    return rebuilt

