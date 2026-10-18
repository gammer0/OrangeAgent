"""C2 TokenEstimator + 水位判定（Watermark / JudgeResult）。

对应文档：

* docs/02-数据结构设计/08-运行时状态.md（Watermark / JudgeResult）
* docs/01-系统架构设计.md §4.3（水位定义）、§6 I5

水位定义::

    AUTO_1      = 0.50 × CAPACITY
    AUTO_2      = 0.75 × CAPACITY
    SESSION_END = 0.85 × CAPACITY   # 不压缩，收尾并提示开新会话

判定顺序（硬约束 I5）：先看 ``SESSION_END``（≥0.85 直接收尾、不压缩），
再看 ``AUTO_2``（≥0.75），最后看 ``AUTO_1``（≥0.50）。每个水位只触发一次。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.message import AssistantMsg, Message, SystemMsg, ToolMsg, UserMsg

__all__ = [
    "AUTO_1",
    "AUTO_2",
    "SESSION_END",
    "JudgeResult",
    "JudgeAction",
    "Watermark",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "judge",
]

#: 一级自动压缩水位（首次跨越 50% 触发）。
AUTO_1 = 0.50
#: 二级自动压缩水位（首次跨越 75% 触发）。
AUTO_2 = 0.75
#: 会话收尾水位（≥85% 只收尾，不压缩）。
SESSION_END = 0.85

#: 每字符对应的 token 粗估系数（不引入 tokenizer，见文档 §1.2 非目标）。
CHARS_PER_TOKEN = 2.0

JudgeAction = Literal["normal", "compress_1", "compress_2", "session_end"]

#: 水位 → 该水位触发时写入 ``compressed_at`` 的键（百分比整数）。
_WATERMARK_KEY: dict[str, int] = {"compress_1": 50, "compress_2": 75}


def estimate_tokens(content: str) -> int:
    """按 ``chars / 2.0`` 粗估 token 数（向上取整，保证非空文本至少 1）。"""
    return int(len(content) / CHARS_PER_TOKEN + 0.999_999_999)


def estimate_message_tokens(message: Message) -> int:
    """估算单条消息的 token 数。

    除 ``content`` 外，``ToolCall`` 的 ``name`` 与 ``arguments`` 也会进入 provider
    请求体，因此一并计入，避免工具调用密集的历史被低估。
    """
    if isinstance(message, (SystemMsg, UserMsg, ToolMsg)):
        return estimate_tokens(message.content)

    if isinstance(message, AssistantMsg):
        total = estimate_tokens(message.content or "")
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(call.arguments)
        return total

    raise TypeError(f"未知消息类型: {type(message).__name__}")  # pragma: no cover


def estimate_messages_tokens(messages: list[Message]) -> int:
    """估算整段历史的 token 数。"""
    return sum(estimate_message_tokens(m) for m in messages)


class Watermark(BaseModel):
    """上下文水位记录（F 组）。

    Attributes:
        capacity: 模型上下文上限（token）。
        compressed_at: 已触发压缩的水位（``50`` / ``75``）。**F1：只增不减，
            同一水位不重复加入。**
        last_estimate: 最近一次估算的 token 数。
    """

    model_config = ConfigDict(validate_assignment=False)

    capacity: int = Field(gt=0)
    compressed_at: set[int] = Field(default_factory=set)
    last_estimate: int = Field(default=0, ge=0)

    def is_compressed(self, level: int) -> bool:
        """该水位是否已触发过压缩。"""
        return level in self.compressed_at

    def mark_compressed(self, level: int) -> bool:
        """记录某水位已压缩。

        F1：集合只增不减；同一水位重复标记不产生变化。

        Returns:
            ``True`` 表示本次是新加入，``False`` 表示该水位此前已标记。
        """
        if level in self.compressed_at:
            return False
        self.compressed_at.add(level)
        return True

    def update_estimate(self, estimate: int) -> float:
        """写入最近估算值并返回占用比例。"""
        self.last_estimate = max(0, estimate)
        return self.ratio

    @property
    def ratio(self) -> float:
        """最近估算占容量的比例。"""
        return self.last_estimate / self.capacity

    def as_set(self) -> frozenset[int]:
        """只读快照，供需要对外暴露时使用（防止外部回写降低集合）。"""
        return frozenset(self.compressed_at)


class JudgeResult(BaseModel):
    """水位判定结果。

    Attributes:
        action: ``normal`` / ``compress_1`` / ``compress_2`` / ``session_end``。
        ratio: 估算占用比例（``estimate / capacity``）。
    """

    model_config = ConfigDict(frozen=True)

    action: JudgeAction
    ratio: float = Field(ge=0.0)

    @property
    def level(self) -> int | None:
        """本结果对应的压缩水位（``50`` / ``75``）；无需压缩时为 ``None``。"""
        return _WATERMARK_KEY.get(self.action)


def judge(
    messages: list[Message],
    capacity: int,
    *,
    watermark: Watermark | None = None,
    estimate: int | None = None,
) -> JudgeResult:
    """判定当前上下文水位。

    Args:
        messages: 当前会话消息列表。
        capacity: 模型上下文上限。
        watermark: 水位记录；传入时会就地更新 ``last_estimate``。
            **不负责记账**：``compressed_at`` 由 ``compressor.compress()``
            在压缩真正发生后写入，保证"判定"与"执行"解耦。
            不传则视为无状态判定（每个水位都会触发）。
        estimate: 已算好的 token 估算值；省略时按 ``messages`` 现算。

    Returns:
        ``JudgeResult``；``ratio`` 恒为 ``estimate / capacity``。

    Raises:
        ValueError: ``capacity`` 非正数。
    """
    if capacity <= 0:
        raise ValueError(f"capacity 必须为正数，实际为 {capacity}")

    if estimate is None:
        estimate = estimate_messages_tokens(messages)

    ratio = (
        watermark.update_estimate(estimate) if watermark is not None else estimate / capacity
    )

    # ponytail: 这里**只判定不记账**。记账交给真正执行压缩的 compress()——
    # 否则 compress() 的 I5 重复触发检查会立刻失败，压缩永远不生效。
    return _decide(ratio, watermark)


def _decide(ratio: float, watermark: Watermark | None) -> JudgeResult:
    """按 I5 顺序判定：session_end → 0.75 → 0.50。"""
    if ratio >= SESSION_END:
        # ≥85% 只收尾，不压缩；未压缩的水位也不会被标记
        return JudgeResult(action="session_end", ratio=ratio)

    if ratio >= AUTO_2 and not _already(watermark, AUTO_2):
        return JudgeResult(action="compress_2", ratio=ratio)

    if ratio >= AUTO_1 and not _already(watermark, AUTO_1):
        return JudgeResult(action="compress_1", ratio=ratio)

    return JudgeResult(action="normal", ratio=ratio)


def _already(watermark: Watermark | None, threshold: float) -> bool:
    """该阈值对应的水位是否已触发过（无 watermark 时永不为真）。"""
    if watermark is None:
        return False
    return watermark.is_compressed(int(round(threshold * 100)))
