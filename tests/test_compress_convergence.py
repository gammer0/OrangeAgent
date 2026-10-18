"""压缩收敛性回归测试。

历史问题：``compress()`` 在"记录不足以切分"时返回 ``list(messages)``（新对象），
用 ``is not history`` 判断会误判为已压缩，导致：

1. 反复报 ``compressed=True`` 但消息数不变
2. 水位被空压缩吃掉，永远推进不到 ``session_end``（卡在 compress_1 空转）
"""

from __future__ import annotations

from harness.context import ContextAssembler
from harness.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg


def _history(records: int, size: int = 200) -> list:
    messages: list = [SystemMsg(content="sys")]
    for index in range(records):
        messages.append(UserMsg(content="u" * size))
        messages.append(
            AssistantMsg(
                content="",
                tool_calls=[ToolCall(id=f"c{index}", name="/t", arguments="{}")],
            )
        )
        messages.append(ToolMsg(tool_call_id=f"c{index}", name="/t", content="r" * size))
    return messages


def test_ineffective_compress_still_advances_to_session_end() -> None:
    """压缩无效时也必须消耗水位，最终能到达 session_end。

    用几乎不压缩的摘要器（只砍 20%）模拟"压了但没压下来"。
    """
    assembler = ContextAssembler(
        capacity=4_000, summarizer=lambda text: text[: int(len(text) * 0.8)]
    )
    history = _history(30)
    seen: list[str] = []

    for _ in range(6):
        result = assembler.assemble(history)
        seen.append(result.action)
        if result.compressed:
            history = result.messages
        if result.action == "session_end":
            break

    assert "session_end" in seen, f"未收敛到收尾，动作序列={seen}"
    assert assembler.watermark.compressed_at == {50, 75}


def test_no_infinite_compress_loop() -> None:
    """不得出现"消息数不变却反复报 compressed"的循环。"""
    assembler = ContextAssembler(
        capacity=4_000, summarizer=lambda text: text[: int(len(text) * 0.9)]
    )
    history = _history(30)
    steps = 0

    for _ in range(5):
        result = assembler.assemble(history)
        steps += 1
        if result.compressed:
            history = result.messages
        if result.action == "session_end":
            break

    assert steps <= 3, f"收敛过慢，用了 {steps} 步"


def test_effective_compress_still_works() -> None:
    """强压缩时行为不变：真的压下来，且只记一次账。"""
    assembler = ContextAssembler(capacity=4_000, summarizer=lambda text: "摘要")
    history = _history(30)
    result = assembler.assemble(history)

    assert result.compressed is True
    assert len(result.messages) < len(history)
    assert len(assembler.watermark.compressed_at) == 1
