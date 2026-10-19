"""C3 压缩链路回归测试。

历史问题：``judge()`` 既判定又记账，导致 ``compress()`` 的 I5 检查
必然抛出"水位已触发过"，压缩永远不生效，且被 ``except ValueError`` 静默吞掉。
"""

from __future__ import annotations

from orange.context import ContextAssembler
from orange.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg

CAPACITY = 4_000


def _summarizer(text: str) -> str:
    return f"[摘要 {len(text)} 字符]"


def _history(records: int, chunk: int = 240) -> list:
    """每条记录约 ``chunk * 2`` 字符，便于精确控制水位。"""
    messages: list = [SystemMsg(content="system")]
    for index in range(records):
        messages.append(UserMsg(content=f"u{index}" + "x" * chunk))
        messages.append(
            AssistantMsg(
                content="",
                tool_calls=[ToolCall(id=f"c{index}", name="/tool/x", arguments="{}")],
            )
        )
        messages.append(ToolMsg(tool_call_id=f"c{index}", name="/tool/x", content="r" * chunk))
    return messages


def test_compress_available_reflects_summarizer() -> None:
    """未注入摘要器时不可用；注入后可用。"""
    assert ContextAssembler(capacity=CAPACITY).compress_available is False
    assert ContextAssembler(capacity=CAPACITY, summarizer=_summarizer).compress_available is True


def test_auto1_level_actually_shrinks_history() -> None:
    """50% 水位必须真正折叠历史（消息数下降）。"""
    assembler = ContextAssembler(capacity=CAPACITY, summarizer=_summarizer)
    history = _history(10)
    result = assembler.assemble(history)

    assert result.action == "compress_1"
    assert result.compressed is True, "judge 记账与 compress 检查冲突会导致此处为 False"
    assert len(result.messages) < len(history)


def test_compressed_history_keeps_system_first() -> None:
    """A1：压缩后首条仍是 SystemMsg。"""
    assembler = ContextAssembler(capacity=CAPACITY, summarizer=_summarizer)
    result = assembler.assemble(_history(10))
    assert isinstance(result.messages[0], SystemMsg)


def test_watermark_is_marked_only_after_compression() -> None:
    """记账发生在压缩真正执行之后，而不是判定阶段。"""
    assembler = ContextAssembler(capacity=CAPACITY, summarizer=_summarizer)
    result = assembler.assemble(_history(10))
    assert result.compressed is True
    assert 50 in assembler.watermark.compressed_at


def test_no_compress_error_on_success() -> None:
    """成功压缩时不应残留错误原因。"""
    assembler = ContextAssembler(capacity=CAPACITY, summarizer=_summarizer)
    assembler.assemble(_history(10))
    assert getattr(assembler, "last_compress_error", None) is None
