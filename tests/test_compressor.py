"""C3 GlobalCompressor 测试。

覆盖：边界只落在 CallRecord 之间、保留段以完整 CallRecord 起始、
压缩后 ``validate_messages`` 通过（A1/A2/A3）、SystemMsg 唯一性、
摘要注入位置、I5 每水位只触发一次。
"""

from __future__ import annotations

import pytest

from harness.compressor import (
    SUMMARY_PREFIX,
    CallRecord,
    compress,
    render_segment,
    split_call_records,
)
from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.watermark import Watermark


def _system() -> SystemMsg:
    return SystemMsg(content="SYS")


def _call(call_id: str, name: str = "/tool/os/file/read") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments='{"path": "a.txt"}')


def _round(call_id: str, *, content: str = "result", with_text: bool = True) -> list[Message]:
    """构造一个完整调用轮次：AssistantMsg(call) + ToolMsg。

    注意：调用序列内不得夹带 ``UserMsg``（I6），用户输入在上层已转为新的
    ``AssistantMsg`` 之前的独立轮次；此处只构造调用记录本身。
    """
    return [
        AssistantMsg(
            content=f"准备调用 {call_id}" if with_text else None,
            tool_calls=[_call(call_id)],
        ),
        ToolMsg(tool_call_id=call_id, name="/tool/os/file/read", content=content),
    ]


def _long_history(rounds: int, *, per_round_user: bool = False) -> list[Message]:
    messages: list[Message] = [_system(), UserMsg(content="开始任务")]
    for i in range(rounds):
        if per_round_user:
            messages.append(UserMsg(content=f"请执行 c{i}"))
        messages.extend(_round(f"c{i}"))
    return messages


def _echo(text: str) -> str:
    return f"SUMMARY({len(text)})"


class TestSplitCallRecords:
    def test_groups_assistant_with_following_tool_msgs(self) -> None:
        messages = _long_history(3)
        split = split_call_records(messages)

        assert isinstance(split.system, SystemMsg)
        assert len(split.records) == 3
        assert split.tail == []
        for record in split.records:
            assert isinstance(record, CallRecord)
            assert isinstance(record.messages[0], AssistantMsg)
            assert isinstance(record.messages[1], ToolMsg)

    def test_head_holds_leading_user_msg(self) -> None:
        split = split_call_records(_long_history(1))
        assert [type(m) for m in split.head] == [UserMsg]

    def test_parallel_calls_share_one_record(self) -> None:
        messages: list[Message] = [
            _system(),
            UserMsg(content="并行"),
            AssistantMsg(tool_calls=[_call("a"), _call("b")]),
            ToolMsg(tool_call_id="a", name="/tool/os/file/read", content="A"),
            ToolMsg(tool_call_id="b", name="/tool/os/file/read", content="B"),
            AssistantMsg(content="完成"),
        ]
        split = split_call_records(messages)
        assert [type(m) for m in split.head] == [UserMsg]
        assert len(split.records) == 2
        assert split.records[0].call_ids == ("a", "b")
        assert split.records[1].messages == (messages[-1],)

    def test_user_msg_between_records_goes_to_head(self) -> None:
        """CallRecord 之间的用户消息是可压缩前导段，不进入 tail。"""
        messages = _long_history(2, per_round_user=True)
        split = split_call_records(messages)
        assert [m.content for m in split.head] == ["开始任务", "请执行 c0", "请执行 c1"]
        assert split.tail == []

    def test_final_answer_is_its_own_record(self) -> None:
        messages = [*_long_history(1), AssistantMsg(content="最终答案")]
        split = split_call_records(messages)
        assert len(split.records) == 2
        assert split.records[-1].messages == (AssistantMsg(content="最终答案"),)

    def test_requires_system_first(self) -> None:
        with pytest.raises(ValueError):
            split_call_records([UserMsg(content="x")])
        with pytest.raises(ValueError):
            split_call_records([])

    def test_render_marks_tool_path(self) -> None:
        text = render_segment(_round("c9"))
        assert "[tool:/tool/os/file/read]" in text
        assert "[assistant:call /tool/os/file/read]" in text

    def test_render_marks_user_text(self) -> None:
        assert render_segment([UserMsg(content="hi")]).startswith("[user]")


class TestCompressBoundaries:
    def test_boundary_only_between_call_records(self) -> None:
        """C3 硬约束①：切口只落在 CallRecord 之间。

        压缩后被丢弃的调用必须**整组**消失（AssistantMsg 与全部 ToolMsg 一起），
        不存在半个调用残留在结果里。
        """
        messages = _long_history(8)
        out = compress(messages, 50, _echo, keep_records=2)
        validate_messages(out)

        ids = [m.tool_call_id for m in out if isinstance(m, ToolMsg)]
        call_ids = [c.id for m in out if isinstance(m, AssistantMsg) for c in m.tool_calls]
        # 每个剩余调用都完整配对，且没有孤立的 ToolMsg
        assert sorted(ids) == sorted(call_ids)

        # 被折叠的组在输出中完全消失
        dropped = {"c0", "c1", "c2", "c3", "c4", "c5"}
        assert dropped.isdisjoint(ids)
        assert dropped.isdisjoint(call_ids)

    def test_kept_segment_starts_with_complete_call_record(self) -> None:
        """C3 硬约束②：保留段以完整 CallRecord 起始。"""
        messages = _long_history(8)
        out = compress(messages, 50, _echo, keep_records=3)

        summary_index = 1
        assert isinstance(out[summary_index], UserMsg)
        assert out[summary_index].content.startswith(SUMMARY_PREFIX)
        first_kept = out[summary_index + 1]
        assert isinstance(first_kept, AssistantMsg)
        # 紧随其后的必须是它的 ToolMsg
        assert isinstance(out[summary_index + 2], ToolMsg)
        assert out[summary_index + 2].tool_call_id == first_kept.tool_calls[0].id

    def test_rebuilt_history_passes_validate_messages(self) -> None:
        """压缩后必须通过 A1/A2/A3。"""
        for keep in (0, 1, 3):
            out = compress(_long_history(6), 50, _echo, keep_records=keep)
            validate_messages(out)
            # keep=0 会让历史只剩摘要，按设计退化为不压缩
            if keep == 0:
                assert out == _long_history(6)

    def test_system_msg_stays_unique_and_first(self) -> None:
        """A1：压缩不破坏 SystemMsg 唯一性。"""
        out = compress(_long_history(6), 50, _echo, keep_records=2)
        systems = [m for m in out if isinstance(m, SystemMsg)]
        assert len(systems) == 1
        assert out[0] is systems[0]
        assert out[0].content == "SYS"

    def test_system_msg_bytes_unchanged(self) -> None:
        """I1：压缩不改动 messages[0]。"""
        original = _system()
        messages = [original, *_long_history(6)[1:]]
        out = compress(messages, 50, _echo, keep_records=2)
        assert out[0] == original

    def test_nothing_to_compress_returns_unchanged(self) -> None:
        messages = _long_history(2)
        out = compress(messages, 50, _echo, keep_records=10)
        assert out == list(messages)

    def test_summary_text_passed_to_summarizer(self) -> None:
        seen: list[str] = []

        def summarizer(text: str) -> str:
            seen.append(text)
            return "折叠后的历史要点"

        messages = _long_history(6)
        out = compress(messages, 50, summarizer, keep_records=1)
        assert len(seen) == 1
        assert "开始任务" in seen[0]
        assert out[1] == UserMsg(content=f"{SUMMARY_PREFIX}\n折叠后的历史要点")

    def test_second_compression_merges_previous_summary(self) -> None:
        first = compress(_long_history(6), 50, lambda _: "第一段摘要", keep_records=2)
        extra = [*first, *_round("cx")]
        second = compress(extra, 75, lambda _: "第二段摘要", keep_records=1)

        validate_messages(second)
        summaries = [
            m for m in second if isinstance(m, UserMsg) and m.content.startswith(SUMMARY_PREFIX)
        ]
        assert len(summaries) == 1
        assert "第一段摘要" in summaries[0].content
        assert "第二段摘要" in summaries[0].content

    def test_head_user_msg_is_compressed_not_duplicated(self) -> None:
        messages = _long_history(6)
        out = compress(messages, 50, _echo, keep_records=1)
        user_texts = [m.content for m in out if isinstance(m, UserMsg)]
        assert "开始任务" not in user_texts
        assert user_texts[0].startswith(SUMMARY_PREFIX)


class TestEachLevelOnceOnly:
    def test_watermark_marks_level_after_compress(self) -> None:
        wm = Watermark(capacity=100_000)
        compress(_long_history(6), 50, _echo, keep_records=1, watermark=wm)
        assert wm.compressed_at == {50}

    def test_same_level_twice_raises(self) -> None:
        """I5：同一水位不得重复压缩。"""
        wm = Watermark(capacity=100_000)
        compress(_long_history(6), 50, _echo, keep_records=1, watermark=wm)
        with pytest.raises(ValueError):
            compress(_long_history(6), 50, _echo, keep_records=1, watermark=wm)

    def test_two_levels_allowed_sequentially(self) -> None:
        wm = Watermark(capacity=100_000)
        first = compress(_long_history(8), 50, _echo, keep_records=3, watermark=wm)
        second = compress(first, 75, _echo, keep_records=1, watermark=wm)
        assert wm.compressed_at == {50, 75}
        validate_messages(second)

    def test_no_op_does_not_consume_level(self) -> None:
        """无可压缩内容时不记账，避免白白用掉一次水位机会。"""
        wm = Watermark(capacity=100_000)
        messages = _long_history(1)
        out = compress(messages, 50, _echo, keep_records=10, watermark=wm)
        assert out == list(messages)
        assert wm.compressed_at == set()


class TestSummaryPosition:
    def test_summary_is_user_msg_right_after_system(self) -> None:
        out = compress(_long_history(6), 50, _echo, keep_records=1)
        assert isinstance(out[0], SystemMsg)
        assert isinstance(out[1], UserMsg)
        assert out[1].content.startswith(SUMMARY_PREFIX)

    def test_summary_user_msg_not_empty(self) -> None:
        """A9：摘要为空串也必须产出合法的 UserMsg。"""
        out = compress(_long_history(6), 50, lambda _: "   ", keep_records=1)
        assert isinstance(out[1], UserMsg)
        assert out[1].content.strip() != ""
