"""上下文装配器测试（A0a–A0e、I1、I4、I5）。"""

from __future__ import annotations

from orange.context import (
    ContextAssembler,
    SystemPromptParts,
    clear_after_compress,
    has_unclosed_tool_calls,
    last_user_text,
)
from orange.discovery import TreeSessionState
from orange.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg, validate_messages

CAPACITY = 1000


def _parts(**kw: object) -> SystemPromptParts:
    base = {
        "tool_index": "/tool      — 操作系统能力\n/tool/os   — 文件与进程",
        "working_dir": "C:/proj",
        "goal": "整理文档",
    }
    base.update(kw)
    return SystemPromptParts(**base)


class TestSystemPromptFiveSegments:
    def test_segments_are_joined_in_fixed_order_A0a(self) -> None:
        parts = _parts()
        content = parts.render()

        # 五段按固定顺序拼接：第1段在前、第5段在末尾
        assert content.startswith(parts.system_requirements)
        assert content.endswith(parts.session_context())
        # 第2/3/4段相对顺序固定
        index_pos = content.index(parts.tool_index)
        memory_pos = content.index("MEMORY.md", index_pos)
        role_pos = content.index(parts.role_guide, memory_pos)
        assert index_pos < memory_pos < role_pos

    def test_segment_count_is_constant_even_when_empty(self) -> None:
        """第 2 段为空时仍占位，保证第 5 段位置不漂移（I1）。"""
        parts = _parts(tool_index="", memory_name="")
        # 空段仍参与拼接：第5段始终在末尾
        assert parts.render().endswith(parts.session_context())

    def test_working_memory_body_not_included_A0b(self) -> None:
        content = _parts().render()
        # 只允许出现相对路径，不得出现正文
        assert "WORK.md" in content
        assert ".agent" not in content
        assert "工作记忆正文" not in content

    def test_long_term_memory_only_filename_A0c(self) -> None:
        content = _parts().render()
        assert "MEMORY.md" in content
        assert "/.agent/MEMORY.md" not in content

    def test_session_context_has_no_system_identifiers_A0d(self) -> None:
        text = _parts().session_context()
        for banned in ("session_id", "sess-", "tenant", "created_at"):
            assert banned not in text
            assert banned not in text

    def test_goal_is_optional_in_last_segment(self) -> None:
        without = _parts(goal=None).session_context()
        assert "本次目标" not in without

    def test_index_depth_is_l1_l2_default_A0e(self) -> None:
        """第 2 段直接采用 ToolTree.tree_index() 的 L1+L2 产物。"""
        parts = _parts(tool_index="/tool      — 操作系统能力\n/tool/os   — 文件与进程")
        content = parts.render()
        start = content.index("/tool      —")
        segment = content[start : content.index("MEMORY.md")]
        assert segment.strip().count("\n") == 1  # L1 + L2 两条


class TestSystemMsgStability:
    def test_build_system_is_cached_and_byte_stable_I1(self) -> None:
        assembler = ContextAssembler(capacity=CAPACITY)
        first = assembler.build_system(_parts())
        second = assembler.build_system(_parts(tool_index="完全不同的索引"))

        assert first is second
        assert first.content == second.content

    def test_reset_system_allows_new_session(self) -> None:
        assembler = ContextAssembler(capacity=CAPACITY)
        first = assembler.build_system(_parts())
        assembler.reset_system()
        second = assembler.build_system(_parts(tool_index="新索引"))
        assert first.content != second.content


class TestAssembleWatermark:
    def _history(self, repeats: int) -> list:
        msgs: list = [SystemMsg(content="sys")]
        for _ in range(repeats):
            msgs.append(UserMsg(content="x" * 200))
            msgs.append(AssistantMsg(content="y" * 200))
        return msgs

    def test_normal_below_threshold(self) -> None:
        assembler = ContextAssembler(capacity=CAPACITY)
        msgs = [SystemMsg(content="sys"), UserMsg(content="hi")]
        result = assembler.assemble(msgs)

        assert result.action == "normal"
        assert not result.compressed
        assert result.messages == msgs

    def test_session_end_does_not_compress_I5(self) -> None:
        # 两个压缩水位均已用过 → 才能进入收尾
        assembler = ContextAssembler(capacity=100, summarizer=lambda text: "摘要")
        assembler.watermark.mark_compressed(50)
        assembler.watermark.mark_compressed(75)
        msgs = self._history(10)  # 远超 85%
        result = assembler.assemble(msgs)

        assert result.action == "session_end"
        assert not result.compressed
        assert len(result.messages) == len(msgs)

    def test_high_water_yields_to_pending_compress(self) -> None:
        """≥85% 但水位仍可用时，先压缩而不是直接收尾。

        实测单步可从 40% 跳到 90%；若收尾优先，压缩永远不触发。
        """
        assembler = ContextAssembler(capacity=100, summarizer=lambda text: "摘要")
        msgs = self._history(10)
        result = assembler.assemble(msgs)

        assert result.action in {"compress_1", "compress_2"}
        assert result.compressed

    def test_without_summarizer_degrades_losslessly_I4(self) -> None:
        assembler = ContextAssembler(capacity=100, summarizer=None)
        msgs = self._history(10)
        result = assembler.assemble(msgs)

        assert not result.compressed
        assert result.messages == msgs

    def test_compress_1_triggers_once_I5(self) -> None:
        calls: list[str] = []

        def summarizer(text: str) -> str:
            calls.append(text)
            return "历史摘要"

        # 4 轮 × 400 字符 ≈ 800 token / 1200 ≈ 67%（落在 50%–75% 区间）
        assembler = ContextAssembler(capacity=1200, summarizer=summarizer, keep_records_50=1)
        msgs = self._history(4)

        first = assembler.assemble(msgs)
        assert first.action == "compress_1"

        # 第二次同样水位不再重复压缩（compressed_at 只增）
        second = assembler.assemble(first.messages)
        assert second.action in {"normal", "compress_2", "session_end"}
        assert len(calls) <= 2

    def test_compressed_messages_stay_valid(self) -> None:
        def summarizer(text: str) -> str:
            return "摘要" * 5

        assembler = ContextAssembler(capacity=200, summarizer=summarizer, keep_records_50=1)
        msgs = [
            SystemMsg(content="sys"),
            UserMsg(content="a" * 100),
            AssistantMsg(
                content=None,
                tool_calls=[ToolCall(id="c1", name="/tool/os/x", arguments="{}")],
            ),
            ToolMsg(tool_call_id="c1", name="/tool/os/x", content="b" * 100),
            AssistantMsg(content="done"),
        ]
        result = assembler.assemble(msgs)
        if result.compressed:
            validate_messages(result.messages)
            assert isinstance(result.messages[0], SystemMsg)

    def test_empty_messages_rejected(self) -> None:
        assembler = ContextAssembler(capacity=CAPACITY)
        try:
            assembler.assemble([])
        except ValueError as exc:
            assert "messages 不得为空" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("应拒绝空消息列表")

    def test_non_system_first_raises(self) -> None:
        assembler = ContextAssembler(capacity=CAPACITY)
        try:
            assembler.assemble([UserMsg(content="hi")])
        except ValueError as exc:
            assert "A1" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("应拒绝非 SystemMsg 开头")


class TestHelpers:
    def test_clear_after_compress_empties_both_caches(self) -> None:
        state = TreeSessionState()
        state.state.mark("/a")
        clear = clear_after_compress(state)
        clear()
        assert state.delivered == frozenset()

    def test_has_unclosed_tool_calls(self) -> None:
        open_calls = [
            SystemMsg(content="s"),
            AssistantMsg(content=None, tool_calls=[ToolCall(id="c1", name="/a/b", arguments="{}")]),
        ]
        assert has_unclosed_tool_calls(open_calls)

        closed = [*open_calls, ToolMsg(tool_call_id="c1", name="/a/b", content="ok")]
        assert not has_unclosed_tool_calls(closed)

    def test_last_user_text_returns_most_recent(self) -> None:
        msgs = [
            SystemMsg(content="s"),
            UserMsg(content="第一句"),
            AssistantMsg(content="答复"),
            UserMsg(content="第二句"),
        ]
        assert last_user_text(msgs) == "第二句"
