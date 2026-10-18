"""C2 水位估算与判定测试。

覆盖：``chars / 2.0`` 估算、0.50 / 0.75 / 0.85 三档判定、
同一水位只触发一次（I5）、F1（compressed_at 只增不减）。
"""

from __future__ import annotations

import pytest

from harness.message import (
    AssistantMsg,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
)
from harness.watermark import (
    AUTO_1,
    AUTO_2,
    SESSION_END,
    Watermark,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    judge,
)

CAPACITY = 10_000


def _history() -> list[object]:
    return [
        SystemMsg(content="sys"),
        UserMsg(content="hi"),
        AssistantMsg(
            content=None,
            tool_calls=[ToolCall(id="c1", name="/tool/os/x", arguments="{}")],
        ),
        ToolMsg(tool_call_id="c1", name="/tool/os/x", content="ok"),
        AssistantMsg(content="done"),
    ]


class TestEstimateTokens:
    def test_chars_divided_by_two(self) -> None:
        assert estimate_tokens("a" * 100) == 50

    def test_odd_length_rounds_up(self) -> None:
        assert estimate_tokens("a" * 101) == 51

    def test_empty_text(self) -> None:
        assert estimate_tokens("") == 0

    def test_message_estimate_sums_history(self) -> None:
        messages = _history()
        total = estimate_messages_tokens(messages)  # type: ignore[arg-type]
        assert total == sum(estimate_message_tokens(m) for m in messages)  # type: ignore[arg-type]
        assert total > 0


class TestWatermark:
    def test_compressed_at_only_grows(self) -> None:
        wm = Watermark(capacity=CAPACITY)
        assert wm.mark_compressed(50) is True
        assert wm.mark_compressed(75) is True
        snapshot = wm.as_set()
        # 再次标记不改变集合（F1）
        assert wm.mark_compressed(50) is False
        assert wm.as_set() == snapshot
        assert wm.compressed_at == {50, 75}

    def test_update_estimate_sets_ratio(self) -> None:
        wm = Watermark(capacity=CAPACITY)
        assert wm.update_estimate(2500) == pytest.approx(0.25)
        assert wm.last_estimate == 2500


class TestJudgeThresholds:
    def test_below_auto1_is_normal(self) -> None:
        assert judge([], CAPACITY, estimate=4999).action == "normal"

    def test_exactly_auto1_triggers_compress_1(self) -> None:
        result = judge([], CAPACITY, estimate=int(CAPACITY * AUTO_1))
        assert result.action == "compress_1"
        assert result.ratio == pytest.approx(0.50)

    def test_exactly_auto2_triggers_compress_2(self) -> None:
        result = judge([], CAPACITY, estimate=int(CAPACITY * AUTO_2))
        assert result.action == "compress_2"

    def test_exactly_session_end_triggers_session_end(self) -> None:
        result = judge([], CAPACITY, estimate=int(CAPACITY * SESSION_END))
        assert result.action == "session_end"
        assert result.level is None

    def test_session_end_takes_priority_over_compress(self) -> None:
        wm = Watermark(capacity=CAPACITY)
        assert judge([], CAPACITY, watermark=wm, estimate=9000).action == "session_end"
        # 收尾不消耗任何压缩水位
        assert wm.compressed_at == set()


class TestEachLevelOnceOnly:
    def test_watermark_level_triggers_once(self) -> None:
        """I5：每个水位只触发一次。"""
        wm = Watermark(capacity=CAPACITY)
        first = judge([], CAPACITY, watermark=wm, estimate=5000)
        second = judge([], CAPACITY, watermark=wm, estimate=5500)
        assert first.action == "compress_1"
        assert second.action == "normal"
        assert wm.compressed_at == {50}

    def test_auto2_then_back_down_still_allows_auto1(self) -> None:
        """I5：跳过 50% 直接到 75% 后，50% 水位仍未触发，回落到 50%–75% 时补触发一次。"""
        wm = Watermark(capacity=CAPACITY)
        assert judge([], CAPACITY, watermark=wm, estimate=8000).action == "compress_2"
        assert judge([], CAPACITY, watermark=wm, estimate=6000).action == "compress_1"
        assert judge([], CAPACITY, watermark=wm, estimate=6000).action == "normal"
        assert wm.compressed_at == {50, 75}

    def test_two_levels_each_once_then_session_end(self) -> None:
        wm = Watermark(capacity=CAPACITY)
        sequence = [
            (5000, "compress_1"),
            (5100, "normal"),
            (7500, "compress_2"),
            (7600, "normal"),
            (8500, "session_end"),
            (9000, "session_end"),
        ]
        for estimate, expected in sequence:
            assert judge([], CAPACITY, watermark=wm, estimate=estimate).action == expected
        assert wm.compressed_at == {50, 75}

    def test_watermark_records_last_estimate(self) -> None:
        wm = Watermark(capacity=CAPACITY)
        result = judge([], CAPACITY, watermark=wm, estimate=6000)
        assert wm.last_estimate == 6000
        assert result.ratio == pytest.approx(0.6)


class TestJudgeWithMessages:
    def test_estimate_from_messages(self) -> None:
        messages = [SystemMsg(content="s" * 2 * CAPACITY)]
        assert judge(messages, CAPACITY).action == "session_end"

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            judge([], 0)
