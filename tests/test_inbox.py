"""接入层双通道测试（G 组）。

对应文档：docs/02-数据结构设计/01-接入层.md
测试名带约束编号，便于按约束回归。
"""

from __future__ import annotations

import threading

import pytest

from orange.inbox import (
    CHAT_CAPACITY,
    DropNotice,
    Inbox,
    InterruptRequest,
    UserInput,
)


def _ui(text: str, channel: str = "stdin") -> UserInput:
    return UserInput(text=text, channel=channel)  # type: ignore[arg-type]


class _Recorder:
    """收集 ``on_drop`` 回调，模拟上层写 TraceEvent。"""

    def __init__(self) -> None:
        self.notices: list[DropNotice] = []
        self._lock = threading.Lock()

    def __call__(self, notice: DropNotice) -> None:
        with self._lock:
            self.notices.append(notice)


class TestChatFifo:
    def test_drain_chat_preserves_fifo_order(self) -> None:
        inbox = Inbox()
        for i in range(5):
            assert inbox.put_chat(_ui(str(i)))

        drained = inbox.drain_chat()

        assert [item.text for item in drained] == ["0", "1", "2", "3", "4"]
        assert inbox.drain_chat() == []

    def test_user_input_defaults_ts_and_channel(self) -> None:
        item = UserInput(text="hi", channel="ws")
        assert item.ts > 0
        assert item.channel == "ws"
        assert UserInput(text="x", channel="stdin").channel == "stdin"

    def test_capacity_is_100(self) -> None:
        assert CHAT_CAPACITY == 100
        assert Inbox().capacity == 100


class TestChatOverflow:
    def test_overflow_drops_oldest_and_notifies_G8(self) -> None:
        recorder = _Recorder()
        inbox = Inbox(on_drop=recorder)

        for i in range(CHAT_CAPACITY):
            inbox.put_chat(_ui(str(i)))
        assert recorder.notices == []

        inbox.put_chat(_ui("overflow"))

        assert len(recorder.notices) == 1
        notice = recorder.notices[0]
        assert notice.reason == "overflow"
        assert notice.channel == "chat"
        assert notice.dropped is not None
        assert notice.dropped.text == "0"

        drained = inbox.drain_chat()
        assert len(drained) == CHAT_CAPACITY
        assert drained[0].text == "1"  # 最旧被丢
        assert drained[-1].text == "overflow"

    def test_overflow_without_callback_still_drops_G8(self) -> None:
        inbox = Inbox(capacity=2, on_drop=None)
        inbox.put_chat(_ui("a"))
        inbox.put_chat(_ui("b"))
        inbox.put_chat(_ui("c"))

        assert [item.text for item in inbox.drain_chat()] == ["b", "c"]

    def test_control_has_no_capacity_limit(self) -> None:
        inbox = Inbox()
        for _ in range(CHAT_CAPACITY * 3):
            inbox.put_control(InterruptRequest(reason="timeout"))
        assert len(inbox) == CHAT_CAPACITY * 3


class TestEmptyTextRejected:
    def test_empty_text_rejected_and_notified_G12(self) -> None:
        recorder = _Recorder()
        inbox = Inbox(on_drop=recorder)

        accepted = inbox.put_chat(_ui(""))

        assert accepted is False
        assert inbox.drain_chat() == []
        assert len(recorder.notices) == 1
        notice = recorder.notices[0]
        assert notice.reason == "empty_text"
        assert notice.text == ""
        assert notice.dropped is None

    def test_whitespace_text_is_not_empty_G12(self) -> None:
        inbox = Inbox()
        assert inbox.put_chat(_ui(" ")) is True
        assert len(inbox.drain_chat()) == 1


class TestControlSemantics:
    def test_take_control_pops_first_and_clears_rest_G7(self) -> None:
        inbox = Inbox()
        first = InterruptRequest(reason="user_abort", text="改一下")
        inbox.put_control(first)
        inbox.put_control(InterruptRequest(reason="timeout"))
        inbox.put_control(InterruptRequest(reason="session_end"))

        taken = inbox.take_control()

        assert taken is first
        # 剩余元素已清空：第二次取为 None
        assert inbox.take_control() is None
        assert inbox.is_empty()

    def test_take_control_on_empty_returns_none(self) -> None:
        assert Inbox().take_control() is None

    def test_full_safe_point_consumes_control_before_chat_G4(self) -> None:
        order: list[str] = []
        inbox = Inbox()
        inbox.put_chat(_ui("chat-1"))
        inbox.put_control(InterruptRequest(reason="user_abort", text="ctl"))
        inbox.put_chat(_ui("chat-2"))

        interrupt, chats = inbox.drain_at_full_safe_point()
        if interrupt is not None:
            order.append(f"control:{interrupt.text}")
        order.extend(f"chat:{item.text}" for item in chats)

        assert order == ["control:ctl", "chat:chat-1", "chat:chat-2"]

    def test_sub_batch_does_not_consume_chat_G2(self) -> None:
        inbox = Inbox()
        inbox.put_chat(_ui("kept"))
        inbox.put_control(InterruptRequest(reason="timeout"))
        inbox.put_control(InterruptRequest(reason="session_end"))

        interrupt = inbox.take_control_at_sub_batch()

        assert interrupt is not None
        assert interrupt.reason == "timeout"
        # chat 既不出队也不丢弃
        assert [item.text for item in inbox.drain_chat()] == ["kept"]

    def test_sub_batch_with_no_control_leaves_chat_intact_G2(self) -> None:
        inbox = Inbox()
        inbox.put_chat(_ui("kept"))
        assert inbox.take_control_at_sub_batch() is None
        assert [item.text for item in inbox.drain_chat()] == ["kept"]


class TestInterruptRequestValidation:
    def test_accepts_three_reasons_G11(self) -> None:
        assert InterruptRequest(reason="user_abort").reason == "user_abort"
        assert InterruptRequest(reason="session_end").reason == "session_end"
        assert InterruptRequest(reason="timeout").reason == "timeout"

    def test_rejects_unknown_reason_G11(self) -> None:
        with pytest.raises(ValueError, match="G11"):
            InterruptRequest(reason="crash")  # type: ignore[arg-type]

    def test_user_abort_may_carry_text_G11(self) -> None:
        request = InterruptRequest(reason="user_abort", text="换个思路")
        assert request.text == "换个思路"

    @pytest.mark.parametrize("reason", ["session_end", "timeout"])
    def test_non_user_abort_text_must_be_none_G11(self, reason: str) -> None:
        with pytest.raises(ValueError, match="G11"):
            InterruptRequest(reason=reason, text="x")  # type: ignore[arg-type]
        assert InterruptRequest(reason=reason).text is None  # type: ignore[arg-type]


class TestConcurrency:
    def test_concurrent_put_chat_loses_nothing(self) -> None:
        inbox = Inbox(capacity=10_000)
        writers, per_writer = 8, 200

        def worker(writer_id: int) -> None:
            for i in range(per_writer):
                inbox.put_chat(_ui(f"{writer_id}:{i}"))

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        drained = inbox.drain_chat()
        assert len(drained) == writers * per_writer
        # 每个写入方的相对顺序保持不变（单一锁保证 FIFO 追加）
        for writer_id in range(writers):
            own = [item.text for item in drained if item.text.startswith(f"{writer_id}:")]
            assert own == [f"{writer_id}:{i}" for i in range(per_writer)]

    def test_concurrent_put_control_keeps_all_writes(self) -> None:
        inbox = Inbox()
        writers, per_writer = 4, 100

        def worker() -> None:
            for _ in range(per_writer):
                inbox.put_control(InterruptRequest(reason="timeout"))

        threads = [threading.Thread(target=worker) for _ in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # take_control 消费首个后清空，因此先断言总数（G7）
        assert len(inbox) == writers * per_writer
        assert inbox.take_control() is not None
        assert len(inbox) == 0
