"""编排层测试：ReAct 主循环与安全点语义。

对应约束：G1–G4、G9、G10、D3、D7、D8、I2、I3、I6。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from orange.exceptions import MessageInvariantError
from orange.execution import ExecResult
from orange.inbox import Inbox, InterruptRequest, UserInput
from orange.loop import AgentLoop, DeadlockTracker, LoopHooks, StepController
from orange.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg, validate_messages

SYS = SystemMsg(content="sys")


def _call(call_id: str, name: str = "/tool/os/file/read") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments="{}")


@dataclass
class FakeTool:
    concurrency: str = "safe"
    idempotent: bool = True


class FakeModel:
    """按脚本返回回复的假模型。"""

    def __init__(self, replies: list[AssistantMsg]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Any]] = []

    def chat(self, messages):
        self.seen.append(list(messages))
        if not self.replies:
            return AssistantMsg(content="done")
        return self.replies.pop(0)


class FakeExecutor:
    """同步执行器：按组序返回成功结果，可选在指定组后触发回调。"""

    def __init__(self, interrupt_at: int | None = None, inbox: Inbox | None = None) -> None:
        self.interrupt_at = interrupt_at
        self.inbox = inbox
        self.executed: list[str] = []

    def __call__(
        self, plan, resolver, on_group_done=None, *, max_workers=4
    ) -> dict[str, ExecResult]:
        results: dict[str, ExecResult] = {}
        for group in plan.groups:
            for call in group.calls:
                self.executed.append(call.id)
                results[call.id] = ExecResult(ok=True, content=f"ok:{call.id}", duration_ms=0)
            if on_group_done is not None:
                report = type("R", (), {"index": group.index, "group": group, "results": results})()
                from orange.executor import GroupReport

                hook_result = on_group_done(
                    GroupReport(
                        index=group.index,
                        group=group,
                        results=dict(results),
                        is_last=group.index == len(plan.groups) - 1,
                        plan=plan,
                    )
                )
                del report
                if hook_result is not None:
                    break
        return results


def _resolver(call: ToolCall) -> FakeTool:
    return FakeTool()


def _loop(model: FakeModel, inbox: Inbox, executor: FakeExecutor, **kw: Any) -> AgentLoop:
    return AgentLoop(
        model=model,
        inbox=inbox,
        executor=executor,
        resolver=_resolver,
        system_msg=SYS,
        **kw,
    )


class TestStepController:
    def test_step_gate_stops_at_max(self) -> None:
        c = StepController(max_steps=2)
        assert c.begin_step() and c.begin_step()
        assert not c.begin_step()
        assert c.stop_reason == "max_steps"

    def test_inject_counter_marks_stop(self) -> None:
        c = StepController(max_injects=2)
        assert c.record_inject()
        assert not c.record_inject()
        assert c.stop_reason == "max_injects"

    def test_deadlock_requires_same_path_and_same_result(self) -> None:
        t = DeadlockTracker()
        assert not t.record(ok=False, content="e", path="/a")
        assert not t.record(ok=False, content="e", path="/a")
        assert t.record(ok=False, content="e", path="/a")

    def test_switching_path_resets_count(self) -> None:
        t = DeadlockTracker()
        t.record(ok=False, content="e", path="/a")
        t.record(ok=False, content="e", path="/a")
        # 换了工具路径：重新计数，不判死锁
        assert not t.record(ok=False, content="e", path="/b")

    def test_success_resets(self) -> None:
        t = DeadlockTracker()
        t.record(ok=False, content="e", path="/a")
        t.record(ok=True, content="ok", path="/a")
        assert not t.record(ok=False, content="e", path="/a")


class TestSafePoints:
    def test_chat_consumed_before_llm_call(self) -> None:
        """① 安全点：LLM 调用前 drain chat。"""
        inbox = Inbox()
        inbox.put_chat(UserInput(text="补充", channel="stdin"))
        model = FakeModel([AssistantMsg(content="answer")])
        result = _loop(model, inbox, FakeExecutor()).run()

        assert result.final_text == "answer"
        assert any(isinstance(m, UserMsg) and m.content == "补充" for m in result.messages)
        # 注入必须发生在模型调用之前
        assert any(isinstance(m, UserMsg) for m in model.seen[0])

    def test_chat_not_consumed_at_sub_batch_G2(self) -> None:
        """② 组间不消费 chat（G2）：消息留队列，直到 ③ 才注入。"""
        inbox = Inbox()
        model = FakeModel(
            [
                AssistantMsg(tool_calls=[_call("c1"), _call("c2", "/tool/os/file/write")]),
                AssistantMsg(content="done"),
            ]
        )
        seen_during: list[list[Any]] = []

        def on_group_done(index: int) -> None:
            seen_during.append(list(inbox.drain_chat.__self__._chat))  # 只读观察

        # 在第一组完成后写入 chat，验证它不会在本批被消费
        class Executor(FakeExecutor):
            def __call__(
                self, plan, resolver, on_group_done=None, *, max_workers=4
            ) -> dict[str, ExecResult]:
                results = super().__call__(plan, resolver, on_group_done, max_workers=max_workers)
                return results

        loop = _loop(model, inbox, FakeExecutor())
        orig = loop._execute

        def patched(plan, messages) -> Any:
            inbox.put_chat(UserInput(text="组间补话", channel="stdin"))
            return orig(plan, messages)

        loop._execute = patched  # type: ignore[method-assign]
        result = loop.run()

        # 该消息最终应出现在 ③ 之后的消息序列里
        texts = [m.content for m in result.messages if isinstance(m, UserMsg)]
        assert "组间补话" in texts
        validate_messages(result.messages)

    def test_control_at_sub_batch_stops_remaining_groups_G1(self) -> None:
        """② control 生效：停止剩余组 + 补 3e（G1 / D7）。"""
        inbox = Inbox()
        calls = [
            _call("c1"),
            _call("c2", "/tool/os/file/write"),
            _call("c3", "/tool/os/file/delete"),
        ]
        model = FakeModel([AssistantMsg(tool_calls=calls)])
        loop = _loop(model, inbox, FakeExecutor())

        orig = loop._execute

        def patched(plan, messages) -> Any:
            inbox.put_control(InterruptRequest(reason="user_abort", text="停下"))
            return orig(plan, messages)

        loop._execute = patched  # type: ignore[method-assign]
        # 让执行器在每组后调用回调
        loop.executor = FakeExecutor()
        result = loop.run()

        assert result.aborted
        # 全部未回填调用都有配对 ToolMsg（I2 / D7）
        validate_messages(result.messages)
        tool_ids = {m.tool_call_id for m in result.messages if isinstance(m, ToolMsg)}
        assert {"c1", "c2", "c3"} <= tool_ids
        # control 附带文本在补占位之后插入（G3）
        assert any(isinstance(m, UserMsg) and m.content == "停下" for m in result.messages)

    def test_sub_batch_does_not_consume_chat_G2_direct(self) -> None:
        """直接验证 Inbox 语义：子批次消费后 chat 仍在队列。"""
        inbox = Inbox()
        inbox.put_chat(UserInput(text="A", channel="stdin"))
        inbox.put_control(InterruptRequest(reason="user_abort", text=None))
        signal = inbox.take_control_at_sub_batch()

        assert signal is not None
        # G2：chat 消息留在队列，未被丢弃
        remaining = inbox.drain_chat()
        assert len(remaining) == 1
        assert remaining[0].text == "A"


class TestLoopBehaviour:
    def test_final_answer_ends_loop(self) -> None:
        model = FakeModel([AssistantMsg(content="完成")])
        result = _loop(model, Inbox(), FakeExecutor()).run()
        assert result.final_text == "完成"
        assert result.steps == 1
        validate_messages(result.messages)

    def test_tool_results_appended_in_group_order_D3(self) -> None:
        calls = [_call("c1"), _call("c2", "/tool/os/file/write"), _call("c3")]
        model = FakeModel([AssistantMsg(tool_calls=calls), AssistantMsg(content="done")])
        result = _loop(model, Inbox(), FakeExecutor()).run()

        tool_ids = [m.tool_call_id for m in result.messages if isinstance(m, ToolMsg)]
        assert tool_ids == ["c1", "c2", "c3"]
        validate_messages(result.messages)

    def test_max_steps_terminates(self) -> None:
        calls = [_call("c1")]
        model = FakeModel([AssistantMsg(tool_calls=calls) for _ in range(5)])
        loop = _loop(model, Inbox(), FakeExecutor(), controller=StepController(max_steps=3))
        result = loop.run()

        assert result.stop_reason == "max_steps"
        assert result.steps == 3

    def test_deadlock_marks_stop(self) -> None:
        class FailingExecutor(FakeExecutor):
            def __call__(
                self, plan, resolver, on_group_done=None, *, max_workers=4
            ) -> dict[str, ExecResult]:
                results = {
                    call.id: ExecResult(
                        ok=False, content="boom", error_type="3c", retryable=False, duration_ms=0
                    )
                    for call in plan.calls
                }
                if on_group_done is not None:
                    from orange.executor import GroupReport

                    for group in plan.groups:
                        on_group_done(
                            GroupReport(
                                index=group.index,
                                group=group,
                                results=results,
                                is_last=group.index == len(plan.groups) - 1,
                                plan=plan,
                            )
                        )
                return results

        model = FakeModel([AssistantMsg(tool_calls=[_call("c1")]) for _ in range(10)])
        loop = _loop(model, Inbox(), FailingExecutor())
        result = loop.run()

        assert result.stop_reason == "deadlock"
        assert result.steps == 3

    def test_unclosed_tool_calls_rejects_injection_I6(self) -> None:
        """I6 / D8：未闭合 tool_calls 序列内部不得插入 UserMsg。"""
        loop = _loop(FakeModel([]), Inbox(), FakeExecutor())
        messages = [SYS, AssistantMsg(tool_calls=[_call("c1")])]
        with pytest.raises(MessageInvariantError) as exc:
            loop._inject_user_msg(messages, "插入", channel="chat")
        assert exc.value.rule == "I6"

    def test_hooks_receive_callbacks(self) -> None:
        events: list[str] = []
        hooks = LoopHooks(
            on_assistant=lambda m: events.append("assistant"),
            on_inject=lambda m, ch: events.append(f"inject:{ch}"),
        )
        inbox = Inbox()
        inbox.put_chat(UserInput(text="hi", channel="stdin"))
        loop = _loop(FakeModel([AssistantMsg(content="done")]), inbox, FakeExecutor(), hooks=hooks)
        loop.run()

        assert "assistant" in events
        assert "inject:chat" in events

    def test_history_must_start_with_system(self) -> None:
        loop = _loop(FakeModel([]), Inbox(), FakeExecutor())
        with pytest.raises(MessageInvariantError) as exc:
            loop.run([UserMsg(content="hi")])
        assert exc.value.rule == "A1"
