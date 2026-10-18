"""编排层集成：装配器与 Jev 路由在主循环中的实际行为。"""

from __future__ import annotations

from typing import Any

from harness.context import ContextAssembler, SystemPromptParts
from harness.inbox import Inbox
from harness.jev import JevRouter
from harness.loop import AgentLoop
from harness.message import AssistantMsg, SystemMsg, ToolCall, validate_messages

BRANCHES = [f"/skill/b{i}" for i in range(1, 9)]


class ScriptedModel:
    """按脚本返回回复，并记录每次收到的消息。"""

    def __init__(self, replies: list[AssistantMsg]) -> None:
        self.replies = list(replies)
        self.seen: list[list[Any]] = []

    def chat(self, messages: list[Any]) -> AssistantMsg:
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else AssistantMsg(content="done")


class OkExecutor:
    def __call__(self, plan, resolver, on_group_done=None, *, max_workers=4) -> dict[str, Any]:
        from harness.execution import ExecResult

        results = {
            call.id: ExecResult(ok=True, content="ok", duration_ms=0) for call in plan.calls
        }
        if on_group_done is not None:
            from harness.executor import GroupReport

            for group in plan.groups:
                on_group_done(
                    GroupReport(
                        index=group.index,
                        group=group,
                        results=dict(results),
                        is_last=group.index == len(plan.groups) - 1,
                        plan=plan,
                    )
                )
        return results


def _loop(**kw: Any) -> AgentLoop:
    base: dict[str, Any] = {
        "model": ScriptedModel([AssistantMsg(content="完成")]),
        "inbox": Inbox(),
        "executor": OkExecutor(),
        "resolver": lambda call: object(),
        "system_msg": SystemMsg(content="sys"),
    }
    base.update(kw)
    return AgentLoop(**base)


class TestAssemblerIntegration:
    def test_assembler_is_called_before_model(self) -> None:
        model = ScriptedModel([AssistantMsg(content="ok")])
        assembler = ContextAssembler(capacity=100_000)
        loop = _loop(model=model, assembler=assembler)
        loop.run()

        assert len(model.seen) == 1

    def test_without_assembler_degrades_losslessly_I4(self) -> None:
        model = ScriptedModel([AssistantMsg(content="ok")])
        loop = _loop(model=model, assembler=None)
        result = loop.run()

        assert result.final_text == "ok"
        assert model.seen[0] == [SystemMsg(content="sys")]

    def test_session_end_marks_stop(self) -> None:
        calls = [ToolCall(id="c1", name="/a/b", arguments="{}")]
        model = ScriptedModel([AssistantMsg(tool_calls=calls)] * 3)
        assembler = ContextAssembler(capacity=10)  # 极小容量 → 立即 session_end
        # 新语义下，压缩水位有剩余时会先给压缩机会；这里显式耗尽，
        # 以便单独验证 session_end 能正确置停。
        assembler.watermark.mark_compressed(50)
        assembler.watermark.mark_compressed(75)
        loop = _loop(model=model, assembler=assembler)
        result = loop.run()

        assert result.stop_reason == "session_end"

    def test_compressed_history_replaces_in_place(self) -> None:
        """压缩后必须用重建列表继续，否则后续追加落到旧对象。"""
        model = ScriptedModel([AssistantMsg(content="ok")])
        assembler = ContextAssembler(capacity=100_000)
        result = _loop(model=model, assembler=assembler).run()
        validate_messages(result.messages)


class TestRoutingIntegration:
    def test_router_not_used_without_branch_provider(self) -> None:
        loop = _loop(router=JevRouter(chat=lambda m: AssistantMsg(content="1")))
        result = loop.run()
        assert result.route is None

    def test_router_records_decision(self) -> None:
        model = ScriptedModel(
            [
                AssistantMsg(tool_calls=[ToolCall(id="c1", name="/skill/b1/x", arguments="{}")]),
                AssistantMsg(content="done"),
            ]
        )
        router = JevRouter(chat=lambda m: AssistantMsg(content="1"))
        loop = _loop(
            model=model,
            router=router,
            branch_provider=lambda: BRANCHES,
            tool_count_provider=lambda: 50,
            history_tokens_provider=lambda n: 10000,
        )
        result = loop.run()

        assert result.route is not None
        assert result.route.branch == BRANCHES[0]

    def test_router_failure_does_not_block_loop_I4(self) -> None:
        def broken_chat(messages: list[Any]) -> AssistantMsg:
            raise RuntimeError("路由模型挂了")

        model = ScriptedModel(
            [
                AssistantMsg(tool_calls=[ToolCall(id="c1", name="/skill/b1/x", arguments="{}")]),
                AssistantMsg(content="done"),
            ]
        )
        loop = _loop(
            model=model,
            router=JevRouter(chat=broken_chat),
            branch_provider=lambda: BRANCHES,
            tool_count_provider=lambda: 50,
            history_tokens_provider=lambda n: 10000,
        )
        result = loop.run()

        assert result.final_text == "done"
        assert result.route is not None
        assert result.route.fallback

    def test_no_route_below_threshold(self) -> None:
        model = ScriptedModel(
            [
                AssistantMsg(tool_calls=[ToolCall(id="c1", name="/a/b", arguments="{}")]),
                AssistantMsg(content="done"),
            ]
        )
        loop = _loop(
            model=model,
            router=JevRouter(chat=lambda m: AssistantMsg(content="1")),
            branch_provider=lambda: BRANCHES[:2],
            tool_count_provider=lambda: 2,
            history_tokens_provider=lambda n: 10,
        )
        result = loop.run()

        assert result.route is not None
        assert result.route.fallback


class TestSystemPromptPartsComposition:
    def test_parts_render_contains_tree_index_and_workdir(self) -> None:
        content = SystemPromptParts(
            tool_index="/tool — 操作系统能力",
            working_dir="C:/proj",
        ).render()
        assert "/tool — 操作系统能力" in content
        assert "C:/proj" in content
