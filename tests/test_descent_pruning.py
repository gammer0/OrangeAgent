"""下钻轨迹的清理时机：必须等到选到具体工具之后。

这是防死循环的关键约束：
  * 纯下钻（还没选到叶子）时**绝不能**清理，否则模型看不到自己刚请求的
    子节点，会重复下钻同一条路径直到 max_steps；
  * 一旦本批次里出现了具体工具调用，之前的下钻链就完成了使命，可以删除，
    因为它揭示的工具已经进入发现缓存，每次请求都会重新注入。
"""

from __future__ import annotations

from typing import Never

from orange.loop import AgentLoop
from orange.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg


class _Model:
    def chat(self, messages) -> Never:  # noqa: ANN001, ANN201, D401
        raise AssertionError("model must not be called")


class _Inbox:
    def put_control(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        pass

    def take_control_at_sub_batch(self) -> None:  # noqa: ANN201
        return None


class _Controller:
    def begin_step(self) -> bool:  # noqa: ANN201
        return False


class _Group:
    def __init__(self, calls) -> None:  # noqa: ANN001
        self.calls = calls


class _Plan:
    def __init__(self, calls) -> None:  # noqa: ANN001
        self.groups = [_Group(calls)]


def _loop(abstract: frozenset[str]) -> AgentLoop:
    loop = AgentLoop(
        model=_Model(),
        inbox=_Inbox(),
        executor=lambda *a, **k: {},
        resolver=lambda call: None,
        system_msg=SystemMsg(content="sys"),
        controller=_Controller(),
    )
    loop.descent_paths = abstract
    loop.descent_call_ids = set()
    return loop


def _chain(descent_names: list[str]) -> tuple[list, list[ToolCall]]:
    """Build messages for a descent chain: each step is assistant + tool result."""
    messages: list = [SystemMsg(content="sys"), UserMsg(content="读文件")]
    calls: list[ToolCall] = []
    for index, name in enumerate(descent_names):
        call = ToolCall(id=f"d{index}", name=name, arguments="{}")
        calls.append(call)
        messages.append(AssistantMsg(content="", tool_calls=[call]))
        messages.append(ToolMsg(tool_call_id=call.id, name=name, content="{}"))
    return messages, calls


class TestPruneTiming:
    def test_pure_descent_is_never_pruned(self) -> None:
        """纯下钻批次必须原样保留 —— 否则会重复下钻（死循环）。"""
        abstract = frozenset({"/tool", "/tool/os"})
        loop = _loop(abstract)
        descent = ToolCall(id="d0", name="/tool", arguments="{}")
        messages, _ = _chain(["/tool"])

        loop._mark_descent_turns(messages, _Plan([descent]))
        loop._prune_resolved_descents(messages, _Plan([descent]))

        assert len(messages) == 4, messages
        assert any(
            isinstance(m, ToolMsg) and m.name == "/tool" for m in messages
        )

    def test_two_step_descent_still_kept_until_leaf(self) -> None:
        """连续两级下钻、尚未调用叶子时，两级都必须保留。"""
        abstract = frozenset({"/tool", "/tool/os", "/tool/os/file"})
        loop = _loop(abstract)
        messages, calls = _chain(["/tool", "/tool/os"])

        for call in calls:
            loop._mark_descent_turns(messages, _Plan([call]))
            loop._prune_resolved_descents(messages, _Plan([call]))

        # 仍然是 sys + user + 2×(assistant+tool)
        assert len(messages) == 6, [type(m).__name__ for m in messages]

    def test_pruned_once_leaf_is_called(self) -> None:
        """出现具体工具调用后，之前的整条下钻链被清除。"""
        abstract = frozenset({"/tool", "/tool/os", "/tool/os/file"})
        loop = _loop(abstract)
        messages, descents = _chain(["/tool", "/tool/os"])

        for call in descents:
            loop._mark_descent_turns(messages, _Plan([call]))
            loop._prune_resolved_descents(messages, _Plan([call]))

        # 现在调用叶子工具
        leaf = ToolCall(id="r1", name="/tool/os/file/read", arguments='{"path":"a"}')
        messages.append(AssistantMsg(content="", tool_calls=[leaf]))
        messages.append(
            ToolMsg(tool_call_id="r1", name="/tool/os/file/read", content="内容")
        )
        loop._mark_descent_turns(messages, _Plan([leaf]))
        loop._prune_resolved_descents(messages, _Plan([leaf]))

        names = [type(m).__name__ for m in messages]
        assert names == ["SystemMsg", "UserMsg", "AssistantMsg", "ToolMsg"], names
        tool_names = [m.name for m in messages if isinstance(m, ToolMsg)]
        assert tool_names == ["/tool/os/file/read"]

    def test_mixed_batch_keeps_real_call_only(self) -> None:
        """同一批次里既有下钻又有真实调用时，只清除下钻那部分。"""
        abstract = frozenset({"/tool"})
        loop = _loop(abstract)
        descent = ToolCall(id="d0", name="/tool", arguments="{}")
        real = ToolCall(id="r1", name="/tool/os/file/read", arguments='{"path":"a"}')

        messages = [
            SystemMsg(content="sys"),
            UserMsg(content="问"),
            AssistantMsg(content="", tool_calls=[descent, real]),
            ToolMsg(tool_call_id="d0", name="/tool", content="{}"),
            ToolMsg(tool_call_id="r1", name="/tool/os/file/read", content="内容"),
        ]
        loop._mark_descent_turns(messages, _Plan([descent, real]))
        loop._prune_resolved_descents(messages, _Plan([descent, real]))

        assistants = [m for m in messages if isinstance(m, AssistantMsg)]
        assert len(assistants) == 1
        assert [c.name for c in assistants[0].tool_calls] == ["/tool/os/file/read"]
        tool_names = [m.name for m in messages if isinstance(m, ToolMsg)]
        assert tool_names == ["/tool/os/file/read"]

    def test_final_answer_is_never_lost(self) -> None:
        """清理不得波及最终回答。"""
        abstract = frozenset({"/tool"})
        loop = _loop(abstract)
        descent = ToolCall(id="d0", name="/tool", arguments="{}")
        leaf = ToolCall(id="r1", name="/tool/os/file/read", arguments='{"path":"a"}')

        final = AssistantMsg(content="最终答案", tool_calls=[])
        messages = [
            SystemMsg(content="sys"),
            UserMsg(content="问"),
            AssistantMsg(content="", tool_calls=[descent]),
            ToolMsg(tool_call_id="d0", name="/tool", content="{}"),
            AssistantMsg(content="", tool_calls=[leaf]),
            ToolMsg(tool_call_id="r1", name="/tool/os/file/read", content="内容"),
            final,
        ]
        loop._mark_descent_turns(messages, _Plan([descent]))
        loop._prune_resolved_descents(messages, _Plan([leaf]))

        assert final in messages
        assert messages[-1] is final
