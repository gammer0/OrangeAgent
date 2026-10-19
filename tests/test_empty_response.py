"""空回复处理回归测试。

历史问题：模型返回 ``content=""`` + 无 tool_calls 时，
``is_final`` 只看 tool_calls 为空，被误判为完成，任务被静默截断。
实测出现过"写了一半就退出、最终回复为空"。
"""

from __future__ import annotations

from orange.loop import AgentLoop, StepController
from orange.message import AssistantMsg, SystemMsg


class _EmptyModel:
    """始终返回空回复的假模型。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages) -> AssistantMsg:  # noqa: ANN001
        self.calls += 1
        return AssistantMsg(content="", tool_calls=[])


class _FinalModel:
    """返回正常最终答案的假模型。"""

    def chat(self, messages) -> AssistantMsg:  # noqa: ANN001
        return AssistantMsg(content="完成", tool_calls=[])


def _loop(model) -> AgentLoop:  # noqa: ANN001
    return AgentLoop(
        model=model,
        inbox=__import__("orange.inbox", fromlist=["Inbox"]).Inbox(),
        executor=lambda plan, resolver, on_group_done=None, **kw: {},
        resolver=lambda call: None,
        system_msg=SystemMsg(content="sys"),
        controller=StepController(max_steps=20),
    )


def test_empty_content_is_not_final() -> None:
    """空 content + 无调用不得被当作最终答案。"""
    message = AssistantMsg(content="", tool_calls=[])
    assert message.is_final is False


def test_whitespace_only_is_not_final() -> None:
    """纯空白同样不算最终答案。"""
    assert AssistantMsg(content="   \n\t ", tool_calls=[]).is_final is False


def test_normal_content_is_final() -> None:
    """有实际文本才是最终答案。"""
    assert AssistantMsg(content="完成了", tool_calls=[]).is_final is True


def test_loop_stops_on_consecutive_empty_responses() -> None:
    """连续空回复达到阈值后中止，而不是静默"完成"。"""
    model = _EmptyModel()
    result = _loop(model).run()

    assert result.stop_reason == "empty_response"
    assert result.final_text is None
    # 阈值 2：不应无限空转
    assert model.calls <= 3


def test_normal_run_unaffected() -> None:
    """正常回复不受影响。"""
    result = _loop(_FinalModel()).run()
    assert result.final_text == "完成"
