"""消息 ADT 的最小验收测试（对应实施计划第 1 步：无损序列化）。"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from harness.exceptions import MessageInvariantError
from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    to_provider,
    validate_messages,
)

MESSAGES_ADAPTER = TypeAdapter(list[Message])


def _system() -> SystemMsg:
    return SystemMsg(content="sys")


def _call(call_id: str = "c1", name: str = "/tool/os/file/read") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments='{"path": "a.txt"}')


class TestConstruction:
    def test_user_msg_rejects_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            UserMsg(content="")

    def test_tool_call_rejects_relative_path(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(id="c1", name="tool/os/file/read", arguments="{}")

    def test_assistant_requires_content_or_calls(self) -> None:
        with pytest.raises(ValidationError):
            AssistantMsg()
        assert AssistantMsg(content="hi").is_final
        assert not AssistantMsg(tool_calls=[_call()]).is_final

    def test_is_final_is_derived_not_constructible(self) -> None:
        with pytest.raises(ValidationError):
            AssistantMsg.model_validate({"content": "x", "is_final": True})

    def test_tool_msg_requires_absolute_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolMsg(tool_call_id="c1", name="read", content="ok")


class TestValidateMessages:
    def test_accepts_well_formed_sequence(self) -> None:
        validate_messages(
            [
                _system(),
                UserMsg(content="读文件"),
                AssistantMsg(content=None, tool_calls=[_call()]),
                ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok"),
            ]
        )

    def test_rejects_when_first_is_not_system(self) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            validate_messages([UserMsg(content="hi")])
        assert exc.value.rule == "A1"

    def test_rejects_second_system_msg(self) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            validate_messages([_system(), _system()])
        assert exc.value.rule == "A1"

    def test_rejects_orphan_tool_msg(self) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            validate_messages(
                [_system(), ToolMsg(tool_call_id="cX", name="/tool/os/file/read", content="")]
            )
        assert exc.value.rule == "A2"

    def test_rejects_missing_tool_msg(self) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            validate_messages([_system(), AssistantMsg(tool_calls=[_call()])])
        assert exc.value.rule == "A3"

    def test_rejects_name_mismatch(self) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            validate_messages(
                [
                    _system(),
                    AssistantMsg(tool_calls=[_call()]),
                    ToolMsg(tool_call_id="c1", name="/tool/os/file/write", content=""),
                ]
            )
        assert exc.value.rule == "A6"


class TestProjection:
    def test_tool_msg_name_not_projected(self) -> None:
        projected = to_provider(
            [
                _system(),
                AssistantMsg(tool_calls=[_call()]),
                ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok"),
            ]
        )
        tool = projected[-1]
        assert tool == {"role": "tool", "tool_call_id": "c1", "content": "ok"}

    def test_assistant_none_content_kept_as_null(self) -> None:
        projected = to_provider([_system(), AssistantMsg(tool_calls=[_call()])])
        assert projected[1]["content"] is None
        assert projected[1]["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "/tool/os/file/read", "arguments": '{"path": "a.txt"}'},
            }
        ]

    def test_without_discovered_is_byte_identical(self) -> None:
        msgs = [_system(), UserMsg(content="hi")]
        assert to_provider(msgs) == to_provider(msgs, discovered=[])

    def test_discovered_appended_at_tail(self) -> None:
        msgs = [_system(), UserMsg(content="hi")]
        projected = to_provider(msgs, discovered=[{"path": "/tool/os", "description": "文件"}])
        assert len(projected) == len(msgs) + 1
        assert projected[-1]["role"] == "system"
        assert "/tool/os — 文件" in projected[-1]["content"]
        assert len(msgs) == 2  # 源列表未被改写（A12）


class TestRoundTrip:
    def test_serialize_and_reparse_is_lossless(self) -> None:
        msgs = [
            _system(),
            UserMsg(content="读文件"),
            AssistantMsg(content=None, tool_calls=[_call()]),
            ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok"),
            AssistantMsg(content="完成"),
        ]
        dumped = [m.model_dump() for m in msgs]
        restored = MESSAGES_ADAPTER.validate_python(dumped)

        assert [type(m) for m in restored] == [type(m) for m in msgs]
        assert [m.model_dump() for m in restored] == dumped
        validate_messages(restored)
