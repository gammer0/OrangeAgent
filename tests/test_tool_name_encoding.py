"""工具名编解码测试。

真实调用暴露的关键约束：OpenAI 协议要求 function.name 匹配
``^[a-zA-Z0-9_-]+$``，而内部 ``ToolCall.name`` 必须是绝对路径（A10）。
适配器边界做可逆编码，内部语义不变。
"""

from __future__ import annotations

import pytest

from harness.message import SystemMsg
from harness.model import (
    AnthropicAdapter,
    OpenAIAdapter,
    decode_tool_name,
    encode_tool_name,
)

SPEC = {
    "type": "function",
    "function": {
        "name": "/tool/os/file/read",
        "description": "读取文件",
        "parameters": {"type": "object", "properties": {}},
    },
}


class TestEncodeDecode:
    def test_encode_replaces_slashes(self) -> None:
        assert encode_tool_name("/tool/os/file/read") == "tool__os__file__read"

    def test_encoded_name_matches_provider_pattern(self) -> None:
        import re

        encoded = encode_tool_name("/tool/os/file/read")
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", encoded)

    def test_roundtrip_is_lossless(self) -> None:
        for path in ("/a", "/skill/mmd/pose/set_keyframe", "/mcp/server-1/tool_2"):
            assert decode_tool_name(encode_tool_name(path)) == path

    def test_decode_is_idempotent_for_paths(self) -> None:
        assert decode_tool_name("/already/a/path") == "/already/a/path"

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError):
            encode_tool_name("/")

    def test_illegal_characters_rejected(self) -> None:
        with pytest.raises(ValueError):
            encode_tool_name("/a b/c")


class TestOpenAIRequestTools:
    def test_tools_encoded_in_request(self) -> None:
        body = OpenAIAdapter().build_request(
            [SystemMsg(content="s")],
            model="m",
            temperature=0.1,
            max_tokens=10,
            tools=[SPEC],
        )
        assert body["tools"][0]["function"]["name"] == "tool__os__file__read"
        assert body["tool_choice"] == "auto"

    def test_caller_spec_not_mutated(self) -> None:
        OpenAIAdapter().build_request(
            [SystemMsg(content="s")],
            model="m",
            temperature=0.1,
            max_tokens=10,
            tools=[SPEC],
        )
        assert SPEC["function"]["name"] == "/tool/os/file/read"

    def test_no_tools_key_when_absent(self) -> None:
        body = OpenAIAdapter().build_request(
            [SystemMsg(content="s")], model="m", temperature=0.1, max_tokens=10
        )
        assert "tools" not in body

    def test_response_name_decoded_back_to_path(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "tool__os__file__read",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        message = OpenAIAdapter().parse_response(payload).message
        assert message.tool_calls[0].name == "/tool/os/file/read"


class TestAnthropicRequestTools:
    def test_tools_use_input_schema_and_encoded_name(self) -> None:
        body = AnthropicAdapter().build_request(
            [SystemMsg(content="s")],
            model="m",
            temperature=0.1,
            max_tokens=10,
            tools=[SPEC],
        )
        tool = body["tools"][0]
        assert tool["name"] == "tool__os__file__read"
        assert tool["input_schema"] == {"type": "object", "properties": {}}

    def test_response_tool_use_decoded(self) -> None:
        payload = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "tool__os__file__read",
                    "input": {"path": "a"},
                }
            ]
        }
        message = AnthropicAdapter().parse_response(payload).message
        assert message.tool_calls[0].name == "/tool/os/file/read"
