"""思考模式 reasoning_content 透传回归测试。

历史问题：DeepSeek 思考模式的响应带 ``reasoning_content``，但代码解析时丢弃，
导致同一会话后续请求 HTTP 400：
"The `reasoning_content` in the thinking mode must be passed back to the API."
"""

from __future__ import annotations

from orange.message import AssistantMsg, UserMsg, to_provider
from orange.model import OpenAIAdapter


def test_reasoning_content_preserved_in_request() -> None:
    """解析出的 reasoning_content 必须回传到请求体。"""
    message = AssistantMsg(content="", tool_calls=[], reasoning_content="我先看看文件")
    projected = to_provider([UserMsg(content="hi"), message])
    assert projected[-1]["reasoning_content"] == "我先看看文件"


def test_absent_reasoning_does_not_pollute_request() -> None:
    """没有推理内容时请求体不新增字段（保证其他供应商请求逐字节不变）。"""
    message = AssistantMsg(content="完成", tool_calls=[])
    projected = to_provider([message])
    assert "reasoning_content" not in projected[0]


def test_empty_response_with_reasoning_is_not_final() -> None:
    """只有推理、没有正文的回复不算最终答案，避免任务被静默截断。"""
    assert AssistantMsg(content="", tool_calls=[], reasoning_content="想").is_final is False
    assert AssistantMsg(content="", tool_calls=[]).is_final is False
    assert AssistantMsg(content="结论").is_final is True


def test_adapter_parses_reasoning_content() -> None:
    """适配器从供应商响应中解析出 reasoning_content。"""
    adapter = OpenAIAdapter()
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "内部推理过程",
                    "tool_calls": [],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "deepseek-flash",
    }
    response = adapter.parse_response(payload)
    assert response.message.reasoning_content == "内部推理过程"


def test_roundtrip_through_adapter() -> None:
    """响应解析 → 请求投影，推理内容不丢失。"""
    adapter = OpenAIAdapter()
    payload = {
        "choices": [{"message": {"content": "", "reasoning_content": "rt", "tool_calls": []}}],
        "usage": {},
        "model": "m",
    }
    message = adapter.parse_response(payload).message
    body = adapter.build_request(
        [message], model="m", temperature=0.7, max_tokens=100
    )
    assert body["messages"][0]["reasoning_content"] == "rt"
