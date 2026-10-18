"""模型层测试：适配器差异转换、代理限流/熔断/重试。"""

from __future__ import annotations

from typing import Any

import pytest

from harness.message import AssistantMsg, SystemMsg, ToolCall, ToolMsg, UserMsg
from harness.model import (
    AnthropicAdapter,
    ModelError,
    OpenAIAdapter,
    ProviderConfig,
    RateLimitError,
    ReverseProxy,
)
from harness.rate import ProviderHealth, RateWindow

SYS = SystemMsg(content="sys")
USER = UserMsg(content="读文件")


def _down_health() -> ProviderHealth:
    """构造一个已熔断的健康记录（state=down）。"""
    health = ProviderHealth(provider="down")
    health.mark_down()
    return health


def _openai_payload(content: str | None = "hi", calls: list[dict] | None = None) -> dict:
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"content": content, "tool_calls": calls or []}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class FakeTransport:
    """按脚本返回响应的假传输。"""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"url": url, "headers": headers, "json": json})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestOpenAIAdapter:
    def test_build_request_shape(self) -> None:
        adapter = OpenAIAdapter()
        body = adapter.build_request(
            [SYS, USER], model="m", temperature=0.5, max_tokens=100
        )
        assert body["model"] == "m"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert body["temperature"] == 0.5

    def test_parse_plain_text(self) -> None:
        response = OpenAIAdapter().parse_response(_openai_payload("答案"))
        assert response.message.content == "答案"
        assert response.message.is_final
        assert response.tokens_in == 10

    def test_parse_tool_calls(self) -> None:
        payload = _openai_payload(
            content=None,
            calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "/tool/os/read", "arguments": '{"path":"a"}'},
                }
            ],
        )
        message = OpenAIAdapter().parse_response(payload).message
        assert not message.is_final
        assert message.tool_calls[0].name == "/tool/os/read"

    def test_missing_choices_is_not_retryable(self) -> None:
        with pytest.raises(ModelError) as exc:
            OpenAIAdapter().parse_response({})
        assert not exc.value.retryable


class TestAnthropicAdapter:
    def test_system_is_split_out(self) -> None:
        body = AnthropicAdapter().build_request(
            [SYS, USER], model="c", temperature=0.2, max_tokens=50
        )
        assert body["system"] == "sys"
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"][0]["type"] == "text"

    def test_assistant_tool_use_mapping(self) -> None:
        msgs = [
            SYS,
            AssistantMsg(
                content=None,
                tool_calls=[ToolCall(id="t1", name="/tool/os/read", arguments='{"path":"a"}')],
            ),
        ]
        body = AnthropicAdapter().build_request(
            msgs, model="c", temperature=0.2, max_tokens=50
        )
        block = body["messages"][0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["input"] == {"path": "a"}

    def test_tool_result_maps_to_user_turn(self) -> None:
        msgs = [SYS, ToolMsg(tool_call_id="t1", name="/tool/os/read", content="ok")]
        body = AnthropicAdapter().build_request(
            msgs, model="c", temperature=0.2, max_tokens=50
        )
        block = body["messages"][0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"

    def test_parse_content_blocks(self) -> None:
        payload = {
            "content": [
                {"type": "text", "text": "答"},
                {"type": "tool_use", "id": "t1", "name": "/a/b", "input": {"x": 1}},
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        response = AnthropicAdapter().parse_response(payload)
        assert response.message.tool_calls[0].arguments == '{"x": 1}'
        assert response.tokens_out == 3

    def test_discovered_appended_to_system(self) -> None:
        body = AnthropicAdapter().build_request(
            [SYS],
            model="c",
            temperature=0.2,
            max_tokens=50,
            discovered=[{"path": "/tool/os", "description": "文件"}],
        )
        assert "/tool/os — 文件" in body["system"]


def _proxy(responses: list[Any], **kw: Any) -> tuple[ReverseProxy, FakeTransport]:
    transport = FakeTransport(responses)
    proxy = ReverseProxy(
        providers=[
            ProviderConfig(
                name="deepseek",
                base_url="https://api.example.com",
                model="deepseek-chat",
                api_key="sk-test",
            )
        ],
        rate_window=RateWindow(max_requests=100, max_tokens=1_000_000),
        health={},
        transport=transport,
        _sleep=lambda _s: None,
        **kw,
    )
    return proxy, transport


class TestReverseProxy:
    def test_success_calls_adapter_and_reports_usage(self) -> None:
        proxy, transport = _proxy([_openai_payload("好的")])
        response = proxy.chat([SYS, USER], session_id="s1")

        assert response.message.content == "好的"
        assert transport.requests[0]["url"].endswith("/chat/completions")
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer sk-test"
        assert proxy.health["deepseek"].state == "healthy"

    def test_retries_retryable_error_then_succeeds(self) -> None:
        error = ModelError("boom", provider="http", retryable=True)
        proxy, transport = _proxy([error, _openai_payload("ok")])
        response = proxy.chat([SYS, USER], session_id="s1")

        assert response.message.content == "ok"
        assert len(transport.requests) == 2

    def test_non_retryable_error_raises_immediately(self) -> None:
        error = ModelError("bad", provider="http", retryable=False)
        proxy, transport = _proxy([error])
        with pytest.raises(ModelError):
            proxy.chat([SYS, USER], session_id="s1")
        assert len(transport.requests) == 1

    def test_rate_limit_rejects_before_sending(self) -> None:
        proxy, transport = _proxy([_openai_payload("x")])
        # token 配额极小，必然拒绝本次请求
        proxy.rate_window = RateWindow(max_requests=10, max_tokens=1)

        with pytest.raises(RateLimitError):
            proxy.chat([SYS, USER], session_id="s1")
        assert transport.requests == []

    def test_failure_marks_provider_degraded(self) -> None:
        error = ModelError("boom", provider="http", retryable=False)
        proxy, _ = _proxy([error])
        with pytest.raises(ModelError):
            proxy.chat([SYS, USER], session_id="s1")
        assert proxy.health["deepseek"].consecutive_failures == 1

    def test_failover_to_second_provider(self) -> None:
        transport = FakeTransport([_openai_payload("来自备用")])
        proxy = ReverseProxy(
            providers=[
                ProviderConfig(
                    name="down",
                    base_url="https://a",
                    model="m1",
                    api_key="k",
                ),
                ProviderConfig(
                    name="backup",
                    base_url="https://b",
                    model="m2",
                    api_key="k",
                ),
            ],
            rate_window=RateWindow(max_requests=100, max_tokens=1_000_000),
            health={"down": _down_health()},
            transport=transport,
            _sleep=lambda _s: None,
        )
        response = proxy.chat([SYS, USER], session_id="s1")

        assert response.message.content == "来自备用"
        assert transport.requests[0]["url"].startswith("https://b")

    def test_empty_providers_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReverseProxy(
                providers=[],
                rate_window=RateWindow(max_requests=1, max_tokens=1),
                health={},
            )
