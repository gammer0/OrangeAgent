"""流式输出的正确性：碎片要能拼回完整消息，且不影响历史。

流式只是**预览**通道：前端靠 delta 逐字渲染，但写进历史的仍是一条完整
AssistantMsg。这组测试固定住三条约束：
  * 文本碎片按到达顺序拼接；
  * tool_calls 跨帧累积（首帧带 id/name，后续帧只追加 arguments）；
  * 不支持流式的 transport 自动退回普通调用。
"""

from __future__ import annotations

from typing import Any

from harness.model import (
    OpenAIAdapter,
    ReverseProxy,
    StreamAccumulator,
)


def _frame(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"delta": delta}], "model": "test-model"}


def test_text_fragments_concatenate() -> None:
    acc = StreamAccumulator()
    pieces = [acc.feed(_frame(content=c)) for c in ("你", "好", "世界")]
    assert pieces == ["你", "好", "世界"]

    response = acc.build()
    assert response.message.content == "你好世界"
    assert response.message.tool_calls == []
    assert response.message.is_final


def test_tool_calls_accumulate_across_frames() -> None:
    acc = StreamAccumulator()
    # First frame announces the call; later frames append argument fragments.
    acc.feed(
        _frame(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "/tool/os/file/read", "arguments": ""},
                }
            ]
        )
    )
    acc.feed(_frame(tool_calls=[{"index": 0, "function": {"arguments": '{"pa'}}]))
    acc.feed(_frame(tool_calls=[{"index": 0, "function": {"arguments": 'th":"a"}'}}]))

    response = acc.build()
    call = response.message.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "/tool/os/file/read"
    assert call.arguments == '{"path":"a"}'
    assert not response.message.is_final


def test_multiple_tool_calls_keep_their_order() -> None:
    acc = StreamAccumulator()
    acc.feed(
        _frame(
            tool_calls=[
                {"index": 0, "id": "a", "function": {"name": "/tool/x", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "/tool/y", "arguments": "{}"}},
            ]
        )
    )
    names = [c.name for c in acc.build().message.tool_calls]
    assert names == ["/tool/x", "/tool/y"]


def test_usage_and_model_are_captured() -> None:
    acc = StreamAccumulator()
    acc.feed(
        {
            "choices": [{"delta": {"content": "hi"}}],
            "model": "deepseek-flash",
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
    )
    response = acc.build()
    assert response.tokens_in == 11
    assert response.tokens_out == 3
    assert response.model == "deepseek-flash"


def test_reasoning_only_recorded_when_enabled() -> None:
    plain = StreamAccumulator()
    plain.feed(_frame(reasoning_content="think", content="x"))
    assert plain.build().message.reasoning_content is None

    reasoner = StreamAccumulator(reasoner=True)
    reasoner.feed(_frame(reasoning_content="think", content="x"))
    assert reasoner.build().message.reasoning_content == "think"


def test_empty_frames_are_ignored() -> None:
    acc = StreamAccumulator()
    assert acc.feed({"choices": []}) == ""
    assert acc.feed({"choices": [{"delta": {}}]}) == ""
    assert acc.build().message.content in {"", None}


class _StreamingTransport:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = frames
        self.sent: dict[str, Any] | None = None

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("streaming path must be used")

    def stream_post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> Any:
        self.sent = json
        yield from self.frames


def _proxy(transport: Any) -> ReverseProxy:
    from harness.model import ProviderConfig, ProviderHealth, RateWindow

    provider = ProviderConfig(
        name="fake",
        base_url="http://x",
        model="m",
        api_key="k",
        adapter=OpenAIAdapter(),
    )
    return ReverseProxy(
        providers=[provider],
        rate_window=RateWindow(max_requests=100, max_tokens=1_000_000),
        health={"fake": ProviderHealth(provider="fake")},
        transport=transport,
    )


def test_proxy_streams_and_requests_stream_flag() -> None:
    transport = _StreamingTransport(
        [_frame(content="a"), _frame(content="b"), _frame(content="c")]
    )
    proxy = _proxy(transport)
    deltas: list[str] = []

    response = proxy.chat_stream(
        [], session_id="s", on_delta=deltas.append, model="m"
    )

    assert deltas == ["a", "b", "c"], "fragments must reach the callback in order"
    assert response.message.content == "abc"
    assert transport.sent is not None
    assert transport.sent["stream"] is True


def test_proxy_falls_back_without_stream_support() -> None:
    class _Plain:
        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> Any:
            return {
                "choices": [{"message": {"content": "whole"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "model": "m",
            }

    proxy = _proxy(_Plain())
    deltas: list[str] = []
    response = proxy.chat_stream([], session_id="s", on_delta=deltas.append, model="m")

    assert response.message.content == "whole"
    assert deltas == [], "a non-streaming transport produces no deltas"


def test_sse_frames_are_decoded_despite_json_parameter_shadowing() -> None:
    """回归：``stream_post`` 的 ``json`` 参数遮蔽了 json 模块。

    早期实现直接调用 ``json.loads``，于是取到的是请求体字典，运行时抛
    ``'dict' object has no attribute 'loads'``——流式完全不可用，而普通路径
    因为不解析 SSE 帧毫无察觉。
    """
    from harness.model import OpenAITransport

    frames = [
        'data: {"choices":[{"delta":{"content":"流"}}]}',
        'data: {"choices":[{"delta":{"content":"式"}}]}',
        "data: [DONE]",
    ]

    class _Response:
        status_code = 200

        def iter_lines(self) -> Any:
            return iter(frames)

        def read(self) -> bytes:
            return b""

        text = ""

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class _Stream:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> _Response:
            return _Response()

        def __exit__(self, *exc: Any) -> None:
            return None

    import httpx

    original = httpx.stream
    httpx.stream = _Stream  # type: ignore[assignment]
    try:
        transport = OpenAITransport("http://x", "k")
        events = list(
            transport.stream_post("http://x/chat", headers={}, json={"stream": True})
        )
    finally:
        httpx.stream = original  # type: ignore[assignment]

    assert [e["choices"][0]["delta"]["content"] for e in events] == ["流", "式"]
