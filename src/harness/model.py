"""模型层：供应商适配器与反向代理。

对应文档：docs/01-系统架构设计.md §4.8

两个组件职责分离：

* :class:`Adapter` —— **只做供应商差异转换**：把内部 ``Message`` 投影成请求体，
  把供应商响应解析回 ``AssistantMsg``。不重试、不限流。
* :class:`ReverseProxy` —— **只做全局策略**：限流（``RateWindow``）、熔断与退避
  （``ProviderHealth``）、故障转移、用量归集。不关心供应商字段差异。

代理对上层完全透明：上层只调用 :meth:`ReverseProxy.chat`。

为什么不用 LiteLLM：``openai`` SDK 已覆盖全部 OpenAI 兼容端点（DeepSeek /
Moonshot / 本地 vLLM 等），差异化部分只剩 Anthropic 风格，约百行即可。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.message import AssistantMsg, Message, SystemMsg, ToolCall, to_provider
from harness.rate import ProviderHealth, RateWindow

__all__ = [
    "ModelError",
    "Adapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LLMResponse",
    "ToolSpec",
    "Transport",
    "OpenAITransport",
    "ReverseProxy",
    "ProviderConfig",
    "encode_tool_name",
    "decode_tool_name",
]


class ModelError(Exception):
    """模型调用失败（异常类型 ``1 ModelError``）。

    Attributes:
        provider: 出错时使用的供应商名。
        retryable: 是否值得重试（超时 / 5xx 为真，4xx 参数错误为假）。
    """

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


#: 工具路径 → provider 函数名的分隔符替换。
#:
#: OpenAI 协议要求 function.name 匹配 ``^[a-zA-Z0-9_-]+$``，不允许 ``/``；
#: 而内部 ``ToolCall.name`` 必须是完整绝对路径（A10）。因此在**适配器边界**
#: 做一次可逆编码：``/tool/os/file/read`` → ``tool__os__file__read``。
#: 内部语义与文档约束不变，只有出网请求体使用编码名。
_NAME_SEP = "__"
_PATH_SEP = "/"


def encode_tool_name(path: str) -> str:
    """把工具路径编码为符合 provider 约束的函数名。

    Args:
        path: 以 ``/`` 开头的完整绝对路径。

    Returns:
        仅含字母、数字、下划线与连字符的函数名。

    Raises:
        ValueError: 路径中含无法无损编码的字符。
    """
    stripped = path.strip(_PATH_SEP)
    if not stripped:
        raise ValueError(f"工具路径不得为空：{path!r}")
    encoded = stripped.replace(_PATH_SEP, _NAME_SEP)
    for ch in encoded:
        if not (ch.isalnum() or ch in "_-"):
            raise ValueError(f"工具路径含非法字符 {ch!r}：{path!r}")
    return encoded


def decode_tool_name(name: str) -> str:
    """把 provider 返回的函数名还原为工具绝对路径。

    Args:
        name: 模型返回的函数名；若已是绝对路径则原样返回（幂等）。

    Returns:
        ``/`` 开头的完整路径。
    """
    if name.startswith(_PATH_SEP):
        return name
    return _PATH_SEP + name.replace(_NAME_SEP, _PATH_SEP)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """适配器解析后的统一响应。

    Attributes:
        message: 可直接追加进会话的 ``AssistantMsg``。
        tokens_in: 输入 token（供应商未返回时为 0）。
        tokens_out: 输出 token（供应商未返回时为 0）。
        model: 实际使用的模型名。
    """

    message: AssistantMsg
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


#: 工具声明：OpenAI ``tools`` 数组中的单个元素。
ToolSpec = dict[str, Any]


class Transport(Protocol):
    """HTTP 传输层协议：便于测试注入假传输。"""

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        """发送 POST 并返回 JSON 响应体。"""
        ...


class Adapter(Protocol):
    """供应商适配器协议。"""

    name: str

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, str]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """把消息投影为供应商请求体。"""
        ...

    def parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        """把供应商响应解析为统一结构。"""
        ...


# ---------------------------------------------------------------------------
# OpenAI 兼容（DeepSeek / Moonshot / 本地 vLLM / OpenAI 本身）
# ---------------------------------------------------------------------------


class OpenAIAdapter:
    """OpenAI Chat Completions 协议适配器。"""

    name = "openai"

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, str]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """投影消息并按 Chat Completions 格式组装请求体。

        已发现工具清单以**末尾临时 system 消息**注入（A11–A13），不写入历史。
        ``tools`` 非空时必须随请求声明，否则模型会把工具调用当作普通文本输出。
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": to_provider(list(messages), discovered),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # provider 只接受 ^[a-zA-Z0-9_-]+$ 的函数名，路径需编码后出网
            body["tools"] = [_encode_spec(item) for item in tools]
            body["tool_choice"] = "auto"
        return body

    def parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        """解析 ``choices[0].message``。

        Raises:
            ModelError: 响应缺少 ``choices`` 或结构非法（属不可重试错误）。
        """
        choices = payload.get("choices")
        if not choices:
            raise ModelError("响应缺少 choices", provider=self.name, retryable=False)

        choice = choices[0]
        raw = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for item in raw.get("tool_calls") or []:
            function = item.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id", "")),
                    name=decode_tool_name(str(function.get("name", ""))),
                    arguments=str(function.get("arguments", "{}")),
                )
            )

        content = raw.get("content")
        message = AssistantMsg(
            content=content if content is not None else (None if tool_calls else ""),
            tool_calls=tool_calls,
            # ponytail: DeepSeek 思考模式要求把 reasoning_content 原样回传，
            # 丢弃会导致同一会话后续请求 400。
            reasoning_content=raw.get("reasoning_content"),
        )

        usage = payload.get("usage") or {}
        return LLMResponse(
            message=message,
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            model=str(payload.get("model", "")),
        )


# ---------------------------------------------------------------------------
# Anthropic 风格
# ---------------------------------------------------------------------------


class AnthropicAdapter:
    """Anthropic Messages 协议适配器。

    system 提示词单独成字段；``tool_use`` / ``tool_result`` 分别对应
    ``AssistantMsg.tool_calls`` 与 ``ToolMsg``。
    """

    name = "anthropic"

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, str]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """按 Messages 协议拆分 system 与对话轮次。"""
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMsg):
                system_parts.append(msg.content)
                continue

            if isinstance(msg, AssistantMsg):
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _safe_json(call.arguments),
                    }
                    for call in msg.tool_calls
                )
                turns.append({"role": "assistant", "content": blocks})
                continue

            if msg.role == "tool":
                turns.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
                continue

            turns.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})

        if discovered:
            listing = "\n".join(
                f"{item['path']} — {item['description']}" for item in discovered
            )
            system_parts.append(f"已发现工具：\n{listing}")

        body: dict[str, Any] = {
            "model": model,
            "messages": turns,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # Anthropic 的 tools 用 input_schema 而非 function.parameters
            body["tools"] = [
                {
                    "name": encode_tool_name(item["function"]["name"]),
                    "description": item["function"].get("description", ""),
                    "input_schema": item["function"].get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                for item in tools
            ]
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        return body

    def parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        """解析 ``content`` 块数组。"""
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ModelError("响应缺少 content", provider=self.name, retryable=False)

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in blocks:
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=decode_tool_name(str(block.get("name", ""))),
                        arguments=_dump_json(block.get("input") or {}),
                    )
                )

        content = "".join(texts)
        message = AssistantMsg(
            content=content if content or not tool_calls else None,
            tool_calls=tool_calls,
        )
        usage = payload.get("usage") or {}
        return LLMResponse(
            message=message,
            tokens_in=int(usage.get("input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0) or 0),
            model=str(payload.get("model", "")),
        )


def _safe_json(raw: str) -> Any:
    """尽力把 JSON 字符串转成对象；失败时原样返回（J1：不修复）。"""
    import json

    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _encode_spec(spec: ToolSpec) -> ToolSpec:
    """把工具声明中的路径名编码为 provider 可接受的函数名。

    输入形如 ``{"type": "function", "function": {"name": "/a/b", ...}}``，
    返回深拷贝，仅替换 ``function.name``，不修改调用方对象。
    """
    function = dict(spec.get("function") or {})
    function["name"] = encode_tool_name(str(function.get("name", "")))
    encoded = dict(spec)
    encoded["function"] = function
    return encoded


def _dump_json(value: Any) -> str:
    """把对象序列化为 JSON 字符串，失败时退回 ``repr``。"""
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# 传输层
# ---------------------------------------------------------------------------


class OpenAITransport:
    """基于 ``httpx`` 的最小传输实现（OpenAI 兼容端点）。"""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        """发送请求；非 2xx 统一抛 :class:`ModelError`。"""
        import httpx

        response = httpx.post(url, headers=headers, json=json, timeout=self.timeout)
        if response.status_code >= 500 or response.status_code == 429:
            raise ModelError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                provider="http",
                retryable=True,
            )
        if response.status_code >= 400:
            raise ModelError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                provider="http",
                retryable=False,
            )
        payload: dict[str, Any] = response.json()
        return payload


# ---------------------------------------------------------------------------
# 反向代理
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """单个供应商的接入配置。

    Attributes:
        name: 供应商标识（用于 ``ProviderHealth`` 与日志）。
        base_url: 请求基址；解析出的完整 URL 由代理拼接结尾路径。
        model: 模型名。
        api_key: 密钥。
        adapter: 适配器实例。
        endpoint: 具体路径，默认 ``/chat/completions``。
        headers: 额外请求头。
    """

    name: str
    base_url: str
    model: str
    api_key: str
    adapter: Adapter = field(default_factory=OpenAIAdapter)
    endpoint: str = "/chat/completions"
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ReverseProxy:
    """全局反向代理：限流 + 熔断 + 退避 + 故障转移。

    多会话共享同一实例，因此限流与健康状态均为**全局**（K4）。

    Attributes:
        providers: 供应商配置序列；第一个为默认。
        rate_window: 全局速率窗口。
        health: 每个供应商的健康状态。
        transport: HTTP 传输；省略时用 :class:`OpenAITransport`。
    """

    providers: list[ProviderConfig]
    rate_window: RateWindow
    health: dict[str, ProviderHealth]
    transport: Transport | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 2
    backoff_s: float = 1.0
    _sleep: Callable[[float], None] = time.sleep
    _clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("至少需要一个 provider")
        for provider in self.providers:
            self.health.setdefault(provider.name, ProviderHealth(provider=provider.name))

    # ---------------------------------------------------------------- 调用

    def chat(
        self,
        messages: Sequence[Message],
        *,
        session_id: str,
        discovered: list[dict[str, str]] | None = None,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """发起一次模型调用。

        Args:
            messages: 会话消息（调用方保证已通过 ``validate_messages``）。
            session_id: 用于用量归集与限流记账（K2）。
            discovered: 已发现工具清单，投影时末尾临时注入。
            model: 覆盖默认模型名。

        Returns:
            适配器解析后的统一响应。

        Raises:
            ModelError: 全部供应商均失败。
            RateLimitError: 全局速率窗口拒绝本次请求。
        """
        errors: list[str] = []
        for provider in self._candidates():
            health = self.health[provider.name]
            decision = health.can_send(self._clock())
            if decision == "fail_fast":
                errors.append(f"{provider.name}: 已熔断（fail_fast）")
                continue
            if decision == "retry_after":
                errors.append(f"{provider.name}: 退避中（retry_after）")
                continue

            try:
                return self._call_with_retry(
                    provider,
                    messages,
                    session_id=session_id,
                    discovered=discovered,
                    model=model or provider.model,
                    tools=tools,
                )
            except RateLimitError:
                # 限流是全局约束，不属于供应商故障：直接上抛，不计入健康度
                raise
            except ModelError as exc:
                health.on_failure(self._clock())
                errors.append(str(exc))
                if not exc.retryable:
                    raise

        raise ModelError("全部供应商失败: " + "; ".join(errors), provider="proxy")

    def _candidates(self) -> list[ProviderConfig]:
        """可用供应商顺序：健康的在前，已熔断的排后（仍尝试以支持恢复）。"""
        return sorted(
            self.providers,
            key=lambda p: 0 if self.health[p.name].state == "healthy" else 1,
        )

    def _call_with_retry(
        self,
        provider: ProviderConfig,
        messages: Sequence[Message],
        *,
        session_id: str,
        discovered: list[dict[str, str]] | None,
        model: str,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """对单个供应商做限流检查与退避重试。"""
        body = provider.adapter.build_request(
            messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            discovered=discovered,
            tools=tools,
        )
        estimated = _estimate_request_tokens(body)

        if not self.rate_window.try_acquire(session_id, estimated, self._clock()):
            raise RateLimitError(f"全局速率窗口拒绝请求（session={session_id}）")

        headers = {"Content-Type": "application/json", **provider.headers}
        if provider.api_key:
            headers.setdefault("Authorization", f"Bearer {provider.api_key}")

        url = provider.base_url.rstrip("/") + provider.endpoint
        transport = self.transport or OpenAITransport(provider.base_url, provider.api_key)

        last_error: ModelError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = transport.post(url, headers=headers, json=body)
                response = provider.adapter.parse_response(payload)
                self.rate_window.record_actual(
                    session_id, response.tokens_in + response.tokens_out, self._clock()
                )
                self.health[provider.name].on_success()
                return response
            except ModelError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.max_retries:
                    break
                self._sleep(self.backoff_s * (2**attempt))

        assert last_error is not None
        raise last_error


class RateLimitError(ModelError):
    """全局速率窗口拒绝（不计入供应商失败次数）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="rate", retryable=True)


def _estimate_request_tokens(body: dict[str, Any]) -> int:
    """粗估请求体 token：取消息文本字符数的一半。"""
    total = 0
    for item in body.get("messages") or []:
        content = item.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                text = block.get("text") if isinstance(block, dict) else None
                if isinstance(text, str):
                    total += len(text)
        for call in item.get("tool_calls") or []:
            function = call.get("function") or {}
            total += len(str(function.get("name", "")))
            total += len(str(function.get("arguments", "")))
    return max(1, int(total / 2.0 + 0.999))
