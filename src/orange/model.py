"""Model layer: vendor adapters and the reverse proxy.

See docs/01-architecture.md §4.8.

The two components have separate responsibilities:

* :class:`Adapter` -- **vendor differences only**: project internal ``Message``
  objects into a request body and parse the vendor response back into
  ``AssistantMsg``. No retries, no rate limiting.
* :class:`ReverseProxy` -- **global policy only**: rate limiting
  (``RateWindow``), circuit breaking and backoff (``ProviderHealth``), failover,
  and usage accounting. It knows nothing about vendor field differences.

The proxy is fully transparent to the upper layers, which only call
:meth:`ReverseProxy.chat`.

Why not LiteLLM: the ``openai`` SDK already covers every OpenAI-compatible
endpoint (DeepSeek / Moonshot / local vLLM, ...). The only remaining difference
is the Anthropic style, which takes about a hundred lines.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from json import loads as _json_loads
from typing import Any, Protocol

from orange.message import AssistantMsg, Message, SystemMsg, ToolCall, to_provider
from orange.rate import ProviderHealth, RateWindow

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
    """A model call failed (error type ``1 ModelError``).

    Attributes:
        provider: The provider name in use when the failure happened.
        retryable: Whether a retry is worthwhile (timeouts / 5xx are true; 4xx
            argument errors are false).
    """

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


#: Separator replacing the tool-path separator when building a provider function
#: name.
#:
#: The OpenAI protocol requires ``function.name`` to match
#: ``^[a-zA-Z0-9_-]+$`` and forbids ``/``, while the internal ``ToolCall.name``
#: must be a full absolute path (A10). So at the **adapter boundary** we apply a
#: reversible encoding: ``/tool/os/file/read`` -> ``tool__os__file__read``.
#: Internal semantics and documented constraints are unchanged; only the
#: outgoing request body uses encoded names.
_NAME_SEP = "__"
_PATH_SEP = "/"


def encode_tool_name(path: str) -> str:
    """Encode a tool path into a provider-compliant function name.

    Args:
        path: A full absolute path starting with ``/``.

    Returns:
        A function name containing only letters, digits, underscores and hyphens.

    Raises:
        ValueError: The path contains a character that cannot be encoded losslessly.
    """
    stripped = path.strip(_PATH_SEP)
    if not stripped:
        raise ValueError(f"tool path must not be empty: {path!r}")
    encoded = stripped.replace(_PATH_SEP, _NAME_SEP)
    for ch in encoded:
        if not (ch.isalnum() or ch in "_-"):
            raise ValueError(f"tool path contains illegal character {ch!r}: {path!r}")
    return encoded


def decode_tool_name(name: str) -> str:
    """Restore a provider-returned function name back to a tool absolute path.

    Args:
        name: Function name returned by the model; returned as-is when it is
            already an absolute path (idempotent).

    Returns:
        A full path starting with ``/``.
    """
    if name.startswith(_PATH_SEP):
        return name
    return _PATH_SEP + name.replace(_NAME_SEP, _PATH_SEP)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The unified response produced by an adapter.

    Attributes:
        message: An ``AssistantMsg`` ready to append to the session.
        tokens_in: Input tokens (0 when the vendor omits them).
        tokens_out: Output tokens (0 when the vendor omits them).
        model: The model name actually used.
    """

    message: AssistantMsg
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


#: Tool declaration: a single element of OpenAI's ``tools`` array.
ToolSpec = dict[str, Any]


class Transport(Protocol):
    """HTTP transport protocol, so tests can inject a fake transport."""

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        """Send a POST and return the JSON response body."""
        ...

    def stream_post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Send a POST and yield each server-sent event payload as it arrives.

        Only the streaming path needs this; transports that cannot stream may
        omit it, in which case the proxy falls back to a single ``post``.
        """
        ...


class Adapter(Protocol):
    """Vendor adapter protocol."""

    name: str

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, Any]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """Project messages into a vendor request body."""
        ...

    def parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        """Parse a vendor response into the unified structure."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible (DeepSeek / Moonshot / local vLLM / OpenAI itself)
# ---------------------------------------------------------------------------


class OpenAIAdapter:
    """Adapter for the OpenAI Chat Completions protocol."""

    name = "openai"

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, Any]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """Project messages and assemble a Chat Completions request body.

        The discovered-tools inventory is injected as a **trailing temporary
        system message** (A11-A13) and never written into history. When ``tools``
        is non-empty it must be declared with the request, otherwise the model
        emits tool calls as plain text.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": to_provider(list(messages), discovered),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # The provider only accepts ^[a-zA-Z0-9_-]+$ function names, so paths
            # must be encoded before going out.
            body["tools"] = [_encode_spec(item) for item in tools]
            body["tool_choice"] = "auto"
        return body

    def parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        """Parse ``choices[0].message``.

        Raises:
            ModelError: The response lacks ``choices`` or is malformed (a
                non-retryable error).
        """
        choices = payload.get("choices")
        if not choices:
            raise ModelError("response is missing choices", provider=self.name, retryable=False)

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
            # Note: DeepSeek's thinking mode requires ``reasoning_content`` to be
            # echoed back verbatim; dropping it makes subsequent requests in the
            # same session fail with 400.
            reasoning_content=raw.get("reasoning_content"),
        )

        usage = payload.get("usage") or {}
        return LLMResponse(
            message=message,
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            model=str(payload.get("model", "")),
        )


class StreamAccumulator:
    """Rebuild a complete reply from OpenAI streaming deltas.

    Streaming sends *fragments*: ``content`` arrives in pieces, and tool calls
    are spread across frames -- the first frame carries ``index``/``id``/``name``
    and later frames append ``arguments`` keyed by that index. This class
    reassembles them so the rest of the system can keep working with a single
    :class:`AssistantMsg`, while :meth:`feed` also reports newly arrived text so
    the UI can render it immediately.

    Attributes:
        reasoner: Whether ``reasoning_content`` is recorded (the caller decides
            this per provider so a non-reasoning model costs nothing).
    """

    def __init__(self, *, reasoner: bool = False) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, dict[str, str]] = {}
        self._model = ""
        self.tokens_in = 0
        self.tokens_out = 0
        self.reasoner = reasoner

    def feed(self, event: dict[str, Any]) -> str:
        """Consume one stream frame and return the text delta it carried.

        Args:
            event: One parsed SSE payload from the provider.

        Returns:
            The newly arrived ``content`` fragment (``""`` when the frame carries
            only tool-call metadata, reasoning, or usage).
        """
        if not self._model:
            self._model = str(event.get("model") or "")

        usage = event.get("usage") or {}
        if usage:
            self.tokens_in = int(usage.get("prompt_tokens", 0) or 0) or self.tokens_in
            self.tokens_out = int(usage.get("completion_tokens", 0) or 0) or self.tokens_out

        choices = event.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}

        reasoning = delta.get("reasoning_content")
        if reasoning and self.reasoner:
            self._reasoning.append(str(reasoning))

        for item in delta.get("tool_calls") or []:
            index = int(item.get("index", 0) or 0)
            slot = self._calls.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            if item.get("id"):
                slot["id"] = str(item["id"])
            function = item.get("function") or {}
            if function.get("name"):
                # Some providers resend the full name instead of a fragment.
                slot["name"] = (
                    str(function["name"])
                    if not slot["name"]
                    else slot["name"] + str(function["name"])
                )
            if function.get("arguments"):
                slot["arguments"] += str(function["arguments"])

        content = delta.get("content")
        if content:
            self._content.append(str(content))
            return str(content)
        return ""

    def build(self) -> LLMResponse:
        """Materialise the accumulated fragments into a normal response."""
        tool_calls = [
            ToolCall(
                id=slot["id"],
                name=decode_tool_name(slot["name"]),
                arguments=slot["arguments"] or "{}",
            )
            for _, slot in sorted(self._calls.items())
        ]
        text = "".join(self._content)
        message = AssistantMsg(
            content=text if text or not tool_calls else None,
            tool_calls=tool_calls,
            reasoning_content="".join(self._reasoning) or None,
        )
        return LLMResponse(
            message=message,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model=self._model,
        )


# ---------------------------------------------------------------------------
# Anthropic style
# ---------------------------------------------------------------------------


class AnthropicAdapter:
    """Adapter for the Anthropic Messages protocol.

    The system prompt is a separate field; ``tool_use`` / ``tool_result`` map to
    ``AssistantMsg.tool_calls`` and ``ToolMsg`` respectively.
    """

    name = "anthropic"

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        discovered: list[dict[str, Any]] | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """Split the system prompt and conversation turns per the Messages protocol."""
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
            system_parts.append(render_discovered(discovered))

        body: dict[str, Any] = {
            "model": model,
            "messages": turns,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # Anthropic tools use input_schema instead of function.parameters
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
        """Parse the ``content`` block array."""
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ModelError("response is missing content", provider=self.name, retryable=False)

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


def render_discovered(items: Sequence[dict[str, Any]]) -> str:
    """Render the discovered-tools inventory as a trailing temporary section.

    Each entry carries its **full parameter schema**, so a tool that has been
    discovered is immediately callable: the model never needs to descend a second
    time just to learn the arguments.

    Only concrete tools appear here -- the cache never holds abstract nodes,
    because they are locations rather than capabilities.

    Note: this text is model-visible, so it intentionally stays in Chinese.
    """
    import json

    lines = ["已发现工具（可直接调用，无需再次下钻）："]
    for item in items:
        path = item.get("path", "")
        description = (item.get("description") or "").strip()
        schema = item.get("schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])

        head = f"- {path}"
        if description:
            head += f" — {description}"
        lines.append(head)
        if props:
            params = []
            for name, spec in props.items():
                kind = (spec or {}).get("type", "any")
                mark = "必填" if name in required else "可选"
                default = (spec or {}).get("default", None)
                if default is not None:
                    shown = json.dumps(default, ensure_ascii=False)
                    extra = f"，默认 {shown}"
                else:
                    extra = ""
                params.append(f"{name}:{kind}（{mark}{extra}）")
            lines.append(f"    参数：{', '.join(params)}")
        else:
            lines.append("    参数：无")
    return "\n".join(lines)


def _safe_json(raw: str) -> Any:
    """Best-effort JSON string to object; returns the input unchanged on failure (J1: no repair)."""
    import json

    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _encode_spec(spec: ToolSpec) -> ToolSpec:
    """Encode the path name in a tool declaration into a provider-acceptable function name.

    Input looks like ``{"type": "function", "function": {"name": "/a/b", ...}}``;
    returns a deep copy with only ``function.name`` replaced, leaving the
    caller's object untouched.
    """
    function = dict(spec.get("function") or {})
    function["name"] = encode_tool_name(str(function.get("name", "")))
    encoded = dict(spec)
    encoded["function"] = function
    return encoded


def _dump_json(value: Any) -> str:
    """Serialize an object to a JSON string, falling back to ``repr`` on failure."""
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class OpenAITransport:
    """A minimal ``httpx``-based transport (OpenAI-compatible endpoints)."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        """Send the request; any non-2xx status raises :class:`ModelError`."""
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

    def stream_post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Stream a chat completion, yielding one parsed event dict per SSE frame.

        The request body must already say ``stream: true``. Frames are parsed
        line by line so the caller can render tokens as they arrive instead of
        waiting for the whole answer -- without this, a long reply appears all at
        once and the UI looks frozen.

        Note:
            ``json`` here is the *request body*, which shadows the module. Frames
            are therefore decoded through :mod:`json`'s function aliased below,
            not through the parameter.

        Raises:
            ModelError: On a non-2xx status, mirroring :meth:`post`.
        """
        import httpx

        decode = _json_loads
        with httpx.stream(
            "POST", url, headers=headers, json=json, timeout=self.timeout
        ) as response:
            if response.status_code >= 500 or response.status_code == 429:
                response.read()
                raise ModelError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    provider="http",
                    retryable=True,
                )
            if response.status_code >= 400:
                response.read()
                raise ModelError(
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    provider="http",
                    retryable=False,
                )
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    yield decode(data)
                except ValueError:
                    continue


# ---------------------------------------------------------------------------
# Reverse proxy
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Attach configuration for a single provider.

    Attributes:
        name: Provider identifier (used by ``ProviderHealth`` and logs).
        base_url: Base URL; the proxy appends the endpoint path when building
            the full URL.
        model: Model name.
        api_key: Secret.
        adapter: Adapter instance.
        endpoint: Concrete path, default ``/chat/completions``.
        headers: Extra request headers.
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
    """Global reverse proxy: rate limiting + circuit breaking + backoff + failover.

    Multiple sessions share one instance, so rate limiting and health state are
    both **global** (K4).

    Attributes:
        providers: Provider configuration sequence; the first is the default.
        rate_window: Global rate window.
        health: Health state per provider.
        transport: HTTP transport; defaults to :class:`OpenAITransport`.
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
            raise ValueError("at least one provider is required")
        for provider in self.providers:
            self.health.setdefault(provider.name, ProviderHealth(provider=provider.name))

    # ------------------------------------------------------------------ calls

    def chat(
        self,
        messages: Sequence[Message],
        *,
        session_id: str,
        discovered: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Perform one model call.

        Args:
            messages: Session messages (the caller guarantees they passed
                ``validate_messages``).
            session_id: Used for usage accounting and rate-limit records (K2).
            discovered: Discovered-tools inventory, injected temporarily at the
                end during projection.
            model: Overrides the default model name.

        Returns:
            The unified response parsed by the adapter.

        Raises:
            ModelError: Every provider failed.
            RateLimitError: The global rate window rejected this request.
        """
        errors: list[str] = []
        for provider in self._candidates():
            health = self.health[provider.name]
            decision = health.can_send(self._clock())
            if decision == "fail_fast":
                errors.append(f"{provider.name}: circuit open (fail_fast)")
                continue
            if decision == "retry_after":
                errors.append(f"{provider.name}: backing off (retry_after)")
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
                # Rate limiting is a global constraint, not a provider fault:
                # re-raise immediately and do not count it against health.
                raise
            except ModelError as exc:
                health.on_failure(self._clock())
                errors.append(str(exc))
                if not exc.retryable:
                    raise

        raise ModelError("all providers failed: " + "; ".join(errors), provider="proxy")

    def _candidates(self) -> list[ProviderConfig]:
        """Provider order: healthy first, circuit-open later.

        Circuit-open providers are still attempted so they can recover.
        """
        return sorted(
            self.providers,
            key=lambda p: 0 if self.health[p.name].state == "healthy" else 1,
        )

    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        session_id: str,
        on_delta: Callable[[str], None] | None = None,
        discovered: list[dict[str, Any]] | None = None,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Perform one model call, reporting text fragments as they arrive.

        The message is *not* assembled incrementally here: a turn may end in tool
        calls rather than text, and callers need one complete message either way.
        ``on_delta`` is therefore purely a preview channel for the UI -- the
        authoritative reply is still the returned :class:`LLMResponse`, which the
        loop appends to history exactly once.

        Streaming falls back to a plain :meth:`chat` when the transport cannot
        stream, so a fake transport in tests keeps working unchanged.

        Args:
            messages: Session messages.
            session_id: For usage accounting and rate limiting.
            on_delta: Called with each newly arrived text fragment.
            discovered: Discovered-tools inventory.
            model: Model override.

        Returns:
            The fully assembled response.
        """
        provider = self._candidates()[0]
        health = self.health[provider.name]
        decision = health.can_send(self._clock())
        if decision != "allow":
            # Do not attempt a stream against an unhealthy provider: the caller
            # gets the same failover behaviour it has today.
            return self.chat(
                messages,
                session_id=session_id,
                discovered=discovered,
                model=model,
                tools=tools,
            )

        body = provider.adapter.build_request(
            messages,
            model=model or provider.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            discovered=discovered,
            tools=tools,
        )
        estimated = _estimate_request_tokens(body)
        if not self.rate_window.try_acquire(session_id, estimated, self._clock()):
            raise RateLimitError(
                f"global rate window rejected the request (session={session_id})"
            )

        headers = {"Content-Type": "application/json", **provider.headers}
        if provider.api_key:
            headers.setdefault("Authorization", f"Bearer {provider.api_key}")
        url = provider.base_url.rstrip("/") + provider.endpoint

        transport = self.transport or OpenAITransport(provider.base_url, provider.api_key)
        stream_post = getattr(transport, "stream_post", None)
        if stream_post is None:
            return self._call_with_retry(
                provider,
                messages,
                session_id=session_id,
                discovered=discovered,
                model=model or provider.model,
                tools=tools,
            )

        accumulator = StreamAccumulator(
            reasoner="reasoning" in provider.model.lower()
        )
        started = False
        try:
            for event in stream_post(url, headers=headers, json={**body, "stream": True}):
                started = True
                fragment = accumulator.feed(event)
                if fragment and on_delta is not None:
                    on_delta(fragment)
        except ModelError:
            if not started:
                # Nothing was emitted yet, so retrying on another provider cannot
                # produce duplicate text.
                health.on_failure(self._clock())
                return self._call_with_retry(
                    provider,
                    messages,
                    session_id=session_id,
                    discovered=discovered,
                    model=model or provider.model,
                    tools=tools,
                )
            raise

        response = accumulator.build()
        self.rate_window.record_actual(
            session_id, response.tokens_in + response.tokens_out, self._clock()
        )
        health.on_success()
        return response

    def _call_with_retry(
        self,
        provider: ProviderConfig,
        messages: Sequence[Message],
        *,
        session_id: str,
        discovered: list[dict[str, Any]] | None,
        model: str,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Rate-limit check and backoff retry against a single provider."""
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
            raise RateLimitError(f"global rate window rejected the request (session={session_id})")

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
    """The global rate window rejected the request (does not count as a provider failure)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="rate", retryable=True)


def _estimate_request_tokens(body: dict[str, Any]) -> int:
    """Rough request token estimate: half the character count of message text."""
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
