"""Message model (ADT-3).

See docs/02-data-structures/03-messages.md.

Type hierarchy::

    Message (ABC)
    ├── SystemMsg        messages[0]; five-section assembly; byte-stable within a session
    ├── UserMsg          user text; must not be empty
    ├── AssistantMsg     model reply; content and tool_calls must not both be empty
    └── ToolMsg          tool result; must match some ToolCall.id
    ToolCall             standalone value object; does not subclass Message

Invariant ownership:

* Per-instance construction constraints (A7/A8/A9) -> ``__post_init__``
* Cross-message sequence constraints (A1/A2/A3/A10) -> ``validate_messages``
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.exceptions import MessageInvariantError

__all__ = [
    "Message",
    "SystemMsg",
    "UserMsg",
    "AssistantMsg",
    "ToolMsg",
    "ToolCall",
    "to_provider",
    "validate_messages",
]

# Path segments are separated by /; ToolCall.name must be a full absolute path (A10)
_PATH_SEGMENT = r"[A-Za-z0-9_\-.]+"
_ABSOLUTE_PATH_PATTERN = rf"^/(?:{_PATH_SEGMENT})(?:/{_PATH_SEGMENT})*$"


class _Base(BaseModel):
    """Common base for all messages: strict mode, no extra fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)

    # Discriminator field. Serialized but not exposed to providers; the
    # projection step builds provider dicts by hand.
    role: str


class SystemMsg(_Base):
    """SystemMsg: exactly one per session, and it must be ``messages[0]`` (A1).

    The five sections are described in the design doc §3.3; assembly is done by
    ``ContextAssembler`` and this class only carries the result.
    """

    role: Literal["system"] = "system"
    content: str


class UserMsg(_Base):
    """UserMsg: user input; must not be an empty string (A9)."""

    role: Literal["user"] = "user"
    content: Annotated[str, Field(min_length=1)]


class ToolCall(BaseModel):
    """ToolCall: a call request returned by the provider.

    A standalone value object, not a Message subtype.

    ``arguments`` keeps the raw JSON string and is not parsed early (J1);
    see the argument-validation design doc for parsing and validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(pattern=_ABSOLUTE_PATH_PATTERN)]
    arguments: str


class AssistantMsg(_Base):
    """AssistantMsg: model reply. ``is_final`` is a derived read-only property (A7/A8)."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Reasoning trace of thinking models (DeepSeek ``reasoning_content``).
    #: This field **must be echoed back verbatim** to the vendor, otherwise
    #: subsequent requests in the same session return 400 ("reasoning_content
    #: in the thinking mode must be passed back"). It carries no history
    #: semantics and is never displayed; it is pass-through only.
    reasoning_content: str | None = None

    @property
    def is_final(self) -> bool:
        """Whether this is the final answer of the turn.

        Note: ``content`` must also be non-blank. Checking only for an empty
        ``tool_calls`` would treat an empty model reply (``content=""`` with no
        calls) as "finished", silently truncating the task -- we have observed
        runs that stopped halfway with an empty reason.
        """
        return not self.tool_calls and bool((self.content or "").strip())

    @model_validator(mode="after")
    def _check_content_or_calls(self) -> AssistantMsg:
        if self.content is None and not self.tool_calls:
            raise ValueError("A7: content and tool_calls must not both be empty")
        return self


class ToolMsg(_Base):
    """ToolMsg: tool execution result.

    ``name`` is an internal field (the tool-tree path) and is not projected
    (A4/A6).
    """

    role: Literal["tool"] = "tool"
    tool_call_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(pattern=_ABSOLUTE_PATH_PATTERN)]
    content: str


Message = Annotated[
    SystemMsg | UserMsg | AssistantMsg | ToolMsg,
    Field(discriminator="role"),
]
"""Union of message types, for list declarations and re-parsing."""


def validate_messages(messages: list[Message]) -> None:
    """Validate sequence-level constraints across messages.

    Constraints covered: A1 (first message is the only SystemMsg), A2
    (``tool_call_id`` is matchable), A3 (``tool_calls`` pair one-to-one with
    ``ToolMsg``), A10 (paths are absolute; already enforced at construction).
    Per-instance constraints (A7/A8/A9) are not re-checked here.

    Args:
        messages: Session message list to validate.

    Raises:
        MessageInvariantError: Any sequence-level constraint is violated;
            ``rule`` carries the constraint identifier.
    """
    if not messages:
        raise MessageInvariantError("A1", "messages must be non-empty and start with SystemMsg")

    if not isinstance(messages[0], SystemMsg):
        raise MessageInvariantError(
            "A1",
            f"messages[0] must be SystemMsg, got {type(messages[0]).__name__}",
        )

    for index, msg in enumerate(messages[1:], start=1):
        if isinstance(msg, SystemMsg):
            raise MessageInvariantError(
                "A1", f"a second SystemMsg appeared at index {index}; only one is allowed"
            )

    pending: dict[str, str] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AssistantMsg) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.id in pending:
                    raise MessageInvariantError(
                        "A2", f"duplicate tool_call id at index {index}: {call.id}"
                    )
                pending[call.id] = call.name
            continue

        if isinstance(msg, ToolMsg):
            expected_name = pending.pop(msg.tool_call_id, None)
            if expected_name is None:
                raise MessageInvariantError(
                    "A2",
                    f"index {index}: tool_call_id={msg.tool_call_id!r} matches no prior ToolCall",
                )
            if expected_name != msg.name:
                raise MessageInvariantError(
                    "A6",
                    f"index {index}: ToolMsg.name={msg.name!r} does not match "
                    f"ToolCall.name={expected_name!r}",
                )

    if pending:
        missing = ", ".join(f"{cid}@{name}" for cid, name in pending.items())
        raise MessageInvariantError(
            "A3", f"unpaired tool_calls (missing ToolMsg): {missing}"
        )


def to_provider(
    messages: list[Message],
    discovered: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Project messages into a provider request body.

    Projection rules are in the design doc §3.6. When ``discovered`` is
    non-empty a temporary ``system`` message is appended at the **end**
    (A11-A13); it is not written into ``messages``, and omitting ``discovered``
    yields a byte-identical result (preserving the KV cache).

    Args:
        messages: Session message list.
        discovered: Already-discovered tools, items shaped like
            ``{"path": ..., "description": ...}``.

    Returns:
        A new list usable directly as the ``messages`` argument of
        ``chat.completions``.
    """
    projected: list[dict[str, object]] = []

    for msg in messages:
        if isinstance(msg, (SystemMsg, UserMsg)):
            projected.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, AssistantMsg):
            item: dict[str, object] = {"role": "assistant", "content": msg.content}
            # Echo only when present, so providers without reasoning content keep
            # a byte-identical request body.
            if msg.reasoning_content:
                item["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in msg.tool_calls
                ]
            projected.append(item)
        elif isinstance(msg, ToolMsg):
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
        else:  # pragma: no cover - the union is closed
            raise TypeError(f"unknown message type: {type(msg).__name__}")

    if discovered:
        lines = ["已发现工具："]
        lines.extend(f"{item['path']} — {item['description']}" for item in discovered)
        projected.append({"role": "system", "content": "\n".join(lines)})

    return projected
