"""C2 TokenEstimator + watermark decision (Watermark / JudgeResult).

See:

* docs/02-data-structures/08-runtime-state.md (Watermark / JudgeResult)
* docs/01-architecture.md §4.3 (watermark definitions), §6 I5

Watermark definitions::

    AUTO_1      = 0.50 x CAPACITY
    AUTO_2      = 0.75 x CAPACITY
    SESSION_END = 0.85 x CAPACITY   # no compression; wrap up and suggest a new session

Decision order (hard constraint I5): check ``SESSION_END`` first (>=0.85 wraps
up without compressing), then ``AUTO_2`` (>=0.75), then ``AUTO_1`` (>=0.50).
Each watermark fires at most once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orange.message import AssistantMsg, Message, SystemMsg, ToolMsg, UserMsg

__all__ = [
    "AUTO_1",
    "AUTO_2",
    "SESSION_END",
    "JudgeResult",
    "JudgeAction",
    "Watermark",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "judge",
]

#: Level-1 auto-compression watermark (fires on first crossing 50%).
AUTO_1 = 0.50
#: Level-2 auto-compression watermark (fires on first crossing 75%).
AUTO_2 = 0.75
#: Session wrap-up watermark (>=85% wraps up without compressing).
SESSION_END = 0.85

#: Rough tokens-per-character coefficient (no tokenizer; see doc §1.2 non-goals).
CHARS_PER_TOKEN = 2.0

JudgeAction = Literal["normal", "compress_1", "compress_2", "session_end"]

#: Watermark -> key written into ``compressed_at`` when it fires (integer percent).
_WATERMARK_KEY: dict[str, int] = {"compress_1": 50, "compress_2": 75}


def estimate_tokens(content: str) -> int:
    """Estimate token count as ``chars / 2.0`` (round up, min 1 for non-empty text)."""
    return int(len(content) / CHARS_PER_TOKEN + 0.999_999_999)


def estimate_message_tokens(message: Message) -> int:
    """Estimate the token count of a single message.

    Besides ``content``, a ``ToolCall``'s ``name`` and ``arguments`` also travel
    in the provider request body, so they are counted too; otherwise
    tool-call-heavy history would be underestimated.
    """
    if isinstance(message, (SystemMsg, UserMsg, ToolMsg)):
        return estimate_tokens(message.content)

    if isinstance(message, AssistantMsg):
        total = estimate_tokens(message.content or "")
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(call.arguments)
        return total

    raise TypeError(f"unknown message type: {type(message).__name__}")  # pragma: no cover


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate the token count of a whole history."""
    return sum(estimate_message_tokens(m) for m in messages)


class Watermark(BaseModel):
    """Context watermark record (group F).

    Attributes:
        capacity: Model context limit (tokens).
        compressed_at: Watermarks already compressed (``50`` / ``75``).
            **F1: grow-only; the same watermark is never added twice.**
        last_estimate: Most recent token estimate.
    """

    model_config = ConfigDict(validate_assignment=False)

    capacity: int = Field(gt=0)
    compressed_at: set[int] = Field(default_factory=set)
    last_estimate: int = Field(default=0, ge=0)

    def is_compressed(self, level: int) -> bool:
        """Whether compression has already fired for this watermark."""
        return level in self.compressed_at

    def mark_compressed(self, level: int) -> bool:
        """Record that a watermark has been compressed.

        F1: the set only grows; re-marking the same watermark is a no-op.

        Returns:
            ``True`` if newly added, ``False`` if already marked.
        """
        if level in self.compressed_at:
            return False
        self.compressed_at.add(level)
        return True

    def update_estimate(self, estimate: int) -> float:
        """Store the latest estimate and return the occupancy ratio."""
        self.last_estimate = max(0, estimate)
        return self.ratio

    @property
    def ratio(self) -> float:
        """Occupancy ratio of the latest estimate relative to capacity."""
        return self.last_estimate / self.capacity

    def as_set(self) -> frozenset[int]:
        """Read-only snapshot for external exposure (prevents callers shrinking it)."""
        return frozenset(self.compressed_at)


class JudgeResult(BaseModel):
    """Watermark decision result.

    Attributes:
        action: ``normal`` / ``compress_1`` / ``compress_2`` / ``session_end``.
        ratio: Estimated occupancy (``estimate / capacity``).
    """

    model_config = ConfigDict(frozen=True)

    action: JudgeAction
    ratio: float = Field(ge=0.0)

    @property
    def level(self) -> int | None:
        """Compression watermark for this result (``50`` / ``75``); ``None`` if not needed."""
        return _WATERMARK_KEY.get(self.action)


def judge(
    messages: list[Message],
    capacity: int,
    *,
    watermark: Watermark | None = None,
    estimate: int | None = None,
) -> JudgeResult:
    """Decide the current context watermark.

    Args:
        messages: Current session message list.
        capacity: Model context limit.
        watermark: Watermark record; when given, ``last_estimate`` is updated
            in place. It **does not** record compression: ``compressed_at`` is
            written by ``compressor.compress()`` once compression actually
            happens, keeping "decide" and "execute" decoupled. Omit it for a
            stateless decision (every watermark fires).
        estimate: Precomputed token estimate; computed from ``messages`` if omitted.

    Returns:
        ``JudgeResult``; ``ratio`` is always ``estimate / capacity``.

    Raises:
        ValueError: ``capacity`` is not positive.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")

    if estimate is None:
        estimate = estimate_messages_tokens(messages)

    ratio = (
        watermark.update_estimate(estimate) if watermark is not None else estimate / capacity
    )

    # Note: this function **only decides, never records**. Recording belongs to
    # compress(), which does the real work -- otherwise compress() would
    # immediately fail its own I5 duplicate-trigger check and never run.
    return _decide(ratio, watermark)


def _decide(ratio: float, watermark: Watermark | None) -> JudgeResult:
    """Decide in I5 order: 0.75 -> 0.50 -> session_end.

    Note: the wrap-up check must come **last**. If it came first, a single step
    jumping from <50% straight past 85% would swallow both compression
    opportunities at once and compression would never fire. When an
    un-compressed watermark is still available, give compression a chance
    first; the ratio drops afterwards and the next round decides wrap-up.
    """
    if ratio >= AUTO_2 and not _already(watermark, AUTO_2):
        return JudgeResult(action="compress_2", ratio=ratio)

    if ratio >= AUTO_1 and not _already(watermark, AUTO_1):
        return JudgeResult(action="compress_1", ratio=ratio)

    if ratio >= SESSION_END:
        # >=85% with no compression watermark left -> wrap up without compressing
        return JudgeResult(action="session_end", ratio=ratio)

    return JudgeResult(action="normal", ratio=ratio)


def _already(watermark: Watermark | None, threshold: float) -> bool:
    """Whether this threshold has already fired (always False without a watermark)."""
    if watermark is None:
        return False
    return watermark.is_compressed(int(round(threshold * 100)))
