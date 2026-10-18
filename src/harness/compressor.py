"""C3 GlobalCompressor: fold history along CallRecord boundaries.

See docs/01-architecture.md §4.3 (C3 state machine and three hard constraints).

The three hard constraints:

1. **The boundary only falls between CallRecords** -- a compression cut must
   never split an AssistantMsg from the ToolMsg sequence that follows it,
   otherwise A2/A3 break immediately.
2. **The kept segment starts at a complete CallRecord** -- the first message
   after the cut must be an ``AssistantMsg`` (either a CallRecord head or the
   final answer), never an isolated ``ToolMsg`` or a ``UserMsg`` wedged into a
   call sequence.
3. **Each watermark fires once** -- tracked by ``Watermark`` (I5); this module
   only provides ``compress``, and callers must run ``watermark.judge`` before
   passing ``level``.

Decision on where the summary is injected (**must stay consistent with A1**):

    The summary is a ``UserMsg`` placed after ``messages[0]`` (the ``SystemMsg``)
    and **before the kept segment**, with the fixed prefix
    ``[上下文摘要]``.

    Rationale:

    * A1 requires ``messages[0]`` to be the ``SystemMsg`` **and the only one**,
      so a new ``SystemMsg`` cannot carry the summary;
    * The summary describes "earlier conversation history" and is semantically
      user-side input to the session rather than a system requirement; keeping
      it in a ``UserMsg`` avoids polluting the system-prompt prefix (I1);
    * ``ToolMsg`` must bind to a ``ToolCall.id`` (A2) and therefore cannot carry
      a summary;
    * The ``UserMsg`` content is non-empty, satisfying A9, and it sits before
      the complete ``CallRecord`` that starts the kept segment, never inserted
      inside an unclosed ``tool_calls`` sequence (I6).

    If a summary already exists (second compression), the old summary is merged
    into the new summary text instead of accumulating.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.watermark import AUTO_2, Watermark

__all__ = [
    "SUMMARY_PREFIX",
    "CallRecord",
    "SplitResult",
    "default_keep_records",
    "split_call_records",
    "compress",
]

#: Fixed prefix of the summary message; makes historical summaries identifiable
#: and mergeable.
#: Note: this is model-visible text, so it intentionally stays in Chinese.
SUMMARY_PREFIX = "[上下文摘要]"

#: ``summarizer`` signature: ``(text) -> summary``.
Summarizer = Callable[[str], str]

#: Unpaired tail: structurally incomplete messages (normally empty).
Tail = list[Message]


@dataclass(frozen=True, slots=True)
class CallRecord:
    """The smallest indivisible unit.

    Attributes:
        messages: Every message in this record, in original order.
        call_ids: Ids of ``AssistantMsg.tool_calls`` in this record, in original
            order.
    """

    messages: tuple[Message, ...]

    @property
    def call_ids(self) -> tuple[str, ...]:
        """Every tool_call id in the record."""
        return tuple(call.id for m in self.messages for call in _calls_of(m))

    def text(self) -> str:
        """Render the record as plain text suitable for the summarizer."""
        return render_segment(self.messages)


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Result of ``split_call_records``.

    Attributes:
        system: The ``messages[0]`` SystemMsg (guaranteed by A1).
        head: Messages after ``system`` and before the first ``AssistantMsg``
            (``UserMsg``, historical summaries, ...); all can be compressed.
        records: The intact CallRecord sequence.
        tail: Unpaired tail (trailing calls without a matching ``ToolMsg``, or
            dangling ToolMsgs).
    """

    system: SystemMsg
    head: list[Message]
    records: list[CallRecord]
    tail: Tail

    @property
    def records_messages(self) -> list[Message]:
        """Flatten every record into a message list."""
        return [m for record in self.records for m in record.messages]


def _calls_of(message: Message) -> list[ToolCall]:
    if isinstance(message, AssistantMsg):
        return list(message.tool_calls)
    return []


def split_call_records(messages: Sequence[Message]) -> SplitResult:
    """Split a valid message sequence into ``SystemMsg + head + CallRecord* + tail``.

    Args:
        messages: A message sequence satisfying A1/A2/A3 (``SystemMsg`` first).

    Returns:
        ``SplitResult``.

    Raises:
        ValueError: ``messages`` is empty or does not start with ``SystemMsg``
            (A1 precondition).
    """
    if not messages:
        raise ValueError("A1: messages must not be empty")
    first = messages[0]
    if not isinstance(first, SystemMsg):
        raise ValueError(f"A1: messages[0] must be SystemMsg, got {type(first).__name__}")
    system = first

    rest = list(messages[1:])
    head: list[Message] = []
    records: list[CallRecord] = []
    tail: Tail = []
    index = 0

    while index < len(rest):
        msg = rest[index]
        if isinstance(msg, (UserMsg, SystemMsg)):
            # A user message sitting between two CallRecords (new task or
            # follow-up) is itself compressible leading text, so it goes into
            # head; it never splits a call sequence, so hard constraint 1 holds.
            # A SystemMsg here means A1 is already broken; leave it to
            # validate_messages to report.
            head.append(msg)
            index += 1
            continue

        if isinstance(msg, AssistantMsg):
            record: list[Message] = [msg]
            index += 1
            if msg.tool_calls:
                expected = {c.id for c in msg.tool_calls}
                while index < len(rest) and isinstance(rest[index], ToolMsg):
                    tool = rest[index]
                    if tool.tool_call_id not in expected:
                        break
                    expected.discard(tool.tool_call_id)
                    record.append(tool)
                    index += 1
                    if not expected:
                        break
                if expected:
                    # Unpaired: fold the whole record and everything after it
                    # into the tail.
                    tail.extend(record)
                    tail.extend(rest[index:])
                    index = len(rest)
                    break
            records.append(CallRecord(messages=tuple(record)))
            continue

        # Dangling ToolMsg (no matching ToolCall): structurally invalid; keep it
        # in the tail so the validator reports A2.
        tail.append(msg)
        index += 1

    return SplitResult(system=system, head=head, records=records, tail=tail)


def _render_message(message: Message) -> str:
    """Render one message into summarizer input text, tagging tool paths via ``ToolMsg.name``."""
    if isinstance(message, SystemMsg):
        return "[system]\n" + message.content
    if isinstance(message, UserMsg):
        return "[user]\n" + message.content
    if isinstance(message, AssistantMsg):
        lines: list[str] = []
        if message.content:
            lines.append(message.content)
        for call in message.tool_calls:
            lines.append(f"[assistant:call {call.name}] {call.arguments}")
        return "\n".join(lines)
    if isinstance(message, ToolMsg):
        # C3 requires the summary to tag the tool path with ToolMsg.name
        return f"[tool:{message.name}]\n{message.content}"
    raise TypeError(f"unknown message type: {type(message).__name__}")  # pragma: no cover


def render_segment(messages: Sequence[Message]) -> str:
    """Render a segment of messages for the summarizer."""
    return "\n\n".join(_render_message(m) for m in messages)


def default_keep_records(capacity: int, level: int) -> int:
    """Compute how many complete CallRecords to keep for a watermark.

    Keep 4 records at the 50% watermark and 2 at 75%. A higher watermark means
    longer history and allows more aggressive folding; this matches the design
    doc's "keep the most recent K".
    """
    if level >= int(round(AUTO_2 * 100)):
        return 2
    return 4


def _summary_user_msg(previous_summary: str, new_summary: str) -> UserMsg:
    body = new_summary.strip() or "（摘要为空）"
    if not previous_summary:
        return UserMsg(content=f"{SUMMARY_PREFIX}\n{body}")
    return UserMsg(content=f"{SUMMARY_PREFIX}\n{previous_summary.strip()}\n{body}")


def _find_previous_summary(messages: Sequence[Message]) -> tuple[int | None, str]:
    for index, message in enumerate(messages):
        if isinstance(message, UserMsg) and message.content.startswith(SUMMARY_PREFIX):
            return index, message.content[len(SUMMARY_PREFIX) :].strip()
    return None, ""


def compress(
    messages: Sequence[Message],
    level: int,
    summarizer: Summarizer,
    *,
    keep_records: int | None = None,
    watermark: Watermark | None = None,
) -> list[Message]:
    """Fold history along CallRecord boundaries and return a new message list.

    Args:
        messages: Current message sequence (must satisfy A1/A2/A3).
        level: Compression watermark, ``50`` or ``75`` (matching
            ``Watermark.compressed_at``).
        summarizer: Injected summarizer ``(text) -> summary``; this module never
            calls the model directly.
        keep_records: Number of complete CallRecords to keep; defaults derived
            from the watermark when omitted.
        watermark: When given, enforces I5 ("each watermark fires once") and
            records the firing.

    Returns:
        The rebuilt list ``[SystemMsg, (UserMsg summary), *kept, *tail]``, already
        validated by ``validate_messages`` (A1/A2/A3).

    Raises:
        ValueError: The watermark already fired (I5), or the input sequence is
            itself invalid.

    Note:
        When the compressible segment is empty (nothing to fold) the input is
        returned as a shallow copy and **no** watermark is recorded, so a no-op
        compression does not consume a watermark opportunity.
    """
    if watermark is not None and watermark.is_compressed(level):
        raise ValueError(f"I5: watermark {level} already compressed; cannot fire again")

    split = split_call_records(list(messages))
    # Validate the input first so we never run an expensive summary on bad input
    validate_messages(list(messages))

    keep = default_keep_records(watermark.capacity if watermark else 0, level)
    if keep_records is not None:
        keep = max(0, keep_records)

    compressible = len(split.records) - keep
    if compressible <= 0:
        return list(messages)

    # ① Boundaries fall only between CallRecords: compressed = head + records[:n]
    dropped_records = split.records[:compressible]
    kept_records = split.records[compressible:]
    # ② The kept segment starts at a complete CallRecord, so kept_records[0]'s
    # first message must be an AssistantMsg.
    if kept_records:
        first_kept = kept_records[0].messages[0]
        if not isinstance(first_kept, AssistantMsg):  # pragma: no cover - split guarantees this
            raise ValueError("C3: the kept segment must start with a complete CallRecord (AssistantMsg)")

    prior_index, prior_summary = _find_previous_summary(split.head)
    older = [m for i, m in enumerate(split.head) if i != prior_index]
    to_summarize = [*older, *(m for r in dropped_records for m in r.messages)]
    if not to_summarize:
        return list(messages)

    if not kept_records and not split.tail:
        # Keeping only the summary would leave no "recent message" in history at
        # all; in that case skip compression and let the upper layer
        # (StepController) wrap up at 85%.
        return list(messages)

    new_summary = summarizer(render_segment(to_summarize))
    summary_msg = _summary_user_msg(prior_summary, new_summary)
    rebuilt: list[Message] = [split.system, summary_msg]
    rebuilt.extend(m for r in kept_records for m in r.messages)
    rebuilt.extend(split.tail)

    validate_messages(rebuilt)

    if watermark is not None:
        watermark.mark_compressed(level)

    return rebuilt
