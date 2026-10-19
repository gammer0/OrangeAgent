"""Ingress dual channel (group G).

See docs/02-data-structures/01-ingress.md.

Structure::

    Inbox
    ├── control: deque[InterruptRequest]   immediate channel, unbounded
    ├── chat: deque[UserInput]             pending channel, FIFO, capped at 100
    └── lock: threading.Lock               guards both channels

This module implements **queue semantics only** and does not implement
``AgentLoop``:

* What a safe point consumes is decided by the caller; this module only offers
  the two semantic entry points ``drain_at_full_safe_point`` (① / ③, G4) and
  ``take_control_at_sub_batch`` (②, G2).
* Drop notifications go through the injected ``on_drop`` callback, avoiding a
  hard dependency on the ``trace`` module (G8 / G12 require writing a
  ``TraceEvent``; the upper layer injects that writing logic here).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CHAT_CAPACITY",
    "INTERRUPT_REASONS",
    "DropReason",
    "DropNotice",
    "UserInput",
    "InterruptRequest",
    "Inbox",
]

#: Capacity limit of the ``chat`` channel (design doc §1.3)
CHAT_CAPACITY = 100

#: The three legal values of ``reason`` (G11)
INTERRUPT_REASONS: tuple[str, str, str] = ("user_abort", "session_end", "timeout")

#: Enqueue drop reasons: G8 (overflow) / G12 (empty text)
DropReason = Literal["overflow", "empty_text"]

#: Drop-notification callback signature. The upper layer writes a ``TraceEvent``.
DropCallback = Callable[["DropNotice"], None]


def _now() -> float:
    return time.time()


@dataclass(frozen=True, slots=True)
class UserInput:
    """An element of the ``chat`` channel.

    Attributes:
        text: Raw text, never rewritten; must be non-empty (empty strings are
            rejected in ``Inbox.put_chat``, G12).
        channel: Source channel, ``stdin`` (terminal) or ``ws`` (WebSocket).
        ts: Unix timestamp when enqueued.
    """

    text: str
    channel: Literal["stdin", "ws"]
    ts: float = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class InterruptRequest:
    """An interrupt signal on the ``control`` channel.

    Invariants (G11):

    * ``reason`` may only be ``user_abort`` / ``session_end`` / ``timeout``;
    * except for ``user_abort``, ``text`` must be ``None``.

    Attributes:
        reason: Interrupt source. ``user_abort`` is written by ``Reader``,
            ``session_end`` by ``StepController``, ``timeout`` by ``Executor``.
        text: Attached text; when non-empty it is inserted as a ``UserMsg``
            after all placeholders are backfilled (3e) (G3).
        ts: Unix timestamp when enqueued.
    """

    reason: Literal["user_abort", "session_end", "timeout"]
    text: str | None = None
    ts: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        # Runtime belt-and-braces: Literal only affects static checking, but
        # calls that bypass type checking must be rejected too.
        if self.reason not in INTERRUPT_REASONS:
            raise ValueError(
                f"G11: reason must be one of {INTERRUPT_REASONS}, got {self.reason!r}"
            )
        if self.reason != "user_abort" and self.text is not None:
            raise ValueError(
                f"G11: text must be None when reason={self.reason!r}, got {self.text!r}"
            )


@dataclass(frozen=True, slots=True)
class DropNotice:
    """Notification of one dropped enqueue, for the upper layer to build a ``TraceEvent``.

    Attributes:
        channel: Channel of the dropped message; always ``"chat"`` (``control``
            is unbounded).
        reason: ``overflow`` (G8) or ``empty_text`` (G12).
        dropped: The dropped element; ``None`` for ``empty_text``.
        text: Text of the dropped message; the original empty text for
            ``empty_text``.
        ts: Time the drop happened.
    """

    channel: Literal["chat"]
    reason: DropReason
    dropped: UserInput | None
    text: str
    ts: float = field(default_factory=_now)


class Inbox:
    """Session-level dual-channel inbox.

    Threading model (G5 / G6):

    * Writers may be multiple (``Reader`` / ``StepController`` / ``Executor``);
      all hold ``lock``.
    * The consumer is the main thread only, and also holds ``lock``.
    * Critical sections **only touch queues**; never call tools or the model
      while holding the lock.
    """

    def __init__(
        self,
        capacity: int = CHAT_CAPACITY,
        on_drop: DropCallback | None = None,
    ) -> None:
        """Create an empty inbox.

        Args:
            capacity: Capacity limit of the ``chat`` channel, default 100.
            on_drop: Drop-notification callback (G8 / G12). Invoked **outside**
                the lock so that the upper layer's ``TraceEvent`` IO cannot
                block writers.
        """
        self._control: deque[InterruptRequest] = deque()
        self._chat: deque[UserInput] = deque()
        self._lock = threading.Lock()
        self._capacity = capacity
        self._on_drop = on_drop

    # ------------------------------------------------------------------
    # Read-only observation (locked; used by the registry eviction check)
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Capacity limit of the ``chat`` channel."""
        return self._capacity

    def is_empty(self) -> bool:
        """Whether both channels are empty (locked)."""
        with self._lock:
            return not self._control and not self._chat

    def __len__(self) -> int:
        """Total number of elements across both channels (locked)."""
        with self._lock:
            return len(self._control) + len(self._chat)

    def __repr__(self) -> str:
        with self._lock:
            return f"Inbox(control={len(self._control)}, chat={len(self._chat)})"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def put_chat(self, item: UserInput) -> bool:
        """Write to the ``chat`` channel (FIFO).

        Empty text is rejected and notified (G12); when over capacity the
        oldest element is dropped and notified (G8).

        Args:
            item: User input.

        Returns:
            Whether the item was enqueued.
        """
        notice: DropNotice | None = None
        accepted = True

        with self._lock:
            if not item.text:
                accepted = False
                notice = DropNotice(
                    channel="chat", reason="empty_text", dropped=None, text=item.text
                )
            else:
                if len(self._chat) >= self._capacity:
                    dropped = self._chat.popleft()
                    notice = DropNotice(
                        channel="chat", reason="overflow", dropped=dropped, text=dropped.text
                    )
                self._chat.append(item)

        self._emit(notice)
        return accepted

    def put_control(self, item: InterruptRequest) -> None:
        """Write to the ``control`` channel (unbounded)."""
        with self._lock:
            self._control.append(item)

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def take_control(self) -> InterruptRequest | None:
        """Take the first ``control`` signal and clear the rest (G7).

        Returns:
            The first signal, or ``None`` when the channel is empty.
        """
        with self._lock:
            if not self._control:
                return None
            first = self._control.popleft()
            self._control.clear()
            return first

    def drain_chat(self) -> list[UserInput]:
        """Take every ``chat`` message, preserving FIFO order."""
        with self._lock:
            items = list(self._chat)
            self._chat.clear()
            return items

    # ------------------------------------------------------------------
    # Safe-point semantic wrappers (no AgentLoop implementation)
    # ------------------------------------------------------------------

    def drain_at_full_safe_point(self) -> tuple[InterruptRequest | None, list[UserInput]]:
        """Full safe-point consumption ① after the LLM returns / ③ after the whole batch (G4).

        When both channels have messages, ``control`` is handled first, then
        ``chat`` is drained.

        Returns:
            ``(control signal or None, all chat messages)``.
        """
        interrupt = self.take_control()
        return interrupt, self.drain_chat()

    def take_control_at_sub_batch(self) -> InterruptRequest | None:
        """Sub-batch safe point between groups ② (G2).

        Takes ``control`` only; ``chat`` messages stay queued, neither consumed
        nor dropped.
        """
        return self.take_control()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(self, notice: DropNotice | None) -> None:
        """Deliver the drop notification outside the lock."""
        if notice is not None and self._on_drop is not None:
            self._on_drop(notice)
