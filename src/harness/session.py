"""Session layer (group H).

See docs/02-data-structures/02-session-layer.md.

Structure::

    SessionRegistry
    ├── sessions: dict[str, Session]
    ├── lock: threading.Lock
    └── policy: EvictionPolicy

Cross-branch coupling (important)
---------------------------------
The session layer must hold three session-level structures that belong to
**other groups**: ``Watermark`` / ``TreeQueryState`` / ``DiscoveredTools``. To
avoid importing context-layer (groups F / B) modules into this branch, the module
defines a **minimal placeholder** for ``Watermark`` and **protocol interfaces**
(``Protocol``) for ``TreeQueryState`` / ``DiscoveredTools``:

* The ``Watermark`` placeholder only carries the three fields from doc §8.1
  (``capacity`` / ``compressed_at`` / ``last_estimate``) plus serialization
  helpers; it does **not** include watermark decision logic (F1).
* ``TreeQueryState`` / ``DiscoveredTools`` describe field shapes via
  ``Protocol``, so no context/capability implementation needs importing at
  runtime -- any equivalent object can be passed during assembly.

``SessionTreeQuery.clear_delivered`` / ``DiscoveredTools.clear`` are not
specified by name in the group-B docs, so ``_reset_discovered`` uses a lenient
strategy: "try ``clear()`` first, otherwise construct a new instance". This keeps
the H8 / H14 semantics achievable without hard-coding the group-B API.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from harness.inbox import Inbox
from harness.message import Message

__all__ = [
    "MAX_RESIDENT",
    "IDLE_TIMEOUT_S",
    "SWEEP_INTERVAL_S",
    "SessionStatus",
    "Watermark",
    "TreeQueryStateProtocol",
    "DiscoveredToolsProtocol",
    "Session",
    "EvictionPolicy",
    "SessionRecord",
    "SessionRegistry",
    "new_session_id",
]

#: Eviction policy defaults (design doc §2.1)
MAX_RESIDENT = 20
IDLE_TIMEOUT_S = 1800
SWEEP_INTERVAL_S = 60

SessionStatus = Literal["active", "idle", "ended"]


def new_session_id() -> str:
    """Generate a globally unique ``session_id`` (the routing key)."""
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
# Minimal placeholders for cross-branch dependencies
# ----------------------------------------------------------------------


@dataclass
class Watermark:
    """**Minimal placeholder** for the context watermark record (same fields as doc §8.1).

    The real implementation lives in the context layer (group F); this branch
    keeps only the data structure to avoid cross-branch import coupling. Watermark
    decisions (``JudgeResult``) and the grow-only semantics of ``compressed_at``
    (F1) are not implemented here.
    """

    capacity: int = 0
    compressed_at: set[int] = field(default_factory=set)
    last_estimate: int = 0

    def to_json(self) -> dict[str, Any]:
        """Serialize into a JSON-compatible dict writable to ``sessions.watermark``."""
        return {
            "capacity": self.capacity,
            "compressed_at": sorted(self.compressed_at),
            "last_estimate": self.last_estimate,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Watermark | None:
        """Deserialize from ``sessions.watermark``; ``None`` is returned unchanged."""
        if data is None:
            return None
        raw_compressed = data.get("compressed_at") or []
        return cls(
            capacity=int(data.get("capacity", 0)),
            compressed_at={int(level) for level in raw_compressed},
            last_estimate=int(data.get("last_estimate", 0)),
        )


@runtime_checkable
class TreeQueryStateProtocol(Protocol):
    """Structural interface of ``TreeQueryState`` (group B).

    ``delivered`` must be cleared after a restore or unload (H8 / H14), so this
    layer only depends on that field being mutable.
    """

    delivered: set[str]


@runtime_checkable
class DiscoveredToolsProtocol(Protocol):
    """Structural interface of ``DiscoveredTools`` (group B).

    The inventory must be cleared after a restore or unload (H8 / H14).
    Implementations should provide ``clear()``; otherwise ``reset_discovered``
    falls back to constructing a new instance.
    """

    entries: dict[str, Any]

    def clear(self) -> None:  # pragma: no cover - structural declaration
        ...


def reset_tree_query(tree_query: TreeQueryStateProtocol) -> None:
    """Clear ``TreeQueryState.delivered`` (H8 / H14)."""
    tree_query.delivered.clear()


def reset_discovered(discovered: DiscoveredToolsProtocol) -> Any:
    """Clear ``DiscoveredTools`` (H8 / H14).

    Prefers ``clear()``; if the implementation lacks it, returns a new empty
    instance for the caller to swap in. The return value is only used by the
    "fallback construction" path; the normal path returns ``None``.
    """
    clear = getattr(discovered, "clear", None)
    if callable(clear):
        clear()
        return None

    # Fallback path: construct a same-type instance and best-effort clear its
    # entries container.
    replacement = type(discovered)()
    entries = getattr(replacement, "entries", None)
    if isinstance(entries, dict):
        entries.clear()
    return replacement


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------


@dataclass
class Session:
    """Runtime state of a single session.

    Isolation boundary (H1): ``messages`` / ``inbox`` / ``watermark`` /
    ``tree_query`` must all be session-exclusive objects and must never be shared
    by reference across sessions.

    ``tree_version`` is fixed at creation (H15): later tool-tree mounts/unmounts
    only affect new sessions.
    """

    session_id: str
    working_dir: str
    messages: list[Message] = field(default_factory=list)
    inbox: Inbox = field(default_factory=Inbox)
    watermark: Watermark = field(default_factory=Watermark)
    tree_query: TreeQueryStateProtocol = field(default_factory=lambda: _default_tree_query())
    tree_version: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    status: SessionStatus = "idle"

    def touch(self) -> None:
        """Refresh ``last_activity``, used for eviction ordering."""
        self.last_activity = time.time()

    def busy(self) -> None:
        """Mark the session as reasoning.

        ``active`` sessions are never unloaded by ``sweep`` (H10).
        """
        self.status = "active"
        self.touch()

    def release(self) -> None:
        """Reasoning finished; the session returns to ``idle`` and becomes evictable again."""
        if self.status != "ended":
            self.status = "idle"
        self.touch()


def _default_tree_query() -> TreeQueryStateProtocol:
    """Build an empty default ``TreeQueryState`` placeholder (avoids importing group B)."""

    class _DefaultTreeQuery:
        def __init__(self) -> None:
            self.delivered: set[str] = set()

    return _DefaultTreeQuery()


@dataclass
class EvictionPolicy:
    """Session eviction policy (design doc §2.1 / §2.4)."""

    max_resident: int = MAX_RESIDENT
    idle_timeout_s: int = IDLE_TIMEOUT_S
    sweep_interval_s: int = SWEEP_INTERVAL_S


@dataclass
class SessionRecord:
    """A persisted row of the ``sessions`` table (SQLite projection)."""

    session_id: str
    working_dir: str
    created_at: float
    updated_at: float
    status: SessionStatus
    watermark: Watermark | None = None


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


class SessionRegistry:
    """Multi-session registry.

    Threading model (H2): every read and write holds ``self._lock``; critical
    sections perform **dict operations only** and must never contain LLM calls or
    tool execution.
    """

    def __init__(self, policy: EvictionPolicy | None = None) -> None:
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.policy = policy or EvictionPolicy()
        # Background sweeper thread (S2): lazily started, off by default
        self._sweeper: threading.Thread | None = None
        self._sweeper_stop = threading.Event()
        self._sweeper_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Basic reads and writes
    # ------------------------------------------------------------------

    def add(self, session: Session) -> Session:
        """Register a session (locked). An existing id is overwritten."""
        with self.lock:
            self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Look up a session by id (locked); returns ``None`` when absent."""
        with self.lock:
            return self.sessions.get(session_id)

    def remove(self, session_id: str) -> Session | None:
        """Remove a session from memory (locked) and return it; ``None`` when absent.

        Drops the in-memory reference only; the database is untouched (H13).
        """
        with self.lock:
            return self.sessions.pop(session_id, None)

    def end(self, session_id: str) -> Session | None:
        """Mark ``ended`` and remove from memory immediately (H12).

        Returns:
            The removed session object; ``None`` when absent.
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            session.status = "ended"
            return self.sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self.lock:
            return len(self.sessions)

    def __contains__(self, session_id: object) -> bool:
        with self.lock:
            return session_id in self.sessions

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evictable(self, session: Session, now: float) -> bool:
        """Whether the session can be unloaded.

        Evictable means inactive (``idle``) with no pending messages (H10).
        """
        return session.status == "idle" and session.inbox.is_empty()

    def sweep(self, now: float | None = None) -> list[str]:
        """Run one eviction scan (design doc §2.4).

        Three-stage rule set, all inside **one lock** (H11: "unloading must hold
        the lock"):

        1. Remove every ``ended`` session under the lock (H12);
        2. **Idle timeout**: unload any evictable session with
           ``now - last_activity > idle_timeout_s``. This is the only reclaim path
           that does not depend on session count;
        3. **Capacity overflow**: if the resident count still exceeds
           ``max_resident``, keep unloading evictable sessions in ascending
           ``last_activity`` order until the limit is met or nothing is evictable.

        Evictable (H10) = ``status == "idle"`` and empty ``inbox``. ``active``
        sessions (currently reasoning) are never unloaded, so the resident count
        may briefly exceed ``max_resident`` -- the limit is a target, not a hard
        constraint.

        Args:
            now: Logical current time (seconds). Defaults to ``time.time()``;
                tests may inject it.

        Returns:
            The list of unloaded (including ``ended`` removals) ``session_id``
            values.
        """
        current = time.time() if now is None else now
        evicted: list[str] = []

        with self.lock:
            for session_id in [sid for sid, s in self.sessions.items() if s.status == "ended"]:
                self.sessions.pop(session_id, None)
                evicted.append(session_id)

            # 1) Idle-timeout eviction (independent of capacity)
            timed_out = [
                s
                for s in self.sessions.values()
                if self._evictable(s, current)
                and current - s.last_activity > self.policy.idle_timeout_s
            ]
            for session in timed_out:
                self.sessions.pop(session.session_id, None)
                evicted.append(session.session_id)

            # 2) Capacity-overflow eviction (LRU)
            overflow = len(self.sessions) - self.policy.max_resident
            if overflow > 0:
                candidates = sorted(
                    (s for s in self.sessions.values() if self._evictable(s, current)),
                    key=lambda s: s.last_activity,
                )
                for session in candidates[:overflow]:
                    self.sessions.pop(session.session_id, None)
                    evicted.append(session.session_id)

        return evicted

    def session_ids(self) -> list[str]:
        """All session ids currently resident in memory (snapshot, locked)."""
        with self.lock:
            return list(self.sessions.keys())

    # ------------------------------------------------------------------
    # Periodic sweeping (S2)
    # ------------------------------------------------------------------

    def start_sweeper(self) -> None:
        """Start the background sweeper thread.

        Calls ``sweep()`` every ``policy.sweep_interval_s``.

        Idempotent: returns immediately when already running. The thread is a
        daemon so it never blocks process exit; ``stop_sweeper()`` wakes and ends
        it promptly (``Event.wait`` is interruptible, so shutdown never waits out
        a full interval).
        """
        with self._sweeper_lock:
            if self._sweeper is not None and self._sweeper.is_alive():
                return

            self._sweeper_stop.clear()
            thread = threading.Thread(
                target=self._sweeper_loop,
                name="harness-session-sweeper",
                daemon=True,
            )
            self._sweeper = thread
            thread.start()

    def stop_sweeper(self, timeout: float = 2.0) -> None:
        """Signal the background sweeper to exit and wait for it (idempotent)."""
        with self._sweeper_lock:
            thread = self._sweeper
            self._sweeper = None
            self._sweeper_stop.set()

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def sweeping(self) -> bool:
        """Whether the background sweeper thread is running."""
        with self._sweeper_lock:
            return self._sweeper is not None and self._sweeper.is_alive()

    def _sweeper_loop(self) -> None:
        """Sweeper body: wait for the interval, then scan, until asked to stop."""
        interval = max(1, int(self.policy.sweep_interval_s))
        # Delay the first round by a full interval so a freshly started service
        # does not produce noise by scanning an empty table.
        while not self._sweeper_stop.wait(interval):
            # Sweeping is best-effort: no exception may take down the service or
            # end the thread.
            with contextlib.suppress(Exception):
                self.sweep()
