"""Model-layer runtime: rate limiting and health state (ADT-10, group K).

See docs/02-data-structures/10-model-layer.md.

Multiple sessions share one ``ReverseProxy``, so both the rate window and the
health state are **global**.

* K1: global rate windows only; no per-session concurrency limit.
* K2: every ``RateEvent`` must carry a ``session_id``.
* K3: window eviction slides out by ``ts`` only, never by session.
* K4: ``ProviderHealth`` is a global singleton; backoff and failover apply to
  every session.
* K5: failures during ``state != "healthy"`` do not increment
  ``consecutive_failures`` again.

All state access is guarded by ``threading.Lock``.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RateEvent",
    "RateWindow",
    "Decision",
    "ProviderState",
    "ProviderHealth",
    "get_provider_health",
    "reset_provider_health_registry",
]

Decision = Literal["allow", "retry_after", "fail_fast"]
ProviderState = Literal["healthy", "degraded", "down"]


# ------------------------------------------------------------ rate limiting


class RateEvent(BaseModel):
    """A single request record inside the window.

    ``tokens`` is the pre-request estimate; ``actual_tokens`` is backfilled by
    :meth:`RateWindow.record_actual` after the response arrives.
    """

    model_config = ConfigDict(extra="forbid")

    ts: float
    session_id: Annotated[str, Field(min_length=1)]  # K2
    tokens: int
    actual_tokens: int | None = None


class RateWindow:
    """Sliding-window rate limiter (one global instance).

    Thread safety: every read and write happens inside ``self._lock``.
    """

    def __init__(self, max_requests: int, max_tokens: int, window_s: int = 60) -> None:
        max_requests = int(max_requests)
        max_tokens = int(max_tokens)
        window_s = int(window_s)
        if max_requests <= 0:
            raise ValueError("max_requests must be a positive integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if window_s <= 0:
            raise ValueError("window_s must be a positive integer")
        self.window_s = window_s
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self._events: deque[RateEvent] = deque()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- views

    @property
    def events(self) -> list[RateEvent]:
        """Snapshot of the window's events, ordered by ascending ``ts``."""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ------------------------------------------------------------- eviction

    def _evict_locked(self, now: float) -> None:
        """Slide out events with ``ts <= now - window_s`` (K3: by ts only, never by session)."""
        cutoff = now - self.window_s
        events = self._events
        while events and events[0].ts <= cutoff:
            events.popleft()

    def _used_tokens_locked(self) -> int:
        """Total tokens occupied in the window (``actual_tokens`` preferred when present)."""
        total = 0
        for event in self._events:
            total += event.actual_tokens if event.actual_tokens is not None else event.tokens
        return total

    # ---------------------------------------------------------------- acquire

    def try_acquire(self, session_id: str, tokens: int, now: float) -> bool:
        """Try to claim one request slot; returns ``False`` when the window is full.

        On rejection no event is written, so the window is left untouched.

        Args:
            session_id: Session making the request (K2).
            tokens: Pre-request token estimate.
            now: Current timestamp (Unix seconds).
        """
        if not session_id:
            raise ValueError("K2: session_id must not be empty")
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        with self._lock:
            self._evict_locked(now)
            if len(self._events) + 1 > self.max_requests:
                return False
            if self._used_tokens_locked() + tokens > self.max_tokens:
                return False
            self._events.append(RateEvent(ts=now, session_id=session_id, tokens=tokens))
            return True

    # ------------------------------------------------------------ backfill

    def record_actual(self, session_id: str, actual_tokens: int, now: float | None = None) -> bool:
        """Backfill ``actual_tokens`` on the session's **earliest** unfilled event (K2).

        Backfill matches by ``session_id``, so concurrent sessions never write
        into each other's records. Returns ``False`` when nothing is available
        to backfill.

        Args:
            session_id: Session id.
            actual_tokens: Real token count obtained after the response.
            now: Current timestamp; when given, the window is evicted first.
        """
        if not session_id:
            raise ValueError("K2: session_id must not be empty")
        if actual_tokens < 0:
            raise ValueError("actual_tokens must not be negative")
        with self._lock:
            if now is not None:
                self._evict_locked(now)
            for event in self._events:
                if event.session_id == session_id and event.actual_tokens is None:
                    event.actual_tokens = actual_tokens
                    return True
            return False


# -------------------------------------------------------------- health state

#: K4: global singleton registry. Keys are provider names; all sessions share
#: the same instance.
_REGISTRY: dict[str, ProviderHealth] = {}
_REGISTRY_LOCK = threading.Lock()


class ProviderHealth:
    """Global health state of a single provider (K4: global singleton).

    Singleton semantics: one provider name maps to exactly one state within the
    process. Obtain it via :func:`get_provider_health` and **do not** construct
    instances yourself, otherwise backoff and failover will not apply to every
    session.
    """

    def __init__(
        self, provider: str, *, failure_threshold: int = 3, backoff_s: float = 30.0
    ) -> None:
        if not provider:
            raise ValueError("provider must not be empty")
        self.provider = provider
        self.state: ProviderState = "healthy"
        self.consecutive_failures = 0
        self.retry_after = 0.0
        self.switched_to: str | None = None
        self.failure_threshold = max(1, int(failure_threshold))
        self.backoff_s = float(backoff_s)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- decisions

    def can_send(self, now: float) -> Decision:
        """Pre-send check.

        Returns:
            ``"allow"`` to send normally; ``"retry_after"`` to wait for
            ``retry_after``; ``"fail_fast"`` to fail immediately without retry.
        """
        with self._lock:
            if self.state == "healthy":
                return "allow"
            if self.state == "degraded":
                return "allow" if now >= self.retry_after else "retry_after"
            return "fail_fast"

    # ------------------------------------------------------------ transitions

    def on_success(self) -> None:
        """A successful call: reset the failure count and return to ``healthy``."""
        with self._lock:
            self.consecutive_failures = 0
            self.state = "healthy"
            self.retry_after = 0.0
            self.switched_to = None

    def on_failure(self, now: float, threshold: int | None = None) -> ProviderState:
        """A failed call; returns the state after the transition.

        K5: during ``state != "healthy"`` the failure count is not incremented
        again, but ``retry_after`` is extended so backoff keeps applying.

        Args:
            now: Current timestamp.
            threshold: Failure threshold for this call; defaults to the instance
                configuration.
        """
        limit = max(1, int(threshold if threshold is not None else self.failure_threshold))
        with self._lock:
            if self.state == "down":
                # K5: no increment while down; keep failing fast
                return self.state
            if self.state == "degraded":
                # K5: no re-increment, but refresh the backoff deadline
                self.retry_after = now + self.backoff_s
                return self.state
            self.consecutive_failures += 1
            if self.consecutive_failures >= limit:
                self.state = "degraded"
                self.retry_after = now + self.backoff_s
            return self.state

    def mark_down(self, switched_to: str | None = None) -> None:
        """Explicitly degrade to ``down`` (e.g. a failover target has been chosen)."""
        with self._lock:
            self.state = "down"
            self.switched_to = switched_to

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, object]:
        """Thread-safe read-only snapshot."""
        with self._lock:
            return {
                "provider": self.provider,
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "retry_after": self.retry_after,
                "switched_to": self.switched_to,
            }

    def __repr__(self) -> str:
        data = self.snapshot()
        return (
            f"ProviderHealth(provider={data['provider']!r}, state={data['state']!r}, "
            f"consecutive_failures={data['consecutive_failures']!r})"
        )


def get_provider_health(
    provider: str, *, failure_threshold: int = 3, backoff_s: float = 30.0
) -> ProviderHealth:
    """Get (creating if needed) a provider's global singleton health state (K4).

    Repeated calls with the same ``provider`` name return **the same object**;
    ``failure_threshold`` / ``backoff_s`` only take effect on first creation and
    later arguments are ignored.
    """
    with _REGISTRY_LOCK:
        health = _REGISTRY.get(provider)
        if health is None:
            health = ProviderHealth(
                provider, failure_threshold=failure_threshold, backoff_s=backoff_s
            )
            _REGISTRY[provider] = health
        return health


def reset_provider_health_registry() -> None:
    """Clear the global registry (test use only)."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
