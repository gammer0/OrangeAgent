"""Trace logging (ADT-7, group E).

See docs/02-data-structures/07-traces.md.

One JSONL file per session; one event per line, persisted immediately.

* E1: append and flush at the end of every step (optional fsync); never buffer
  until the end.
* E2: every line must be a completely parseable JSON object (on failure the
  whole line is not written).
* E3: every event carries ``session_id``, and files are isolated per session.

Threading model: ``TraceLogger`` holds an internal lock, so concurrent writers
to the same file never interleave half-lines; different sessions write to
different files and never block each other.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "TraceKind",
    "TraceEvent",
    "TraceLogger",
    "TRACE_DIR_NAME",
    "trace_file_path",
]

#: Trace subdirectory name: ``<trace_root>/<session_id>.jsonl``
TRACE_DIR_NAME = "traces"

TraceKind = Literal[
    "llm_call",
    "tool_exec",
    "compress",
    "inject",
    "error",
    "abort",
    "memory_extract",
]


def trace_file_path(trace_root: str | os.PathLike[str], session_id: str) -> Path:
    """Return the session trace file path ``<trace_root>/<session_id>.jsonl``.

    The file name sanitizes ``session_id`` (keeping only alphanumerics and
    ``-_.``) to prevent path traversal (E3: isolation per session, never
    escaping into another directory).
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in session_id)
    if not safe:
        raise ValueError("E3: session_id must not be empty")
    return Path(trace_root) / f"{safe}.jsonl"


class TraceEvent(BaseModel):
    """One trace event (ADT-7).

    The ``payload`` shape is described in design doc §7.2; this model does not
    validate payload fields per kind, which keeps it forward compatible (new
    fields never break parsing of old logs).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=0)]
    ts: float
    kind: TraceKind
    payload: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------- serialization

    def to_json_line(self) -> str:
        """Serialize to a single-line JSON string (no newline); E2 guarantees parseability."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_json_line(cls, line: str) -> TraceEvent:
        """Deserialize from one JSON line (for log replay and tests)."""
        return cls.model_validate(json.loads(line))


class TraceLogger:
    """Session-level trace writer: one JSONL per session (E1/E2/E3).

    Every ``log`` appends a complete line and flushes, so **other processes and
    tests see it immediately** (E1). The lock is per instance; multiple
    instances pointing at the same file rely on OS append semantics plus a
    single ``write`` call to keep lines intact, while multiple threads within
    one instance are fully safe.
    """

    def __init__(
        self,
        trace_root: str | os.PathLike[str],
        session_id: str,
        *,
        fsync: bool = False,
        clock: Any = time.time,
    ) -> None:
        if not session_id:
            raise ValueError("E3: session_id must not be empty")
        self._session_id = session_id
        self._path = trace_file_path(trace_root, session_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._clock = clock
        self._lock = threading.Lock()

    # ------------------------------------------------------------ properties

    @property
    def session_id(self) -> str:
        """Owning session id (E3)."""
        return self._session_id

    @property
    def path(self) -> Path:
        """JSONL file path."""
        return self._path

    # --------------------------------------------------------------- writing

    def log(self, step: int, kind: TraceKind, payload: dict[str, Any] | None = None) -> TraceEvent:
        """Append one event and persist it immediately; returns the written event.

        Args:
            step: ReAct step number.
            kind: Event kind.
            payload: Event payload; fields in design doc §7.2.
        """
        event = TraceEvent(
            session_id=self._session_id,
            step=step,
            ts=float(self._clock()),
            kind=kind,
            payload=payload or {},
        )
        line = event.to_json_line() + "\n"
        # A single write emits the whole line; the lock preserves ordering and
        # prevents interleaving within this instance.
        with self._lock, self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        return event

    # ------------------------------------------------- 7 convenience methods

    def log_llm_call(
        self,
        step: int,
        *,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        tool_call_count: int,
    ) -> TraceEvent:
        """Record one LLM call."""
        return self.log(
            step,
            "llm_call",
            {
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_ms": duration_ms,
                "tool_call_count": tool_call_count,
            },
        )

    def log_tool_exec(
        self,
        step: int,
        *,
        path: str,
        tool_id: str | None,
        arguments: Any,
        ok: bool,
        error_type: str | None,
        duration_ms: float,
    ) -> TraceEvent:
        """Record one tool execution."""
        return self.log(
            step,
            "tool_exec",
            {
                "path": path,
                "tool_id": tool_id,
                "arguments": arguments,
                "ok": ok,
                "error_type": error_type,
                "duration_ms": duration_ms,
            },
        )

    def log_compress(
        self,
        step: int,
        *,
        level: int,
        before_tokens: int,
        after_tokens: int,
        summary_chars: int,
    ) -> TraceEvent:
        """Record one compression."""
        return self.log(
            step,
            "compress",
            {
                "level": level,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "summary_chars": summary_chars,
            },
        )

    def log_inject(
        self, step: int, *, channel: Literal["chat", "control"], content_chars: int
    ) -> TraceEvent:
        """Record one external injection."""
        return self.log(step, "inject", {"channel": channel, "content_chars": content_chars})

    def log_error(
        self,
        step: int,
        *,
        error_type: str,
        message: str,
        retry_count: int,
    ) -> TraceEvent:
        """Record one error."""
        return self.log(
            step,
            "error",
            {"error_type": error_type, "message": message, "retry_count": retry_count},
        )

    def log_abort(self, step: int, *, reason: str, coalesced: bool) -> TraceEvent:
        """Record one abort."""
        return self.log(step, "abort", {"reason": reason, "coalesced": coalesced})

    def log_memory_extract(
        self, step: int, *, status: str, entries_added: int, duration_ms: float
    ) -> TraceEvent:
        """Record one settlement-extraction result."""
        return self.log(
            step,
            "memory_extract",
            {"status": status, "entries_added": entries_added, "duration_ms": duration_ms},
        )

    # --------------------------------------------------------------- replay

    def read_events(self) -> list[TraceEvent]:
        """Replay every persisted event of this session (each line must parse, E2)."""
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [TraceEvent.from_json_line(line) for line in lines if line.strip()]
