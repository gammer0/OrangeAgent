"""Session-layer SQLite persistence (H6-H9, H13, H14).

See docs/02-data-structures/02-session-layer.md §2.3.

* Database location: ``<home>/orange.db`` (system data directory, see
  ``orange.paths``) -- never inside the working directory the user is editing.
* ``WAL`` mode, one connection per thread (``threading.local``)
* ``SystemMsg`` is **never persisted** (H6)
* The transaction unit is a **single ``ExecGroup``**: one group commits in one
  transaction (H7)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from orange import paths
from orange.exceptions import MessageInvariantError
from orange.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from orange.session import SessionRecord, SessionStatus, Watermark

__all__ = [
    "DB_RELATIVE_PATH",
    "SCHEMA_SQL",
    "StoredMessage",
    "RestoredSession",
    "SessionStore",
]

#: Location of the database relative to the installation data directory.
#: Kept as a ``Path`` for backward compatibility; the file is no longer placed
#: under the working directory.
DB_RELATIVE_PATH = Path("orange.db")

#: Schema SQL, verbatim from design doc §2.3
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    working_dir  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    status       TEXT NOT NULL,     -- active | idle | ended
    watermark    TEXT               -- JSON serialized; nullable
);

CREATE TABLE IF NOT EXISTS messages (
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,  -- monotonic within a session
    type         TEXT NOT NULL,     -- user | assistant | tool
    content      TEXT,              -- nullable (assistant with tool calls only)
    tool_calls   TEXT,              -- JSON string, assistant only
    tool_call_id TEXT,              -- tool only
    name         TEXT,              -- tool only; internal field
    created_at   REAL NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_working_dir ON sessions(working_dir);

CREATE TABLE IF NOT EXISTS trace_events (
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,  -- monotonic within a session
    kind         TEXT NOT NULL,     -- tool | assistant | route | ...
    payload      TEXT NOT NULL,     -- JSON string
    created_at   REAL NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_trace_events_session ON trace_events(session_id);
"""

MessageType = Literal["user", "assistant", "tool"]

#: Placeholder SystemMsg used only by ``validate_messages`` (never persisted,
#: never returned; see H6)
_VALIDATION_SYSTEM_STUB = SystemMsg(content="<validation-stub>")


@dataclass(frozen=True)
class StoredMessage:
    """Read view of one row in the ``messages`` table."""

    session_id: str
    seq: int
    type: MessageType
    content: str | None
    tool_calls: str | None
    tool_call_id: str | None
    name: str | None
    created_at: float


@dataclass(frozen=True)
class StoredTraceEvent:
    """Read view of one row in the ``trace_events`` table."""

    session_id: str
    seq: int
    kind: str
    payload: str
    created_at: float


@dataclass(frozen=True)
class RestoredSession:
    """Session data rebuilt from SQLite (**without** the ``SystemMsg``, H6).

    Attributes:
        record: The ``sessions`` row.
        messages: Messages in ascending ``seq`` order, **already validated by
            ``validate_messages``**. Step 4 of the restore flow (re-assembling
            the ``SystemMsg``) is done by the context layer, so the first item
            here is either an existing ``SystemMsg`` or the first historical
            user message.
        delivered: Always an empty set placeholder, satisfying H8.
        discovered: Always an empty dict placeholder, satisfying H8.
    """

    record: SessionRecord
    messages: list[Message]
    delivered: set[str]
    discovered: dict[str, Any]


class SessionStore:
    """SQLite session storage.

    One connection per thread (design doc §2.3); connections use
    ``check_same_thread=False`` to disable sqlite3's own check and rely on
    ``threading.local`` for thread isolation instead.
    """

    def __init__(
        self,
        workdir: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        """Open (creating if needed) the shared session database.

        Args:
            workdir: Accepted for backward compatibility and **ignored**: the
                database is not stored in the working directory.
            db_path: Explicit database location; defaults to ``<home>/orange.db``.
        """
        self.workdir = Path(workdir) if workdir is not None else None
        self.db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else paths.orange_home() / DB_RELATIVE_PATH
        )
        self._local = threading.local()
        # Serializes first-time database creation (WAL + schema): protects
        # initialization only, not day-to-day reads and writes.
        self._init_lock = threading.Lock()
        # Serializes seq allocation (H9): guards the MAX(seq) read-modify-write.
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return the current thread's connection, creating the DB and enabling WAL on first use.

        When several threads connect for the first time concurrently,
        ``PRAGMA journal_mode=WAL`` can fail with ``database is locked`` because
        another thread is still creating the DB; a ``timeout`` plus serialized
        initialization handles that.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Database creation (WAL switch and schema) must be serialized, otherwise
        # concurrent first connections fight over the lock.
        with self._init_lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the current thread's connection (idempotent)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @property
    def journal_mode(self) -> str:
        """The database's current ``journal_mode`` (lets tests assert WAL is active)."""
        row = self._connect().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------
    # sessions table
    # ------------------------------------------------------------------

    def upsert_session(self, record: SessionRecord) -> None:
        """Insert or update a ``sessions`` row."""
        watermark_json = (
            json.dumps(record.watermark.to_json(), ensure_ascii=False, sort_keys=True)
            if record.watermark is not None
            else None
        )
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (session_id, working_dir, created_at, updated_at, status, watermark)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    working_dir = excluded.working_dir,
                    created_at  = excluded.created_at,
                    updated_at  = excluded.updated_at,
                    status      = excluded.status,
                    watermark   = excluded.watermark
                """,
                (
                    record.session_id,
                    record.working_dir,
                    record.created_at,
                    record.updated_at,
                    record.status,
                    watermark_json,
                ),
            )

    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        watermark: Watermark | None = None,
    ) -> None:
        """Update the session status (and optional watermark snapshot) plus ``updated_at``."""
        watermark_json = (
            json.dumps(watermark.to_json(), ensure_ascii=False, sort_keys=True)
            if watermark is not None
            else None
        )
        conn = self._connect()
        with conn:
            if watermark_json is None:
                conn.execute(
                    "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                    (status, time.time(), session_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE sessions
                       SET status = ?, updated_at = ?, watermark = ?
                     WHERE session_id = ?
                    """,
                    (status, time.time(), watermark_json, session_id),
                )

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Read a ``sessions`` row; ``None`` if absent."""
        row = (
            self._connect()
            .execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            .fetchone()
        )
        if row is None:
            return None
        raw_watermark = row["watermark"]
        watermark = (
            Watermark.from_json(json.loads(raw_watermark)) if raw_watermark is not None else None
        )
        return SessionRecord(
            session_id=row["session_id"],
            working_dir=row["working_dir"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            watermark=watermark,
        )

    def list_sessions(self) -> list[SessionRecord]:
        """List every session row, ascending by ``updated_at``."""
        rows = (
            self._connect()
            .execute("SELECT * FROM sessions ORDER BY updated_at ASC, session_id ASC")
            .fetchall()
        )
        records: list[SessionRecord] = []
        for row in rows:
            raw_watermark = row["watermark"]
            records.append(
                SessionRecord(
                    session_id=row["session_id"],
                    working_dir=row["working_dir"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    status=row["status"],
                    watermark=Watermark.from_json(json.loads(raw_watermark))
                    if raw_watermark is not None
                    else None,
                )
            )
        return records

    # ------------------------------------------------------------------
    # messages table
    # ------------------------------------------------------------------

    def append_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
        *,
        commit: bool = True,
    ) -> list[int]:
        """Append a batch of messages, committed **in one transaction** (H7).

        Typical usage: once an ``ExecGroup`` finishes, write all of that group's
        ``ToolMsg`` values at once; the same applies to ``3e`` placeholders. With
        ``commit=False`` the writes join the caller's transaction, which the
        caller commits inside its own ``with conn:`` block.

        Args:
            session_id: Session id.
            messages: Messages to write, assigned consecutive ``seq`` values in
                the given order (H9).
            commit: Whether to commit here. ``False`` relies on an outer
                transaction.

        Returns:
            The ``seq`` values assigned to this batch.

        Raises:
            MessageInvariantError: A ``SystemMsg`` appears (H6: SystemMsg is
                never persisted).
        """
        if not messages:
            return []

        for msg in messages:
            if isinstance(msg, SystemMsg):
                raise MessageInvariantError(
                    "A1", "H6: SystemMsg is never persisted; it must not enter the messages table"
                )

        conn = self._connect()
        now = time.time()

        # H9: seq must be monotonic and gapless. Reading and inserting around
        # MAX(seq) must be atomic, otherwise concurrent writes to the same
        # session race for the same seq and hit a primary-key conflict.
        with self._write_lock:
            if commit:
                with conn:
                    return self._insert_batch(conn, session_id, messages, now)
            return self._insert_batch(conn, session_id, messages, now)

    def _insert_batch(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        messages: Sequence[Message],
        now: float,
    ) -> list[int]:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_seq = int(row["max_seq"]) + 1

        seqs: list[int] = []
        for msg in messages:
            seq = next_seq
            next_seq += 1
            conn.execute(
                """
                INSERT INTO messages
                    (session_id, seq, type, content, tool_calls, tool_call_id, name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _to_row(session_id, seq, msg, now),
            )
            seqs.append(seq)
        return seqs

    def append_trace_events(
        self,
        session_id: str,
        events: Sequence[tuple[str, dict[str, Any], float]],
    ) -> int:
        """Append trace events to their own table.

        Trace history and message history are deliberately separate. A tree
        descent is *not* a message: once the tool it revealed has been called,
        the descent turn is pruned from the conversation so the model never
        re-reads it. But the user still needs to see how the tool was found,
        including after a restart, so the event stream is stored here instead.

        Args:
            session_id: Owning session.
            events: ``(kind, payload, created_at)`` triples in emission order.

        Returns:
            The number of rows written.
        """
        if not events:
            return 0
        import json as _json

        conn = self._connect()
        with self._write_lock:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM trace_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            start = int(row["m"])
            rows = [
                (
                    session_id,
                    start + offset,
                    kind,
                    _json.dumps(payload, ensure_ascii=False),
                    created_at,
                )
                for offset, (kind, payload, created_at) in enumerate(events, start=1)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO trace_events "
                "(session_id, seq, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def load_trace_events(self, session_id: str) -> list[StoredTraceEvent]:
        """Return a session's stored trace events in ascending ``seq`` order."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT session_id, seq, kind, payload, created_at FROM trace_events "
            "WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        return [
            StoredTraceEvent(
                session_id=r["session_id"],
                seq=int(r["seq"]),
                kind=r["kind"],
                payload=r["payload"],
                created_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def append_group(
        self,
        session_id: str,
        group_index: int,
        messages: Sequence[Message],
    ) -> list[int]:
        """Write all messages of a single ``ExecGroup`` and commit atomically (H7).

        ``group_index`` is used for diagnostic logs only and is not persisted;
        the transaction boundary is what carries the semantics.
        """
        del group_index
        return self.append_messages(session_id, messages)

    def load_messages(self, session_id: str) -> list[StoredMessage]:
        """Read every message row of a session, ascending by ``seq``."""
        rows = (
            self._connect()
            .execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            )
            .fetchall()
        )
        return [
            StoredMessage(
                session_id=row["session_id"],
                seq=row["seq"],
                type=row["type"],
                content=row["content"],
                tool_calls=row["tool_calls"],
                tool_call_id=row["tool_call_id"],
                name=row["name"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def next_seq(self, session_id: str) -> int:
        """Return the next available ``seq`` (monotonic within the session, H9)."""
        row = (
            self._connect()
            .execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM messages WHERE session_id = ?",
                (session_id,),
            )
            .fetchone()
        )
        return int(row["max_seq"]) + 1

    # ------------------------------------------------------------------
    # Restore (design doc §2.3, five steps)
    # ------------------------------------------------------------------

    def restore(self, session_id: str) -> RestoredSession | None:
        """Rebuild session data from SQLite.

        Steps:

        1. Read the ``sessions`` row (return ``None`` if absent);
        2. ``SELECT messages ... ORDER BY seq``;
        3. Validate the sequence with ``validate_messages``;
        4. Do **not** assemble the ``SystemMsg`` here (the context layer rebuilds
           it from the current tool tree, H6);
        5. Return empty ``delivered`` and ``discovered`` placeholders (H8 / H14).

        Raises:
            MessageInvariantError: The stored message sequence violates group-A
                constraints.
        """
        record = self.get_session(session_id)
        if record is None:
            return None

        messages = [row_to_message(row) for row in self.load_messages(session_id)]
        if messages:
            # The SystemMsg is not persisted (H6), but validate_messages requires
            # the first message to be a SystemMsg (A1). A temporary stub lets us
            # run the sequence-level check without adding it to the returned
            # value; the real SystemMsg is re-assembled by the context layer from
            # the current tool tree (restore step 4).
            validate_messages([_VALIDATION_SYSTEM_STUB, *messages])

        return RestoredSession(
            record=record,
            messages=messages,
            delivered=set(),
            discovered={},
        )


# ----------------------------------------------------------------------
# Row <-> message mapping
# ----------------------------------------------------------------------


def _to_row(
    session_id: str,
    seq: int,
    msg: Message,
    now: float,
) -> tuple[str, int, str, str | None, str | None, str | None, str | None, float]:
    """Flatten a message into one row of the ``messages`` table."""
    if isinstance(msg, UserMsg):
        return (session_id, seq, "user", msg.content, None, None, None, now)
    if isinstance(msg, AssistantMsg):
        tool_calls = (
            json.dumps([call.model_dump() for call in msg.tool_calls], ensure_ascii=False)
            if msg.tool_calls
            else None
        )
        return (session_id, seq, "assistant", msg.content, tool_calls, None, None, now)
    if isinstance(msg, ToolMsg):
        return (session_id, seq, "tool", msg.content, None, msg.tool_call_id, msg.name, now)
    raise MessageInvariantError(
        "A1", f"unsupported message type for persistence: {type(msg).__name__}"
    )


def row_to_message(row: StoredMessage) -> Message:
    """Restore one ``messages`` row into a message object."""
    if row.type == "user":
        return UserMsg(content=row.content or "")
    if row.type == "assistant":
        calls = (
            [ToolCall.model_validate(item) for item in json.loads(row.tool_calls)]
            if row.tool_calls
            else []
        )
        return AssistantMsg(content=row.content, tool_calls=calls)
    if row.type == "tool":
        return ToolMsg(
            tool_call_id=row.tool_call_id or "",
            name=row.name or "",
            content=row.content or "",
        )
    raise MessageInvariantError("A1", f"unknown message type column value: {row.type!r}")


def iter_rows(rows: Iterable[StoredMessage]) -> Iterable[Message]:  # pragma: no cover
    """Helper: restore rows in bulk (kept for upper-layer migration scripts)."""
    return (row_to_message(row) for row in rows)
