"""会话层 SQLite 持久化（H6–H9、H13、H14）。

对应文档：docs/02-数据结构设计/02-会话层.md §2.3

* 数据库位置：``<workdir>/.agent/harness.db``
* ``WAL`` 模式，每线程独立连接（``threading.local``）
* ``SystemMsg`` **不入库**（H6）
* 事务单元是**单个 ``ExecGroup``**：一组消息同事务提交（H7）
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

from harness.exceptions import MessageInvariantError
from harness.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.session import SessionRecord, SessionStatus, Watermark

__all__ = [
    "DB_RELATIVE_PATH",
    "SCHEMA_SQL",
    "StoredMessage",
    "RestoredSession",
    "SessionStore",
]

#: 数据库相对于工作目录的位置
DB_RELATIVE_PATH = Path(".agent") / "harness.db"

#: 建表 SQL，与文档 §2.3 逐字一致
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    working_dir  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    status       TEXT NOT NULL,     -- active | idle | ended
    watermark    TEXT               -- JSON 序列化，可空
);

CREATE TABLE IF NOT EXISTS messages (
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,  -- 会话内单调递增
    type         TEXT NOT NULL,     -- user | assistant | tool
    content      TEXT,              -- 可空（assistant 纯工具调用）
    tool_calls   TEXT,              -- JSON 字符串，仅 assistant
    tool_call_id TEXT,              -- 仅 tool
    name         TEXT,              -- 仅 tool，内部字段
    created_at   REAL NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

MessageType = Literal["user", "assistant", "tool"]

#: 仅供 ``validate_messages`` 使用的占位 SystemMsg（不落库、不返回，见 H6）
_VALIDATION_SYSTEM_STUB = SystemMsg(content="<validation-stub>")


@dataclass(frozen=True)
class StoredMessage:
    """``messages`` 表一行的读取视图。"""

    session_id: str
    seq: int
    type: MessageType
    content: str | None
    tool_calls: str | None
    tool_call_id: str | None
    name: str | None
    created_at: float


@dataclass(frozen=True)
class RestoredSession:
    """从 SQLite 重建出的会话数据（**不含** ``SystemMsg``，H6）。

    Attributes:
        record: ``sessions`` 行。
        messages: 按 ``seq`` 升序的消息，**已通过 ``validate_messages``**。
            恢复流程的第 4 步（重新装配 ``SystemMsg``）由上下文层完成，
            因此这里的首条是已存在的 ``SystemMsg`` 或历史首条用户消息。
        delivered: 恒为空集合占位，满足 H8。
        discovered: 恒为空字典占位，满足 H8。
    """

    record: SessionRecord
    messages: list[Message]
    delivered: set[str]
    discovered: dict[str, Any]


class SessionStore:
    """SQLite 会话存储。

    每线程独立连接（文档 §2.3），连接使用 ``check_same_thread=False`` 关闭
    sqlite3 自带检查，改由 ``threading.local`` 保证线程隔离。
    """

    def __init__(self, workdir: str | Path) -> None:
        """打开（必要时创建）工作目录下的数据库。

        Args:
            workdir: 工作目录；数据库位于 ``<workdir>/.agent/harness.db``。
        """
        self.workdir = Path(workdir)
        self.db_path = self.workdir / DB_RELATIVE_PATH
        self._local = threading.local()
        # 首次建库（WAL + 建表）的串行化锁：仅保护初始化，不保护日常读写
        self._init_lock = threading.Lock()
        # seq 分配串行化锁（H9）：保护 ``MAX(seq)`` 读-改-写
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """返回当前线程的连接，首次调用时建库并启用 WAL。

        并发首次建连时，``PRAGMA journal_mode=WAL`` 可能因其他线程正在建库
        而报 ``database is locked``；这里用 ``timeout`` + 建库串行化处理。
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # 建库（含 WAL 切换与建表）必须串行，否则并发首次连接会互相抢锁。
        with self._init_lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """关闭当前线程的连接（幂等）。"""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @property
    def journal_mode(self) -> str:
        """当前数据库的 ``journal_mode``（供测试断言 WAL 生效）。"""
        row = self._connect().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------
    # sessions 表
    # ------------------------------------------------------------------

    def upsert_session(self, record: SessionRecord) -> None:
        """写入或更新 ``sessions`` 行。"""
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
        """更新会话状态（及可选的水位快照）与 ``updated_at``。"""
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
        """读取 ``sessions`` 行；不存在返回 ``None``。"""
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
        """按 ``updated_at`` 升序列出全部会话行。"""
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
    # messages 表
    # ------------------------------------------------------------------

    def append_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
        *,
        commit: bool = True,
    ) -> list[int]:
        """追加一批消息，**同事务**提交（H7）。

        典型用法：一个 ``ExecGroup`` 完成后，把该组全部 ``ToolMsg`` 一次写入；
        补占位（3e）同理。``commit=False`` 时并入调用方事务，由调用方在
        ``with conn:`` 块内统一提交。

        Args:
            session_id: 会话 id。
            messages: 待写入消息，按给定顺序分配连续 ``seq``（H9）。
            commit: 是否自行提交。``False`` 时依赖外层事务。

        Returns:
            本批消息分配的 ``seq`` 列表。

        Raises:
            MessageInvariantError: 出现 ``SystemMsg``（H6：SystemMsg 不入库）。
        """
        if not messages:
            return []

        for msg in messages:
            if isinstance(msg, SystemMsg):
                raise MessageInvariantError("A1", "H6: SystemMsg 不入库，不得写入 messages 表")

        conn = self._connect()
        now = time.time()

        # H9：seq 必须单调且无空洞。``MAX(seq)`` 读取与插入必须原子，
        # 否则并发写同一会话会争用同一 seq 触发主键冲突。
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

    def append_group(
        self,
        session_id: str,
        group_index: int,
        messages: Sequence[Message],
    ) -> list[int]:
        """写入单个 ``ExecGroup`` 的全部消息并原子提交（H7）。

        ``group_index`` 仅用于诊断日志，不落库；事务边界才是语义所在。
        """
        del group_index
        return self.append_messages(session_id, messages)

    def load_messages(self, session_id: str) -> list[StoredMessage]:
        """按 ``seq`` 升序读取某会话全部消息行。"""
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
        """返回下一个可用 ``seq``（会话内单调递增，H9）。"""
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
    # 恢复（文档 §2.3 五步）
    # ------------------------------------------------------------------

    def restore(self, session_id: str) -> RestoredSession | None:
        """从 SQLite 重建会话数据。

        步骤：

        1. 读 ``sessions`` 行（不存在返回 ``None``）；
        2. ``SELECT messages ... ORDER BY seq``；
        3. ``validate_messages`` 校验序列；
        4. **不**在此装配 ``SystemMsg``（由上下文层按当前工具树重装，H6）；
        5. 返回空的 ``delivered`` 与 ``discovered`` 占位（H8 / H14）。

        Raises:
            MessageInvariantError: 库内消息序列违反 A 组约束。
        """
        record = self.get_session(session_id)
        if record is None:
            return None

        messages = [row_to_message(row) for row in self.load_messages(session_id)]
        if messages:
            # SystemMsg 不入库（H6），但 validate_messages 要求首条为 SystemMsg（A1）。
            # 这里用临时占位对象完成序列级校验，不并入返回值；真正的 SystemMsg 由
            # 上下文层按当前工具树重新装配（恢复流程第 4 步）。
            validate_messages([_VALIDATION_SYSTEM_STUB, *messages])

        return RestoredSession(
            record=record,
            messages=messages,
            delivered=set(),
            discovered={},
        )


# ----------------------------------------------------------------------
# 行 ↔ 消息映射
# ----------------------------------------------------------------------


def _to_row(
    session_id: str,
    seq: int,
    msg: Message,
    now: float,
) -> tuple[str, int, str, str | None, str | None, str | None, str | None, float]:
    """把消息拍平成 ``messages`` 表的一行。"""
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
    raise MessageInvariantError("A1", f"不支持的入库消息类型: {type(msg).__name__}")


def row_to_message(row: StoredMessage) -> Message:
    """把 ``messages`` 表一行还原为消息对象。"""
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
    raise MessageInvariantError("A1", f"未知消息类型列值: {row.type!r}")


def iter_rows(rows: Iterable[StoredMessage]) -> Iterable[Message]:  # pragma: no cover
    """辅助：批量把行还原为消息（保留供上层迁移脚本使用）。"""
    return (row_to_message(row) for row in rows)
