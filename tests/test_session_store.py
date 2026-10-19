"""SQLite 持久化测试（H6–H9、H13、H14）。

对应文档：docs/02-数据结构设计/02-会话层.md §2.3
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from harness.exceptions import MessageInvariantError
from harness.message import (
    AssistantMsg,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from harness.session import SessionRecord, Watermark
from harness.paths import harness_home
from harness.session_store import DB_RELATIVE_PATH, SessionStore


def _record(sid: str = "s1", workdir: str = "w", status: str = "idle") -> SessionRecord:
    return SessionRecord(
        session_id=sid,
        working_dir=workdir,
        created_at=100.0,
        updated_at=200.0,
        status=status,  # type: ignore[arg-type]
        watermark=Watermark(capacity=1000, compressed_at={50}, last_estimate=400),
    )


def _call(call_id: str = "c1", name: str = "/tool/os/file/read") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments='{"path": "a.txt"}')


@pytest.fixture
def store(tmp_path):
    s = SessionStore(db_path=tmp_path / "harness.db")
    s.upsert_session(_record(workdir=str(tmp_path)))
    yield s
    s.close()


class TestDatabaseSetup:
    def test_db_defaults_outside_working_dir(self, tmp_path) -> None:
        """数据库不得写进工作目录：默认落在系统数据目录。"""
        store = SessionStore(tmp_path)
        store.upsert_session(_record())
        assert store.db_path == harness_home() / DB_RELATIVE_PATH
        assert store.db_path.is_file()
        assert not (tmp_path / ".agent").exists()
        store.close()

    def test_explicit_db_path_is_honoured(self, tmp_path) -> None:
        target = tmp_path / "custom" / "sessions.db"
        store = SessionStore(db_path=target)
        store.upsert_session(_record())
        assert store.db_path == target
        assert target.is_file()
        store.close()

    def test_wal_mode_enabled(self, store) -> None:
        assert store.journal_mode == "wal"

    def test_tables_match_design(self, store) -> None:
        conn = store._connect()
        session_cols = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)")]
        message_cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)")]
        assert session_cols == [
            "session_id",
            "working_dir",
            "created_at",
            "updated_at",
            "status",
            "watermark",
        ]
        assert message_cols == [
            "session_id",
            "seq",
            "type",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
            "created_at",
        ]


class TestSessionsTable:
    def test_upsert_and_get(self, store, tmp_path) -> None:
        record = store.get_session("s1")
        assert record is not None
        assert record.working_dir == str(tmp_path)
        assert record.status == "idle"
        assert record.watermark == Watermark(capacity=1000, compressed_at={50}, last_estimate=400)

    def test_upsert_updates_existing_row(self, store) -> None:
        store.upsert_session(_record(status="ended"))
        record = store.get_session("s1")
        assert record is not None
        assert record.status == "ended"
        assert len(store.list_sessions()) == 1

    def test_missing_session_returns_none(self, store) -> None:
        assert store.get_session("nope") is None
        assert store.restore("nope") is None

    def test_watermark_nullable(self, store) -> None:
        store.upsert_session(
            SessionRecord(
                session_id="nowm",
                working_dir="w",
                created_at=1.0,
                updated_at=1.0,
                status="idle",
                watermark=None,
            )
        )
        record = store.get_session("nowm")
        assert record is not None
        assert record.watermark is None

    def test_update_status(self, store) -> None:
        store.update_status("s1", "active", watermark=Watermark(capacity=2000))
        record = store.get_session("s1")
        assert record is not None
        assert record.status == "active"
        assert record.watermark is not None
        assert record.watermark.capacity == 2000


class TestMessagesTable:
    def test_append_assigns_monotonic_seq_H9(self, store) -> None:
        seqs = store.append_messages("s1", [UserMsg(content="hi"), AssistantMsg(content="hello")])
        assert seqs == [1, 2]

        more = store.append_messages("s1", [UserMsg(content="again")])
        assert more == [3]
        assert store.next_seq("s1") == 4

        rows = store.load_messages("s1")
        assert [row.seq for row in rows] == [1, 2, 3]

    def test_seq_is_per_session_H9(self, store) -> None:
        store.upsert_session(_record("s2"))
        store.append_messages("s1", [UserMsg(content="a")])
        store.append_messages("s2", [UserMsg(content="b")])
        assert [row.seq for row in store.load_messages("s2")] == [1]
        assert store.next_seq("s1") == 2

    def test_assistant_tool_calls_roundtrip(self, store) -> None:
        assistant = AssistantMsg(content=None, tool_calls=[_call()])
        tool = ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok")
        store.append_messages("s1", [assistant, tool])

        rows = store.load_messages("s1")
        assert rows[0].tool_calls is not None
        assert "c1" in rows[0].tool_calls
        assert rows[1].tool_call_id == "c1"
        assert rows[1].name == "/tool/os/file/read"

    def test_system_msg_rejected_H6(self, store) -> None:
        with pytest.raises(MessageInvariantError) as exc:
            store.append_messages("s1", [SystemMsg(content="sys")])
        assert "H6" in exc.value.detail
        assert store.load_messages("s1") == []

    def test_empty_batch_is_noop(self, store) -> None:
        assert store.append_messages("s1", []) == []
        assert store.next_seq("s1") == 1


class TestGroupAtomicity:
    def test_group_commits_atomically_H7(self, store) -> None:
        assistant = AssistantMsg(content=None, tool_calls=[_call("c1"), _call("c2")])
        store.append_messages("s1", [assistant])

        results = [
            ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok"),
            ToolMsg(tool_call_id="c2", name="/tool/os/file/read", content="ok"),
        ]
        seqs = store.append_group("s1", 0, results)

        assert seqs == [2, 3]
        assert len(store.load_messages("s1")) == 3

    def test_failed_group_leaves_no_partial_rows_H7(self, store) -> None:
        assistant = AssistantMsg(content=None, tool_calls=[_call("c1"), _call("c2")])
        store.append_messages("s1", [assistant])

        good = ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok")
        bad = SystemMsg(content="boom")  # 批次中途触发 H6 拒绝

        with pytest.raises(MessageInvariantError), store._connect():
            # 模拟真实调用方：外层事务 + commit=False
            store.append_messages("s1", [good], commit=False)
            store.append_messages("s1", [bad], commit=False)

        rows = store.load_messages("s1")
        assert [row.seq for row in rows] == [1]  # 只有 assistant，组内写入全回滚
        assert store.next_seq("s1") == 2  # seq 无空洞

    def test_foreign_key_enforced(self, store) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            store.append_messages("ghost", [UserMsg(content="x")])


class TestRestore:
    def test_restore_returns_validated_messages_H6(self, store) -> None:
        store.append_messages(
            "s1",
            [
                UserMsg(content="读文件"),
                AssistantMsg(content=None, tool_calls=[_call()]),
                ToolMsg(tool_call_id="c1", name="/tool/os/file/read", content="ok"),
                AssistantMsg(content="完成"),
            ],
        )

        restored = store.restore("s1")

        assert restored is not None
        assert restored.record.session_id == "s1"
        assert len(restored.messages) == 4
        assert not any(isinstance(m, SystemMsg) for m in restored.messages)
        validate_messages([SystemMsg(content="reassembled"), *restored.messages])

    def test_restore_resets_session_state_H8(self, store) -> None:
        store.append_messages("s1", [UserMsg(content="hi")])
        restored = store.restore("s1")
        assert restored is not None
        assert restored.delivered == set()
        assert restored.discovered == {}

    def test_restore_seq_is_monotonic_and_gapless_H9(self, store) -> None:
        for i in range(10):
            store.append_messages("s1", [UserMsg(content=f"m{i}")])

        rows = store.load_messages("s1")
        assert [row.seq for row in rows] == list(range(1, 11))

        restored = store.restore("s1")
        assert restored is not None
        # 恢复后再追加，seq 从 11 继续，无重复
        assert store.append_messages("s1", [UserMsg(content="after")]) == [11]

    def test_restore_detects_corrupted_sequence(self, store) -> None:
        store.append_messages("s1", [AssistantMsg(content=None, tool_calls=[_call()])])
        # 人为制造缺失 ToolMsg 的不完整状态（模拟非事务写入）
        with store._connect() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(session_id, seq, type, content, tool_calls, tool_call_id, name, created_at) "
                "VALUES (?, 2, 'user', 'hi', NULL, NULL, NULL, 0)",
                ("s1",),
            )
        with pytest.raises(MessageInvariantError) as exc:
            store.restore("s1")
        assert exc.value.rule == "A3"

    def test_restore_empty_session(self, store) -> None:
        restored = store.restore("s1")
        assert restored is not None
        assert restored.messages == []


class TestPerThreadConnections:
    def test_each_thread_gets_own_connection(self, store) -> None:
        conns: dict[int, sqlite3.Connection] = {}
        lock = threading.Lock()

        def worker() -> None:
            store.append_messages("s1", [UserMsg(content="thread")])
            conn = store._connect()
            with lock:
                conns[threading.get_ident()] = conn

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 连接对象仍被 conns 持有，因此 id() 不会被回收复用；改为两两身份比较
        values = list(conns.values())
        assert len(values) == len(threads)
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                assert left is not right

    def test_concurrent_appends_get_unique_seq_H9(self, store) -> None:
        """并发写同一会话时 seq 必须唯一且连续（主键冲突由调用方重试或串行化）。"""
        writers, per_writer = 4, 25
        lock = threading.Lock()
        failures = 0

        def worker(writer_id: int) -> None:
            nonlocal failures
            for _ in range(per_writer):
                for attempt in range(50):
                    try:
                        with lock:
                            store.append_messages("s1", [UserMsg(content=f"w{writer_id}")])
                        break
                    except sqlite3.IntegrityError:  # pragma: no cover - 竞态重试
                        if attempt == 49:  # pragma: no cover
                            failures += 1

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert failures == 0
        rows = store.load_messages("s1")
        assert [row.seq for row in rows] == list(range(1, writers * per_writer + 1))

    def test_concurrent_insert_conflict_is_reported(self, tmp_path) -> None:
        """并发写同一 seq 时由主键保证不重复（至少一方报错或全部串行成功）。"""
        store = SessionStore(tmp_path)
        store.upsert_session(_record())
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                seqs = store.append_messages("s1", [UserMsg(content="x")])
            except sqlite3.IntegrityError:
                return
            with lock:
                results.extend(seqs)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == len(set(results))
        store.close()
