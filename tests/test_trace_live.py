"""轨迹事件的重复与补发语义。

前端有两条来源：选中会话时从 ``/trace`` 加载已落盘的积压，以及随后的 SSE
实时流。两者在时间上重叠，因此必须满足：
  * 同一事件不能渲染两次（按 seq 去重）；
  * 断线重连只补发客户端尚未收到的部分（``after_seq``）；
  * 读取 ``/trace`` 前要先刷写缓冲，否则刚产生的事件会漏掉。
"""

from __future__ import annotations

from pathlib import Path

from apps.server.sessions import DemoToolRegistry, SessionRunner

from harness.session import SessionRecord
from harness.session_store import SessionStore


def _runner(tmp_path: Path, session_id: str) -> SessionRunner:
    tools = DemoToolRegistry(str(tmp_path))
    runner = SessionRunner(
        session_id=session_id,
        workdir=str(tmp_path),
        proxy=None,  # never used in these tests
        tools=tools,
        model="fake",
    )
    runner.store = SessionStore(db_path=tmp_path / "harness.db")
    runner.store.upsert_session(
        SessionRecord(
            session_id=session_id,
            working_dir=str(tmp_path),
            created_at=1.0,
            updated_at=1.0,
            status="idle",
            watermark=None,
        )
    )
    return runner


def test_emit_buffers_then_flushes(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "s1")

    runner.emit("tool", {"path": "/tool"})
    runner.emit("tool", {"path": "/tool/os"})
    assert len(runner._trace_pending) == 2, "emit must buffer, not write through"

    assert runner.persist_trace() == 2
    assert runner._trace_pending == [], "a flush drains the buffer"

    rows = runner.store.load_trace_events("s1")  # type: ignore[union-attr]
    assert [r.seq for r in rows] == [1, 2]


def test_repeated_flush_does_not_duplicate(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "s2")

    runner.emit("tool", {"path": "/a"})
    runner.persist_trace()
    # A second flush with nothing pending must write nothing.
    assert runner.persist_trace() == 0

    rows = runner.store.load_trace_events("s2")  # type: ignore[union-attr]
    assert len(rows) == 1


def test_restore_trace_replays_in_order(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "s3")
    for path in ("/tool", "/tool/os", "/tool/os/file"):
        runner.emit("tool", {"path": path})
    runner.persist_trace()

    # Simulate a restart: same database, fresh in-memory state.
    fresh = _runner(tmp_path, "s3")
    fresh._recent.clear()
    fresh._seq = 0
    assert fresh.restore_trace()

    paths = [e.payload.get("path") for e in fresh._recent]
    assert paths == ["/tool", "/tool/os", "/tool/os/file"]
    assert fresh._seq == 3, "seq must continue where the store left off"


def test_subscribe_after_seq_skips_the_backlog(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "s4")
    runner.emit("tool", {"path": "/one"})
    runner.emit("tool", {"path": "/two"})

    # A client that already loaded both events must receive nothing older.
    queue = runner.subscribe(after_seq=2)
    assert queue.empty()

    # A client that loaded only the first receives the second.
    queue = runner.subscribe(after_seq=1)
    assert not queue.empty()
    assert queue.get_nowait().payload["path"] == "/two"


def test_restored_runner_seq_matches_store(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "s5")
    runner.emit("tool", {"path": "/x"})
    runner.persist_trace()

    fresh = _runner(tmp_path, "s5")
    fresh.restore_trace()
    fresh.emit("tool", {"path": "/y"})
    fresh.persist_trace()

    rows = fresh.store.load_trace_events("s5")  # type: ignore[union-attr]
    assert [r.seq for r in rows] == [1, 2], "seq must not restart after a restore"
