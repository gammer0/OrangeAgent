"""下钻与轨迹的分工：下钻不进消息库，但轨迹必须落盘。

这两条约束共同定义了审计语义：
  * 消息表是"模型看到的上下文"，下钻脚手架在叶子工具被调用后即被剪除，
    落盘时自然不存在；
  * 轨迹表是"用户看到的审计流"，必须独立持久化，否则重启后无法回答
    "这个工具是怎么被找到的"。
"""

from __future__ import annotations

import json

from orange.session import SessionRecord
from orange.session_store import SessionStore


def _store(tmp_path) -> SessionStore:  # noqa: ANN001
    return SessionStore(db_path=tmp_path / "orange.db")


def _record(session_id: str) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        working_dir=".",
        created_at=1.0,
        updated_at=1.0,
        status="idle",
        watermark=None,
    )


def test_trace_events_survive_reload(tmp_path) -> None:  # noqa: ANN001
    store = _store(tmp_path)
    store.upsert_session(_record("s1"))

    payload = {"path": "/tool/os", "preview": "children", "content": "[]"}
    written = store.append_trace_events("s1", [("tool", payload, 123.0)])
    assert written == 1

    rows = store.load_trace_events("s1")
    assert len(rows) == 1
    assert rows[0].kind == "tool"
    assert rows[0].seq == 1
    assert json.loads(rows[0].payload)["path"] == "/tool/os"


def test_trace_seq_is_monotonic_across_flushes(tmp_path) -> None:  # noqa: ANN001
    store = _store(tmp_path)
    store.upsert_session(_record("s1"))

    store.append_trace_events("s1", [("tool", {"path": "/a"}, 1.0)])
    store.append_trace_events(
        "s1", [("tool", {"path": "/b"}, 2.0), ("assistant", {}, 3.0)]
    )

    rows = store.load_trace_events("s1")
    assert [r.seq for r in rows] == [1, 2, 3]
    assert [json.loads(r.payload).get("path") for r in rows] == ["/a", "/b", None]


def test_trace_is_scoped_per_session(tmp_path) -> None:  # noqa: ANN001
    store = _store(tmp_path)
    store.upsert_session(_record("s1"))
    store.upsert_session(_record("s2"))

    store.append_trace_events("s1", [("tool", {"path": "/a"}, 1.0)])
    store.append_trace_events("s2", [("tool", {"path": "/b"}, 1.0)])

    assert [json.loads(r.payload)["path"] for r in store.load_trace_events("s1")] == [
        "/a"
    ]
    assert [json.loads(r.payload)["path"] for r in store.load_trace_events("s2")] == [
        "/b"
    ]


def test_empty_flush_is_a_noop(tmp_path) -> None:  # noqa: ANN001
    store = _store(tmp_path)
    store.upsert_session(_record("s1"))
    assert store.append_trace_events("s1", []) == 0
    assert store.load_trace_events("s1") == []


def test_trace_survives_a_fresh_connection(tmp_path) -> None:  # noqa: ANN001
    first = _store(tmp_path)
    first.upsert_session(_record("s1"))
    first.append_trace_events("s1", [("tool", {"path": "/tool/os"}, 1.0)])
    first.close()

    second = SessionStore(db_path=tmp_path / "orange.db")
    rows = second.load_trace_events("s1")
    assert len(rows) == 1
    assert json.loads(rows[0].payload)["path"] == "/tool/os"


def test_descent_absent_from_messages_but_present_in_trace(tmp_path) -> None:  # noqa: ANN001
    """The two stores answer different questions and must diverge here.

    A descent turn is pruned from the conversation once the tool it revealed is
    called, so nothing about it reaches the ``messages`` table. The audit trail
    still needs it, which is exactly why the event went to ``trace_events``.
    """
    store = _store(tmp_path)
    store.upsert_session(_record("s1"))

    # What the conversation holds after pruning: the leaf call and its result.
    from orange.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

    leaf = ToolCall(id="c2", name="/tool/os/file/read", arguments='{"path":"a"}')
    store.append_messages(
        "s1",
        [
            UserMsg(content="read a"),
            AssistantMsg(content="", tool_calls=[leaf]),
            ToolMsg(tool_call_id="c2", name="/tool/os/file/read", content="ok"),
        ],
    )
    # What the audit trail holds: the descent events that produced it.
    store.append_trace_events(
        "s1",
        [
            ("tool", {"path": "/tool", "preview": "children"}, 1.0),
            ("tool", {"path": "/tool/os", "preview": "children"}, 2.0),
            ("tool", {"path": "/tool/os/file/read", "preview": "ok"}, 3.0),
        ],
    )

    restored = store.restore("s1")
    names = [
        getattr(m, "name", None)
        for m in restored.messages
        if getattr(m, "name", None) is not None
    ]
    assert names == ["/tool/os/file/read"], "descents must not be in the conversation"

    trace_paths = [
        json.loads(r.payload)["path"] for r in store.load_trace_events("s1")
    ]
    assert trace_paths == ["/tool", "/tool/os", "/tool/os/file/read"], (
        "descents must remain in the audit trail"
    )

