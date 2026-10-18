"""Web 层会话持久化测试（P1）：落盘、恢复与状态同步。

覆盖三个此前完全断开的环节：

* ``SessionManager.create`` / ``delete`` 是否写 ``sessions`` 表；
* ``SessionRunner`` 是否按 ``ExecGroup`` 边界把消息写进 ``messages`` 表（H7）；
* 进程重启后 ``SessionManager.restore`` 能否把会话与历史读回内存。

约束引用：H6（``SystemMsg`` 永不落盘）、H7（事务单元为一个 ``ExecGroup``）、
H13（删除标记为 ``ended`` 而不是删行）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.server.sessions import SessionManager

from harness.message import AssistantMsg, ToolCall, ToolMsg, UserMsg
from harness.model import ProviderConfig, ReverseProxy
from harness.rate import RateWindow
from harness.session_store import SessionStore

Reply = tuple[str | None, list[dict[str, Any]]]


class FakeTransport:
    """按脚本返回响应的假传输，避免真实网络调用。"""

    def __init__(self, replies: list[Reply] | None = None) -> None:
        self.replies = list(replies or [])

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> dict[str, Any]:
        content, calls = self.replies.pop(0) if self.replies else ("完成", [])
        return {
            "model": json.get("model", "fake"),
            "choices": [{"message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _proxy(replies: list[Reply] | None = None) -> ReverseProxy:
    return ReverseProxy(
        providers=[
            ProviderConfig(
                name="fake",
                base_url="https://fake.invalid",
                model="fake-model",
                api_key="sk-fake",
            )
        ],
        rate_window=RateWindow(max_requests=100, max_tokens=1_000_000),
        health={},
        transport=FakeTransport(replies),
    )


def _manager(tmp_path: Path, replies: list[Reply] | None = None) -> SessionManager:
    """构造一个写盘到 ``tmp_path`` 的会话管理器。"""
    return SessionManager(
        workdir=str(tmp_path),
        proxy=_proxy(replies),
        default_model="fake-model",
        capacity=10_000,
    )


def _db(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path)


class TestSessionRowPersistence:
    def test_create_writes_session_row(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()

        record = _db(tmp_path).get_session(runner.session_id)
        assert record is not None
        assert record.status == "idle"
        assert record.working_dir == str(tmp_path)

    def test_delete_marks_ended_without_dropping_history(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()
        runner.messages.append(UserMsg(content="要保留的历史"))
        runner.persist_pending()

        assert manager.delete(runner.session_id) is True

        store = _db(tmp_path)
        record = store.get_session(runner.session_id)
        # H13：删除只标记 ended，行与消息都留在库里
        assert record is not None
        assert record.status == "ended"
        assert [m.content for m in store.load_messages(runner.session_id)] == ["要保留的历史"]

    def test_running_state_is_persisted_as_active(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()

        runner._set_state("running")
        assert _db(tmp_path).get_session(runner.session_id).status == "active"

        runner._set_state("done")
        assert _db(tmp_path).get_session(runner.session_id).status == "idle"

    def test_missing_store_degrades_to_memory_only(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        manager.store = None
        runner = manager.create()

        # 不抛异常，且内存消息照常工作
        assert runner.persist_pending() == 0
        runner.messages.append(UserMsg(content="hello"))
        assert runner.persist_pending() == 0
        assert manager.delete(runner.session_id) is True


class TestMessagePersistence:
    def test_system_message_is_never_persisted(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()
        runner.messages.append(UserMsg(content="hi"))
        runner.persist_pending()

        stored = _db(tmp_path).load_messages(runner.session_id)
        # H6：SystemMsg 不落盘
        assert [m.type for m in stored] == ["user"]
        assert stored[0].content == "hi"

    def test_flush_is_incremental(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()

        runner.messages.append(UserMsg(content="第一轮"))
        assert runner.persist_pending() == 1
        runner.messages.append(AssistantMsg(content="回答"))
        assert runner.persist_pending() == 1
        # 没有新消息时再刷一次不应重复写入
        assert runner.persist_pending() == 0

        store = _db(tmp_path)
        assert [m.seq for m in store.load_messages(runner.session_id)] == [1, 2]
        assert store.next_seq(runner.session_id) == 3

    def test_group_boundary_writes_assistant_and_tool_results(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()
        runner.messages.append(UserMsg(content="读文件"))

        call = ToolCall(id="c1", name="/fs/read", arguments='{"path": "a.txt"}')
        runner.messages.append(AssistantMsg(content=None, tool_calls=[call]))
        runner.messages.append(
            ToolMsg(tool_call_id="c1", name="/fs/read", content="内容一")
        )
        runner.persist_pending()

        stored = _db(tmp_path).load_messages(runner.session_id)
        # H7：一个 ExecGroup 的 assistant 与其 tool 结果在一次事务里落盘
        assert [m.type for m in stored] == ["user", "assistant", "tool"]
        assert stored[1].tool_calls is not None

    def test_restore_roundtrip_rebuilds_history(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        runner = manager.create()
        runner.messages.append(UserMsg(content="问题"))
        runner.messages.append(AssistantMsg(content="答案"))
        runner.persist_pending()
        session_id = runner.session_id

        restored = _db(tmp_path).restore(session_id)
        assert restored is not None
        assert [(type(m).__name__, getattr(m, "content", None)) for m in restored.messages] == [
            ("UserMsg", "问题"),
            ("AssistantMsg", "答案"),
        ]


class TestRestoreOnStartup:
    def test_restore_rebuilds_sessions_and_messages(self, tmp_path: Path) -> None:
        first = _manager(tmp_path)
        runner = first.create()
        runner.messages.append(UserMsg(content="重启前的问题"))
        runner.messages.append(AssistantMsg(content="重启前的回答"))
        runner.persist_pending()
        session_id = runner.session_id

        # 模拟进程重启：全新的 manager 读同一个 workdir
        second = _manager(tmp_path)
        restored = second.restore()

        assert [r.session_id for r in restored] == [session_id]
        revived = second.get(session_id)
        assert revived is not None
        # 第一个元素是重建出来的 SystemMsg（H6），其余是历史
        assert isinstance(revived.messages[0], type(revived.system))
        assert [m.content for m in revived.messages[1:]] == ["重启前的问题", "重启前的回答"]
        assert revived.state == "idle"

    def test_restored_cursor_prevents_duplicate_writes(self, tmp_path: Path) -> None:
        first = _manager(tmp_path)
        runner = first.create()
        runner.messages.append(UserMsg(content="只此一条"))
        runner.persist_pending()
        session_id = runner.session_id

        # 重启后不得把已落盘的历史再写一遍
        second = _manager(tmp_path)
        second.restore()
        revived = second.get(session_id)
        assert revived is not None
        assert revived.persist_pending() == 0

        revived.messages.append(UserMsg(content="新增一条"))
        assert revived.persist_pending() == 1
        assert [m.content for m in _db(tmp_path).load_messages(session_id)] == [
            "只此一条",
            "新增一条",
        ]

    def test_restore_survives_ended_session_and_skips_it(self, tmp_path: Path) -> None:
        first = _manager(tmp_path)
        runner = first.create()
        runner.messages.append(UserMsg(content="将被删除"))
        runner.persist_pending()
        first.delete(runner.session_id)

        second = _manager(tmp_path)
        assert second.restore() == []
        assert second.get(runner.session_id) is None
        # 历史仍在库里（H13）
        assert _db(tmp_path).get_session(runner.session_id).status == "ended"

    def test_restore_isolated_per_session(self, tmp_path: Path) -> None:
        first = _manager(tmp_path)
        a = first.create()
        a.messages.append(UserMsg(content="A 的消息"))
        a.persist_pending()
        b = first.create()
        b.messages.append(UserMsg(content="B 的消息"))
        b.persist_pending()

        second = _manager(tmp_path)
        restored = second.restore()
        assert len(restored) == 2
        assert [m.content for m in second.get(a.session_id).messages[1:]] == ["A 的消息"]
        assert [m.content for m in second.get(b.session_id).messages[1:]] == ["B 的消息"]

    def test_restore_without_database_is_empty(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        # 从未创建过会话时，库里没有会话行
        fresh = _manager(tmp_path)
        assert fresh.restore() == []
        assert manager.list() == []


class TestEmptyTurn:
    def test_user_message_persisted_before_reasoning(self, tmp_path: Path) -> None:
        """用户输入必须在推理开始前落盘，崩溃时不丢请求。"""
        manager = _manager(tmp_path, replies=[("回答", [])])
        runner = manager.create()

        runner.start("必须留存的问题")
        runner._thread.join(timeout=5)

        contents = [m.content for m in _db(tmp_path).load_messages(runner.session_id)]
        assert contents[0] == "必须留存的问题"
        assert "回答" in contents
