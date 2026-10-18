"""会话层测试（H 组）。

对应文档：docs/02-数据结构设计/02-会话层.md
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from harness.inbox import Inbox, InterruptRequest, UserInput
from harness.session import (
    EvictionPolicy,
    Session,
    SessionRecord,
    SessionRegistry,
    Watermark,
    reset_discovered,
    reset_tree_query,
)


class _FakeDiscovered:
    """最小 DiscoveredTools 替身。"""

    def __init__(self) -> None:
        self.entries: dict[str, object] = {"a": 1}

    def clear(self) -> None:
        self.entries.clear()


class _FakeTreeQuery:
    def __init__(self) -> None:
        self.delivered: set[str] = {"/tool/os"}


def _session(sid: str = "s1", status: str = "idle") -> Session:
    return Session(session_id=sid, working_dir="C:/work", status=status)  # type: ignore[arg-type]


class TestSessionFields:
    def test_defaults(self) -> None:
        session = _session()
        assert session.messages == []
        assert isinstance(session.inbox, Inbox)
        assert isinstance(session.watermark, Watermark)
        assert session.tree_version == 0
        assert session.status == "idle"
        assert session.created_at > 0
        assert session.last_activity >= session.created_at

    def test_touch_updates_last_activity(self) -> None:
        session = _session()
        session.last_activity = time.time() - 100
        before = session.last_activity
        session.touch()
        assert session.last_activity > before

    def test_sessions_do_not_share_mutable_state_H1(self) -> None:
        a, b = _session("a"), _session("b")
        a.messages.append("x")  # type: ignore[arg-type]
        a.watermark.compressed_at.add(50)
        a.tree_query.delivered.add("/tool/x")
        a.inbox.put_chat(UserInput(text="hi", channel="stdin"))

        assert b.messages == []
        assert b.watermark.compressed_at == set()
        assert b.tree_query.delivered == set()
        assert b.inbox.drain_chat() == []

    def test_tree_version_is_fixed_at_creation_H15(self) -> None:
        session = Session(session_id="s", working_dir="w", tree_version=7)
        assert session.tree_version == 7


class TestWatermarkPlaceholder:
    def test_watermark_roundtrip_json(self) -> None:
        wm = Watermark(capacity=128_000, compressed_at={75, 50}, last_estimate=42_000)
        restored = Watermark.from_json(wm.to_json())
        assert restored is not None
        assert restored.capacity == 128_000
        assert restored.compressed_at == {50, 75}
        assert restored.last_estimate == 42_000
        assert wm.to_json()["compressed_at"] == [50, 75]

    def test_from_json_none(self) -> None:
        assert Watermark.from_json(None) is None

    def test_does_not_import_context_layer(self) -> None:
        import harness.session as session_module

        source = Path(session_module.__file__).read_text(encoding="utf-8")
        assert "import harness.context" not in source


class TestResetHelpers:
    def test_reset_tree_query_clears_delivered_H8(self) -> None:
        tree = _FakeTreeQuery()
        reset_tree_query(tree)
        assert tree.delivered == set()

    def test_reset_discovered_uses_clear_when_available_H14(self) -> None:
        discovered = _FakeDiscovered()
        assert reset_discovered(discovered) is None
        assert discovered.entries == {}

    def test_reset_discovered_falls_back_to_new_instance_H14(self) -> None:
        class NoClear:
            def __init__(self, entries: dict[str, object] | None = None) -> None:
                self.entries = entries if entries is not None else {"a": 1}

        replacement = reset_discovered(NoClear())  # type: ignore[arg-type]
        assert replacement is not None
        assert replacement.entries == {}


class TestRegistryBasics:
    def test_add_get_remove(self) -> None:
        registry = SessionRegistry()
        session = _session()
        registry.add(session)

        assert registry.get("s1") is session
        assert "s1" in registry
        assert len(registry) == 1

        assert registry.remove("s1") is session
        assert registry.get("s1") is None
        assert registry.remove("s1") is None
        assert len(registry) == 0

    def test_default_policy(self) -> None:
        policy = SessionRegistry().policy
        assert policy.max_resident == 20
        assert policy.idle_timeout_s == 1800
        assert policy.sweep_interval_s == 60

    def test_end_removes_immediately_H12(self) -> None:
        registry = SessionRegistry()
        registry.add(_session("s1"))
        ended = registry.end("s1")
        assert ended is not None
        assert ended.status == "ended"
        assert registry.get("s1") is None
        assert registry.end("missing") is None

    def test_session_record_fields(self) -> None:
        record = SessionRecord(
            session_id="s1",
            working_dir="w",
            created_at=1.0,
            updated_at=2.0,
            status="idle",
            watermark=Watermark(capacity=1),
        )
        assert record.status == "idle"
        assert record.watermark is not None


class TestEviction:
    def test_active_never_evicted_H10(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=1))
        active = _session("active", status="active")
        active.last_activity = 0.0
        registry.add(active)
        registry.add(_session("idle-late"))
        registry.add(_session("idle-latest"))

        evicted = registry.sweep()

        # max_resident=1，驻留 active 已占满名额 → 两个 idle 均被卸载
        assert registry.get("active") is active
        assert evicted == ["idle-late", "idle-latest"]
        assert len(registry) == 1

    def test_end_is_removed_regardless_of_residency_H12(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=100))
        registry.add(_session("a", status="ended"))
        registry.add(_session("b", status="active"))

        evicted = registry.sweep()

        assert evicted == ["a"]
        assert registry.get("a") is None
        assert registry.get("b") is not None

    def test_idle_with_non_empty_channels_not_evicted_H11(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=1))
        busy = _session("busy")
        busy.last_activity = 0.0
        busy.inbox.put_chat(UserInput(text="pending", channel="stdin"))
        registry.add(busy)
        registry.add(_session("new", status="active"))

        evicted = registry.sweep()

        assert evicted == []
        assert registry.get("busy") is busy
        assert len(registry) == 2  # 无可卸载对象时允许超过上限

    def test_idle_with_control_signal_not_evicted_H11(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=1))
        busy = _session("busy")
        busy.last_activity = 0.0
        busy.inbox.put_control(InterruptRequest(reason="timeout"))
        registry.add(busy)
        registry.add(_session("new", status="active"))

        assert registry.sweep() == []
        assert registry.get("busy") is busy

    def test_idle_empty_evicted_by_last_activity_order_H11(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=2))
        oldest = _session("oldest")
        oldest.last_activity = 1.0
        middle = _session("middle")
        middle.last_activity = 2.0
        newest = _session("newest")
        newest.last_activity = 3.0
        for s in (oldest, middle, newest):
            registry.add(s)

        evicted = registry.sweep()

        assert evicted == ["oldest"]
        assert registry.session_ids() == ["middle", "newest"]

    def test_sweep_under_limit_is_noop(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=20))
        registry.add(_session("a"))
        registry.add(_session("b"))
        assert registry.sweep() == []
        assert len(registry) == 2

    def test_eviction_keeps_database_records_H13(self, tmp_path) -> None:
        from harness.session_store import SessionStore

        store = SessionStore(tmp_path)
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=1))
        victim = _session("victim")
        victim.last_activity = 0.0
        registry.add(victim)
        store.upsert_session(
            SessionRecord(
                session_id="victim",
                working_dir=str(tmp_path),
                created_at=victim.created_at,
                updated_at=victim.last_activity,
                status="idle",
            )
        )
        registry.add(_session("new", status="active"))

        assert registry.sweep() == ["victim"]
        assert registry.get("victim") is None

        # 卸载仅丢内存对象，数据库记录仍在（H13），可重建（H14）
        restored = store.restore("victim")
        assert restored is not None
        assert restored.record.session_id == "victim"
        assert restored.delivered == set()
        assert restored.discovered == {}
        store.close()


class TestRegistryConcurrency:
    def test_concurrent_add_get_remove_is_consistent(self) -> None:
        registry = SessionRegistry()
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(200):
                    sid = f"{worker_id}-{i}"
                    registry.add(Session(session_id=sid, working_dir="w"))
                    assert registry.get(sid) is not None
                    registry.remove(sid)
            except BaseException as exc:  # pragma: no cover - 失败才触发
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(registry) == 0

    def test_sweep_and_add_concurrently_H2(self) -> None:
        registry = SessionRegistry(policy=EvictionPolicy(max_resident=10))
        stop = threading.Event()

        def sweeper() -> None:
            while not stop.is_set():
                registry.sweep()
                time.sleep(0.001)

        def adder() -> None:
            for i in range(200):
                registry.add(Session(session_id=f"c{i}", working_dir="w"))

        sweep_thread = threading.Thread(target=sweeper)
        sweep_thread.start()
        try:
            adder()
        finally:
            stop.set()
            sweep_thread.join()

        # 加完后不再并发，最终状态可控
        assert len(registry) <= 200
        assert all(isinstance(sid, str) for sid in registry.session_ids())


def test_eviction_policy_defaults() -> None:
    policy = EvictionPolicy()
    assert (policy.max_resident, policy.idle_timeout_s, policy.sweep_interval_s) == (20, 1800, 60)


@pytest.mark.parametrize("status", ["active", "idle"])
def test_status_literal_accepts_documented_values(status: str) -> None:
    session = _session(status=status)
    assert session.status == status
