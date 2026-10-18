"""模型层运行时测试（K 组：K1/K2/K3/K4/K5）。"""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from harness.rate import (
    ProviderHealth,
    RateEvent,
    RateWindow,
    get_provider_health,
    reset_provider_health_registry,
)


class TestRateEvent:
    def test_k2_session_id_required(self) -> None:
        with pytest.raises(ValidationError):
            RateEvent(ts=0.0, session_id="", tokens=1)

    def test_actual_tokens_defaults_to_none(self) -> None:
        assert RateEvent(ts=0.0, session_id="a", tokens=3).actual_tokens is None


class TestRateWindow:
    def test_k2_every_event_has_session_id(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=1000)
        window.try_acquire("a", 10, now=0.0)
        window.try_acquire("b", 10, now=0.5)
        assert [e.session_id for e in window.events] == ["a", "b"]

    def test_rejects_empty_session_id(self) -> None:
        window = RateWindow(max_requests=1, max_tokens=1)
        with pytest.raises(ValueError):
            window.try_acquire("", 0, now=0.0)

    def test_request_limit(self) -> None:
        window = RateWindow(max_requests=2, max_tokens=1000)
        assert window.try_acquire("a", 1, now=0.0)
        assert window.try_acquire("a", 1, now=0.1)
        assert not window.try_acquire("a", 1, now=0.2)
        assert len(window) == 2

    def test_token_limit(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=100)
        assert window.try_acquire("a", 60, now=0.0)
        assert not window.try_acquire("b", 60, now=0.1)
        assert window.try_acquire("b", 40, now=0.2)
        assert len(window) == 2

    def test_k3_eviction_only_by_ts(self) -> None:
        """不同会话的事件同样按 ts 滑出；会话不参与淘汰（K3）。"""
        window = RateWindow(max_requests=2, max_tokens=1000, window_s=60)
        assert window.try_acquire("a", 1, now=0.0)
        assert window.try_acquire("b", 1, now=10.0)
        assert not window.try_acquire("c", 1, now=20.0)

        # now=61 时 a(ts=0) 滑出，与它属于哪个会话无关
        assert window.try_acquire("c", 1, now=61.0)
        assert [e.session_id for e in window.events] == ["b", "c"]

    def test_k3_all_old_events_evicted_even_single_session(self) -> None:
        window = RateWindow(max_requests=2, max_tokens=1000, window_s=60)
        window.try_acquire("only", 1, now=0.0)
        window.try_acquire("only", 1, now=1.0)
        assert window.try_acquire("only", 1, now=70.0)
        assert [e.ts for e in window.events] == [70.0]

    def test_k3_token_budget_freed_after_sliding(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=100, window_s=60)
        assert window.try_acquire("a", 100, now=0.0)
        assert not window.try_acquire("b", 100, now=30.0)
        assert window.try_acquire("b", 100, now=61.0)

    def test_k3_events_sorted_by_ts(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=1000)
        window.try_acquire("a", 1, now=5.0)
        window.try_acquire("b", 1, now=6.0)
        window.try_acquire("c", 1, now=7.0)
        ts = [e.ts for e in window.events]
        assert ts == sorted(ts)

    def test_k2_record_actual_backfills_same_session_only(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=1000)
        window.try_acquire("a", 10, now=0.0)
        window.try_acquire("b", 10, now=1.0)

        assert window.record_actual("b", 999) is True
        events = {e.session_id: e for e in window.events}
        assert events["b"].actual_tokens == 999
        assert events["a"].actual_tokens is None

        assert window.record_actual("a", 7) is True
        assert {e.session_id: e.actual_tokens for e in window.events} == {"a": 7, "b": 999}

    def test_record_actual_without_pending_returns_false(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=1000)
        window.try_acquire("a", 1, now=0.0)
        assert window.record_actual("a", 1)
        assert not window.record_actual("a", 1)
        assert not window.record_actual("ghost", 1)

    def test_record_actual_uses_actual_for_budget(self) -> None:
        window = RateWindow(max_requests=10, max_tokens=100, window_s=60)
        assert window.try_acquire("a", 80, now=0.0)
        window.record_actual("a", 20)
        # 回填后按实际 20 计算，剩余 80 可用
        assert window.try_acquire("b", 80, now=1.0)

    def test_rejected_request_not_recorded(self) -> None:
        window = RateWindow(max_requests=1, max_tokens=10)
        assert window.try_acquire("a", 1, now=0.0)
        assert not window.try_acquire("b", 100, now=0.1)
        assert len(window) == 1

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            RateWindow(max_requests=0, max_tokens=1)

    def test_thread_safety_never_exceeds_limit(self) -> None:
        window = RateWindow(max_requests=20, max_tokens=10_000)
        granted: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            ok = window.try_acquire("s", 1, now=0.0)
            with lock:
                granted.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(60)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(granted) == 20
        assert len(window) == 20


class TestProviderHealth:
    def test_initial_state_healthy(self) -> None:
        health = ProviderHealth("p")
        assert health.state == "healthy"
        assert health.can_send(now=0.0) == "allow"

    def test_success_resets(self) -> None:
        health = ProviderHealth("p", failure_threshold=2)
        health.on_failure(now=0.0)
        health.on_failure(now=1.0)
        assert health.state == "degraded"
        health.on_success()
        assert health.state == "healthy"
        assert health.consecutive_failures == 0
        assert health.retry_after == 0.0

    def test_degraded_waits_retry_after(self) -> None:
        health = ProviderHealth("p", failure_threshold=1, backoff_s=30.0)
        health.on_failure(now=100.0)
        assert health.state == "degraded"
        assert health.retry_after == 130.0
        assert health.can_send(now=110.0) == "retry_after"
        assert health.can_send(now=130.0) == "allow"

    def test_down_fails_fast(self) -> None:
        health = ProviderHealth("p")
        health.mark_down(switched_to="backup")
        assert health.can_send(now=1e9) == "fail_fast"
        assert health.switched_to == "backup"

    def test_k5_degraded_failures_not_accumulated(self) -> None:
        health = ProviderHealth("p", failure_threshold=1, backoff_s=30.0)
        health.on_failure(now=0.0)
        assert health.consecutive_failures == 1
        health.on_failure(now=1.0)
        health.on_failure(now=2.0)
        assert health.consecutive_failures == 1
        assert health.state == "degraded"
        # 退避时间被刷新，持续生效
        assert health.retry_after == 32.0

    def test_k5_down_failures_not_counted(self) -> None:
        health = ProviderHealth("p")
        health.mark_down()
        for i in range(5):
            health.on_failure(now=float(i))
        assert health.consecutive_failures == 0
        assert health.state == "down"

    def test_healthy_accumulates_until_threshold(self) -> None:
        health = ProviderHealth("p", failure_threshold=3)
        assert health.on_failure(now=0.0) == "healthy"
        assert health.on_failure(now=1.0) == "healthy"
        assert health.consecutive_failures == 2
        assert health.on_failure(now=2.0) == "degraded"
        assert health.consecutive_failures == 3

    def test_threshold_argument_overrides(self) -> None:
        health = ProviderHealth("p", failure_threshold=10)
        assert health.on_failure(now=0.0, threshold=1) == "degraded"

    def test_k4_global_singleton(self) -> None:
        reset_provider_health_registry()
        try:
            a = get_provider_health("deepseek")
            b = get_provider_health("deepseek")
            assert a is b
            a.mark_down(switched_to="backup")
            assert b.state == "down"
            assert get_provider_health("other") is not a
        finally:
            reset_provider_health_registry()

    def test_k4_global_state_visible_across_sessions(self) -> None:
        reset_provider_health_registry()
        try:
            # 会话 A 的失败影响会话 B 的发送决策（全局信号）
            health = get_provider_health("p", failure_threshold=1, backoff_s=60.0)
            health.on_failure(now=0.0)
            assert get_provider_health("p").can_send(now=1.0) == "retry_after"
        finally:
            reset_provider_health_registry()

    def test_snapshot_is_consistent(self) -> None:
        health = ProviderHealth("p", failure_threshold=1)
        health.on_failure(now=5.0)
        snap = health.snapshot()
        assert snap["state"] == "degraded"
        assert snap["consecutive_failures"] == 1

    def test_thread_safety_failure_counter(self) -> None:
        health = ProviderHealth("p", failure_threshold=1000)
        threads = [threading.Thread(target=health.on_failure, args=(0.0,)) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert health.consecutive_failures == 50
