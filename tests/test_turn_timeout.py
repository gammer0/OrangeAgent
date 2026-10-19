"""推理线程的超时保护。

daemon 线程阻塞在网络调用上时既不会被回收，也不会自己退出；而 ``sweep``
只清理 ``idle`` 会话，卡死的 ``active`` 会话永远轮不到。于是每挂一次就永久
损失一个 runner 槽位。这组测试固定住守卫的三条行为：
  * 只有「运行中且长时间无进展」才被判定超时；
  * 有增量或批次完成就续命，长回答不会被误杀；
  * 一轮结束后锚点清空，不影响下一轮。
"""

from __future__ import annotations

import threading
from pathlib import Path

from apps.server.sessions import DemoToolRegistry, SessionRunner


def _runner(tmp_path: Path, session_id: str = "s1") -> SessionRunner:
    return SessionRunner(
        session_id=session_id,
        workdir=str(tmp_path),
        proxy=None,  # never called: the tests drive the guard directly
        tools=DemoToolRegistry(str(tmp_path)),
        model="fake",
        turn_timeout_s=60.0,
    )


def _fake_running(runner: SessionRunner) -> None:
    """Pretend a turn is in flight without spawning a real one."""

    def wait() -> None:
        threading.Event().wait(30)

    runner._thread = threading.Thread(target=wait, daemon=True)
    runner._thread.start()
    runner.state = "running"


def test_idle_session_is_never_reaped(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    # No turn at all: nothing to reap, regardless of the clock.
    assert runner.stalled_for(now=10_000.0) is None
    assert runner.reap_if_stalled(now=10_000.0) is False


def test_stall_under_threshold_is_ignored(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _fake_running(runner)
    runner._turn_started_at = 100.0
    runner._last_progress_at = 100.0

    # 30 s of silence with a 60 s budget: still healthy.
    assert runner.stalled_for(now=130.0) == 30.0
    assert runner.reap_if_stalled(now=130.0) is False
    assert runner.state == "running"


def test_progress_resets_the_timer(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _fake_running(runner)
    runner._turn_started_at = 100.0
    runner._last_progress_at = 100.0

    # A delta arrives at t=200, so the turn is alive again.
    runner._on_delta("x")
    assert runner.stalled_for(now=200.0) < 60.0
    assert runner.reap_if_stalled(now=200.0) is False


def test_stalled_turn_is_reaped_and_session_becomes_usable(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _fake_running(runner)
    runner._turn_started_at = 100.0
    runner._last_progress_at = 100.0

    reaped = runner.reap_if_stalled(now=1000.0)
    assert reaped is True
    assert runner.state == "error", "the UI must leave the running state"
    assert runner._turn_started_at is None
    assert runner._cancelled is True

    # The next turn is judged by its own timestamps, not the dead one's.
    assert runner.stalled_for(now=1001.0) is None


def test_emit_reports_timeout_to_the_frontend(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _fake_running(runner)
    runner._turn_started_at = 100.0
    runner._last_progress_at = 100.0
    subscription = runner.subscribe()

    runner.reap_if_stalled(now=1000.0)

    kinds = []
    while not subscription.empty():
        kinds.append(subscription.get_nowait().kind)
    assert "error" in kinds
    assert "state" in kinds


def test_manager_watchdog_reaps_every_stalled_runner(tmp_path: Path) -> None:
    from apps.server.sessions import SessionManager

    manager = SessionManager(
        workdir=str(tmp_path),
        proxy=None,  # type: ignore[arg-type]
        default_model="fake",
    )
    try:
        stuck = _runner(tmp_path, "stuck")
        fresh = _runner(tmp_path, "fresh")
        _fake_running(stuck)
        stuck._turn_started_at = 100.0
        stuck._last_progress_at = 100.0
        _fake_running(fresh)
        fresh._turn_started_at = 990.0
        fresh._last_progress_at = 990.0
        manager._sessions = {"stuck": stuck, "fresh": fresh}

        reaped = manager.watch_stalls(now=1000.0)

        assert reaped == ["stuck"]
        assert fresh.state == "running", "a healthy turn must not be touched"
    finally:
        if manager.registry is not None:
            manager.registry.stop_sweeper()
        manager.stop_watchdog()
