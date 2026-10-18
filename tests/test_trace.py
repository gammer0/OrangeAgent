"""轨迹日志测试（E 组：E1/E2/E3）。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.trace import TraceEvent, TraceLogger, trace_file_path


def _logger(tmp_path: Path, session_id: str = "s1", **kwargs: object) -> TraceLogger:
    return TraceLogger(tmp_path / "traces", session_id, **kwargs)


class TestTraceEvent:
    def test_e3_session_id_required(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(session_id="", step=0, ts=1.0, kind="inject")

    def test_kind_whitelist(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(session_id="s", step=0, ts=1.0, kind="unknown")  # type: ignore[arg-type]

    def test_all_seven_kinds_accepted(self) -> None:
        for kind in (
            "llm_call",
            "tool_exec",
            "compress",
            "inject",
            "error",
            "abort",
            "memory_extract",
        ):
            TraceEvent(session_id="s", step=0, ts=1.0, kind=kind)  # type: ignore[arg-type]

    def test_e2_json_line_roundtrip(self) -> None:
        event = TraceEvent(session_id="s", step=3, ts=1.5, kind="error", payload={"a": "中文"})
        line = event.to_json_line()
        assert "\n" not in line
        assert json.loads(line)["payload"]["a"] == "中文"
        assert TraceEvent.from_json_line(line) == event

    def test_step_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(session_id="s", step=-1, ts=1.0, kind="inject")


class TestTraceLoggerFiles:
    def test_e3_one_file_per_session(self, tmp_path: Path) -> None:
        a = _logger(tmp_path, "alpha")
        b = _logger(tmp_path, "beta")
        a.log_error(0, error_type="X", message="m", retry_count=0)
        b.log_error(0, error_type="Y", message="m", retry_count=0)
        assert a.path != b.path
        assert a.path.name == "alpha.jsonl"
        assert "X" in a.path.read_text(encoding="utf-8")
        assert "Y" not in a.path.read_text(encoding="utf-8")

    def test_e3_session_id_sanitised(self, tmp_path: Path) -> None:
        assert trace_file_path(tmp_path, "../evil").name == ".._evil.jsonl"
        assert trace_file_path(tmp_path, "../evil").parent == tmp_path

    def test_rejects_empty_session_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            TraceLogger(tmp_path, "")

    def test_every_event_carries_session_id(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path, "sess-42")
        logger.log_llm_call(
            0, model="m", tokens_in=1, tokens_out=2, duration_ms=3.0, tool_call_count=0
        )
        for raw in logger.path.read_text(encoding="utf-8").splitlines():
            assert json.loads(raw)["session_id"] == "sess-42"


class TestE1ImmediateFlush:
    def test_line_visible_before_object_gone(self, tmp_path: Path) -> None:
        """每步结束即可从文件读到该行（不缓存到最后）。"""
        logger = _logger(tmp_path)
        logger.log_inject(0, channel="chat", content_chars=3)
        lines = logger.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

        logger.log_tool_exec(
            1,
            path="/tool/os/file/read",
            tool_id="c1",
            arguments={"path": "a"},
            ok=True,
            error_type=None,
            duration_ms=1.0,
        )
        assert len(logger.path.read_text(encoding="utf-8").splitlines()) == 2

    def test_steps_are_appended_in_order(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        for step in range(5):
            logger.log_abort(step, reason="user", coalesced=False)
        steps = [e.step for e in logger.read_events()]
        assert steps == [0, 1, 2, 3, 4]


class TestE2Parsable:
    def test_every_line_is_complete_json(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        logger.log_llm_call(
            0, model="m", tokens_in=1, tokens_out=2, duration_ms=3.0, tool_call_count=1
        )
        logger.log_compress(0, level=1, before_tokens=100, after_tokens=40, summary_chars=7)
        logger.log_memory_extract(1, status="done", entries_added=2, duration_ms=9.0)
        raw = logger.path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
        assert len(logger.read_events()) == 3

    def test_payload_fields_follow_doc_7_2(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        logger.log_llm_call(
            0, model="gpt", tokens_in=10, tokens_out=20, duration_ms=30.0, tool_call_count=2
        )
        logger.log_tool_exec(
            1,
            path="/tool/os",
            tool_id="c1",
            arguments="{}",
            ok=False,
            error_type="ToolError",
            duration_ms=1.0,
        )
        logger.log_compress(2, level=2, before_tokens=9, after_tokens=3, summary_chars=4)
        logger.log_inject(2, channel="control", content_chars=5)
        logger.log_error(3, error_type="LLMError", message="boom", retry_count=1)
        logger.log_abort(3, reason="stop", coalesced=True)
        logger.log_memory_extract(4, status="failed", entries_added=0, duration_ms=0.5)

        events = logger.read_events()
        assert [e.kind for e in events] == [
            "llm_call",
            "tool_exec",
            "compress",
            "inject",
            "error",
            "abort",
            "memory_extract",
        ]
        assert set(events[0].payload) == {
            "model",
            "tokens_in",
            "tokens_out",
            "duration_ms",
            "tool_call_count",
        }
        assert set(events[1].payload) == {
            "path",
            "tool_id",
            "arguments",
            "ok",
            "error_type",
            "duration_ms",
        }
        assert set(events[2].payload) == {
            "level",
            "before_tokens",
            "after_tokens",
            "summary_chars",
        }
        assert set(events[3].payload) == {"channel", "content_chars"}
        assert set(events[4].payload) == {"error_type", "message", "retry_count"}
        assert set(events[5].payload) == {"reason", "coalesced"}
        assert set(events[6].payload) == {"status", "entries_added", "duration_ms"}


class TestConcurrency:
    def test_multi_thread_writes_are_not_interleaved(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        threads: list[threading.Thread] = []

        def worker(index: int) -> None:
            for step in range(40):
                logger.log_error(
                    step,
                    error_type=f"T{index}",
                    message="x" * 200,
                    retry_count=index,
                )

        for i in range(8):
            threads.append(threading.Thread(target=worker, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = logger.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8 * 40
        for line in lines:
            parsed = json.loads(line)  # 任一行不可解析即失败
            assert parsed["session_id"] == "s1"
            assert parsed["payload"]["message"] == "x" * 200

    def test_fsync_option_still_parsable(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path, fsync=True)
        logger.log_inject(0, channel="chat", content_chars=1)
        assert len(logger.read_events()) == 1
