"""默认配置一致性测试。

确保 ``harness.defaults`` 与各模块实际常量、以及文档清单三方一致，
避免"文档写一套、代码跑另一套"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import (
    compressor,
    executor,
    inbox,
    jev,
    loop,
    memory,
    session,
    trace,
    truncate,
    watermark,
)
from harness.defaults import (
    DEFAULT_CONTEXT,
    DEFAULT_EXECUTION,
    DEFAULT_LOOP,
    DEFAULT_MEMORY,
    DEFAULT_RATE,
    DEFAULT_SESSIONS,
    DEFAULT_TOOLS,
    as_dict,
)

DOC = Path(__file__).resolve().parent.parent / "docs" / "04-初始配置清单.md"


class TestMatchesModuleConstants:
    def test_watermark_thresholds(self) -> None:
        assert DEFAULT_CONTEXT.auto_1 == watermark.AUTO_1
        assert DEFAULT_CONTEXT.auto_2 == watermark.AUTO_2
        assert DEFAULT_CONTEXT.session_end == watermark.SESSION_END
        assert DEFAULT_CONTEXT.chars_per_token == watermark.CHARS_PER_TOKEN

    def test_truncate_limit(self) -> None:
        assert DEFAULT_CONTEXT.max_obs_chars == truncate.DEFAULT_MAX_CHARS

    def test_keep_records(self) -> None:
        assert DEFAULT_CONTEXT.keep_records_50 == compressor.default_keep_records(32768, 50)
        assert DEFAULT_CONTEXT.keep_records_75 == compressor.default_keep_records(32768, 75)

    def test_loop_limits(self) -> None:
        assert DEFAULT_LOOP.max_steps == loop.MAX_STEPS
        assert DEFAULT_LOOP.max_injects == loop.MAX_INJECTS
        assert DEFAULT_LOOP.deadlock_threshold == loop.DEADLOCK_THRESHOLD

    def test_execution_limits(self) -> None:
        assert DEFAULT_EXECUTION.max_workers == executor.DEFAULT_MAX_WORKERS
        assert DEFAULT_EXECUTION.chat_capacity == inbox.CHAT_CAPACITY

    def test_route_thresholds(self) -> None:
        assert DEFAULT_TOOLS.route_tool_count == jev.ROUTE_ENABLE_TOOL_COUNT
        assert DEFAULT_TOOLS.route_branch_count == jev.ROUTE_ENABLE_BRANCH_COUNT
        assert DEFAULT_TOOLS.route_history_tokens == jev.ROUTE_ENABLE_HISTORY_TOKENS
        assert DEFAULT_TOOLS.route_confidence == jev.CONFIDENCE_THRESHOLD

    def test_session_limits(self) -> None:
        assert DEFAULT_SESSIONS.max_resident == session.MAX_RESIDENT
        assert DEFAULT_SESSIONS.idle_timeout_s == session.IDLE_TIMEOUT_S
        assert DEFAULT_SESSIONS.sweep_interval_s == session.SWEEP_INTERVAL_S

    def test_memory_paths(self) -> None:
        assert DEFAULT_MEMORY.agent_dir == memory.AGENT_DIR_NAME
        assert DEFAULT_MEMORY.work_memory == memory.WORK_MEMORY_FILENAME
        assert DEFAULT_MEMORY.long_term_memory == memory.LONG_TERM_MEMORY_FILENAME
        assert DEFAULT_MEMORY.extract_principle == memory.EXTRACT_PRINCIPLE_FILENAME
        assert DEFAULT_MEMORY.trace_dir == trace.TRACE_DIR_NAME

    def test_rate_defaults_match_provider_health(self) -> None:
        from harness.rate import ProviderHealth

        health = ProviderHealth(provider="p")
        assert DEFAULT_RATE.failure_threshold == health.failure_threshold
        assert DEFAULT_RATE.health_backoff_s == health.backoff_s


class TestInvariants:
    def test_thresholds_are_ordered(self) -> None:
        assert DEFAULT_CONTEXT.auto_1 < DEFAULT_CONTEXT.auto_2 < DEFAULT_CONTEXT.session_end
        assert DEFAULT_CONTEXT.session_end <= 1.0

    def test_deeper_watermark_keeps_fewer_records(self) -> None:
        assert DEFAULT_CONTEXT.keep_records_75 < DEFAULT_CONTEXT.keep_records_50

    def test_tree_threshold_below_route_threshold(self) -> None:
        """工具少时全量暴露；达到路由阈值才启用 Jev。"""
        assert DEFAULT_TOOLS.tree_enable_threshold < DEFAULT_TOOLS.route_tool_count

    def test_description_limit_grows_with_depth(self) -> None:
        assert DEFAULT_TOOLS.description_l1 < DEFAULT_TOOLS.description_deep

    def test_sweep_interval_less_than_idle_timeout(self) -> None:
        assert DEFAULT_SESSIONS.sweep_interval_s < DEFAULT_SESSIONS.idle_timeout_s

    def test_positive_values(self) -> None:
        data = as_dict()
        for group, values in data.items():
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    assert value > 0, f"{group}.{name} 必须为正数"


class TestDocumented:
    @pytest.mark.skipif(not DOC.is_file(), reason="配置文档不存在")
    def test_document_covers_every_field(self) -> None:
        """文档清单必须提到每个配置字段名。"""
        text = DOC.read_text(encoding="utf-8")
        missing: list[str] = []
        for group, values in as_dict().items():
            for name in values:
                # 文档使用大写形式或原始字段名
                if name not in text and name.upper() not in text:
                    missing.append(f"{group}.{name}")
        assert not missing, f"文档缺少字段: {missing}"

    @pytest.mark.skipif(not DOC.is_file(), reason="配置文档不存在")
    def test_document_has_required_sections(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for heading in ("环境变量", "超参数清单", "文件布局", "能力集", "启动检查清单"):
            assert heading in text, f"缺少章节: {heading}"


class TestAsDict:
    def test_groups_are_complete(self) -> None:
        data = as_dict()
        assert set(data) == {
            "context",
            "loop",
            "execution",
            "tools",
            "sessions",
            "rate",
            "memory",
        }

    def test_returns_plain_values(self) -> None:
        data = as_dict()
        assert data["context"]["capacity"] == 32768
        assert data["loop"]["max_steps"] == 20

    def test_no_secret_field(self) -> None:
        """默认值里绝不能出现密钥类字段。"""
        flat = {k for group in as_dict().values() for k in group}
        assert not any("key" in name or "secret" in name for name in flat)
