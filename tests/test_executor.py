"""执行器测试：并发、按序回填、异常 3c、安全点回调（D2/D3/D4/D7）。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from harness.execution import (
    BatchPlan,
    ExecResult,
    fill_placeholders,
    plan_batch,
)
from harness.executor import Committed, GroupReport, execute_batch
from harness.message import ToolCall

SAFE = "/tool/os/file/read"
EXCL = "/tool/os/shell/run"


@dataclass
class FakeToolDef:
    """最小 ``ExecToolDef`` 实现（满足 Protocol 即可被注入）。"""

    fn: Any
    concurrency: str = "safe"
    idempotent: bool = False
    schema: dict[str, Any] = field(default_factory=dict)


def _call(call_id: str, name: str = SAFE, arguments: str = "{}") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _resolver(tools: dict[str, FakeToolDef]) -> Any:
    def resolve(call: ToolCall) -> FakeToolDef:
        return tools[call.name]

    return resolve


class TestExecuteBatch:
    def test_all_safe_runs_in_single_parallel_group(self) -> None:
        """D1：全 safe 批次形成单组，结果齐全。"""
        tools = {SAFE: FakeToolDef(fn=lambda **_: "ok")}
        calls = [_call(f"c{i}") for i in range(3)]
        plan = plan_batch(calls, lambda c: "safe")

        results = execute_batch(plan, _resolver(tools))

        assert len(plan.groups) == 1
        assert plan.mode == "parallel"
        assert set(results) == {"c0", "c1", "c2"}
        assert all(r.ok for r in results.values())

    def test_results_follow_group_and_call_order_not_completion(self) -> None:
        """D3：慢调用先提交也按组内顺序回填。"""

        def slow(**_kwargs: Any) -> str:
            time.sleep(0.05)
            return "slow"

        def fast(**_kwargs: Any) -> str:
            return "fast"

        # 使用不同工具名区分，避免 resolver 复用同一 fn
        slow_name = "/tool/os/file/slow"
        fast_name = "/tool/os/file/fast"
        tools = {
            slow_name: FakeToolDef(fn=slow),
            fast_name: FakeToolDef(fn=fast),
        }
        calls = [_call("c1", slow_name), _call("c2", fast_name)]
        plan = plan_batch(calls, lambda c: "safe")

        execute_batch(plan, _resolver(tools))

        assert list(plan.results) == ["c1", "c2"]
        assert plan.results["c1"].content == "slow"
        assert plan.results["c2"].content == "fast"

    def test_parallel_group_actually_concurrent(self) -> None:
        """并行组确实并发：4 个 0.2s 调用总耗时 < 0.7s（宽松断言）。"""

        def sleep_tool(**_kwargs: Any) -> str:
            time.sleep(0.2)
            return "done"

        names = [f"/tool/os/file/s{i}" for i in range(4)]
        tools = {name: FakeToolDef(fn=sleep_tool) for name in names}
        calls = [_call(f"c{i}", names[i]) for i in range(4)]
        plan = plan_batch(calls, lambda c: "safe")

        started = time.perf_counter()
        execute_batch(plan, _resolver(tools), max_workers=4)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.7, f"并行组耗时 {elapsed:.2f}s，疑似串行执行"

    def test_exclusive_group_runs_serially(self) -> None:
        """exclusive 组串行：两个 0.1s 调用耗时 >= 0.2s。"""
        active = 0
        peak = 0
        lock = threading.Lock()

        def work(**_kwargs: Any) -> str:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.1)
            with lock:
                active -= 1
            return "ok"

        tools = {EXCL: FakeToolDef(fn=work, concurrency="exclusive")}
        calls = [_call("c1", EXCL), _call("c2", EXCL)]
        plan = plan_batch(calls, lambda c: "exclusive")

        execute_batch(plan, _resolver(tools))

        assert [g.mode for g in plan.groups] == ["serial", "serial"]
        assert peak == 1

    def test_mixed_groups_execute_in_order(self) -> None:
        """safe 组 → exclusive 组 → safe 组，按组序执行。"""
        timeline: list[str] = []
        lock = threading.Lock()

        def make(tag: str) -> Any:
            def fn(**_kwargs: Any) -> str:
                with lock:
                    timeline.append(tag)
                return tag

            return fn

        names = {
            "safe": "/tool/os/file/read",
            "excl": "/tool/os/shell/run",
        }
        tools = {
            names["safe"]: FakeToolDef(fn=make("safe")),
            names["excl"]: FakeToolDef(fn=make("excl"), concurrency="exclusive"),
        }
        calls = [
            _call("c1", names["safe"]),
            _call("c2", names["excl"]),
            _call("c3", names["safe"]),
        ]
        plan = plan_batch(calls, lambda c: tools[c.name].concurrency)

        execute_batch(plan, _resolver(tools))

        assert len(plan.groups) == 3
        assert [g.mode for g in plan.groups] == ["parallel", "serial", "parallel"]
        assert timeline == ["safe", "excl", "safe"]
        assert list(plan.results) == ["c1", "c2", "c3"]

    def test_tool_exception_becomes_3c_with_string_content(self) -> None:
        """D4：工具抛异常 → 3c，content 仍是字符串。"""

        def boom(**_kwargs: Any) -> str:
            raise RuntimeError("kaboom")

        tools = {SAFE: FakeToolDef(fn=boom, idempotent=True)}
        plan = plan_batch([_call("c1")], lambda c: "safe")

        results = execute_batch(plan, _resolver(tools))

        result = results["c1"]
        assert result.ok is False
        assert result.error_type == "3c"
        assert isinstance(result.content, str)
        assert "kaboom" in result.content
        assert json.loads(result.content)["error_type"] == "3c"

    def test_retryable_depends_only_on_idempotent(self) -> None:
        """§4.13：retryable 只由 idempotent 决定，与 concurrency 无关。"""
        cases = [
            ("safe", True, True),
            ("safe", False, False),
            ("exclusive", True, True),
            ("exclusive", False, False),
        ]
        for concurrency, idempotent, expected in cases:
            tools = {
                SAFE: FakeToolDef(
                    fn=lambda **_: "ok",
                    concurrency=concurrency,
                    idempotent=idempotent,
                )
            }
            plan = plan_batch([_call("c1")], lambda c, value=concurrency: value)
            result = execute_batch(plan, _resolver(tools))["c1"]
            assert result.retryable is expected, (concurrency, idempotent)

    def test_retryable_is_set_for_failures_too(self) -> None:
        """失败结果的 retryable 同样由 idempotent 决定（幂等工具可重试）。"""

        def boom(**_kwargs: Any) -> str:
            raise ValueError("bad")

        tools = {SAFE: FakeToolDef(fn=boom, idempotent=True)}
        plan = plan_batch([_call("c1")], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.error_type == "3c"
        assert result.retryable is True

    def test_arguments_are_expanded_as_kwargs(self) -> None:
        """arguments 被解析为关键字参数传给 fn。"""
        seen: dict[str, Any] = {}

        def fn(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "ok"

        tools = {SAFE: FakeToolDef(fn=fn)}
        plan = plan_batch(
            [_call("c1", SAFE, '{"path": "a.txt", "offset": 3}')], lambda c: "safe"
        )
        execute_batch(plan, _resolver(tools))
        assert seen == {"path": "a.txt", "offset": 3}

    def test_invalid_arguments_json_becomes_3d(self) -> None:
        """非法 JSON 由执行器在校验关拦截，归 3d（模型可自我修正）。

        历史：曾经只做 JSON 解析不做 schema 校验，参数错误一律变成 3c
        （执行异常），模型拿到的是一段 traceback，无法据此改参数。
        """
        tools = {SAFE: FakeToolDef(fn=lambda **_: "ok")}
        plan = plan_batch([_call("c1", SAFE, "{bad}")], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is False
        assert result.error_type == "3d"
        assert result.retryable is False  # J4：3d 不自动重试

    def test_missing_required_field_becomes_3d(self) -> None:
        """缺必填字段必须被拦截，不能静默传给 fn。"""
        called: list[dict] = []

        def fn(**kwargs) -> str:  # noqa: ANN003 - 记录实际收到的参数
            called.append(kwargs)
            return "should-not-run"

        tools = {
            SAFE: FakeToolDef(
                fn=fn,
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        }
        plan = plan_batch([_call("c1", SAFE, '{"limit": 5}')], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is False
        assert result.error_type == "3d"
        assert called == [], "参数不合法时不应调用 fn"
        assert "missing" in result.content

    def test_wrong_type_becomes_3d(self) -> None:
        """类型不符必须被拦截。"""
        tools = {
            SAFE: FakeToolDef(
                fn=lambda **_: "ok",
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        }
        plan = plan_batch([_call("c1", SAFE, '{"path": 123}')], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is False
        assert result.error_type == "3d"
        assert "type" in result.content

    def test_coercible_literal_is_accepted(self) -> None:
        """字符串字面量 '"5"' 应被转换为 5，不算错误。"""
        seen: dict = {}

        def fn(**kwargs) -> str:  # noqa: ANN003
            seen.update(kwargs)
            return "ok"

        tools = {
            SAFE: FakeToolDef(
                fn=fn,
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["path"],
                },
            )
        }
        plan = plan_batch([_call("c1", SAFE, '{"path": "a", "limit": "5"}')], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is True
        assert seen["limit"] == 5

    def test_extra_fields_ignored(self) -> None:
        """多余字段忽略，不报错，也不传给 fn。"""
        seen: dict = {}

        def fn(**kwargs) -> str:  # noqa: ANN003
            seen.update(kwargs)
            return "ok"

        tools = {
            SAFE: FakeToolDef(
                fn=fn,
                schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        }
        plan = plan_batch([_call("c1", SAFE, '{"path": "a", "bogus": 1}')], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is True
        assert seen == {"path": "a"}

    def test_default_filled(self) -> None:
        """schema 有 default 时自动补齐。"""
        seen: dict = {}

        def fn(**kwargs) -> str:  # noqa: ANN003
            seen.update(kwargs)
            return "ok"

        tools = {
            SAFE: FakeToolDef(
                fn=fn,
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "default": 100},
                    },
                    "required": ["path"],
                },
            )
        }
        plan = plan_batch([_call("c1", SAFE, '{"path": "a"}')], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.ok is True
        assert seen["limit"] == 100

    def test_duration_ms_is_non_negative(self) -> None:
        tools = {SAFE: FakeToolDef(fn=lambda **_: "ok")}
        plan = plan_batch([_call("c1")], lambda c: "safe")
        result = execute_batch(plan, _resolver(tools))["c1"]
        assert result.duration_ms >= 0

    def test_results_keys_match_calls(self) -> None:
        """D2：结果键与 calls 一一对应。"""
        names = [f"/tool/os/file/f{i}" for i in range(5)]
        tools = {name: FakeToolDef(fn=lambda **_: "ok") for name in names}
        calls = [_call(f"c{i}", names[i]) for i in range(5)]
        mutable = {"c3": "exclusive"}
        plan = plan_batch(calls, lambda c: mutable.get(c.id, "safe"))

        results = execute_batch(plan, _resolver(tools))

        assert list(results) == [c.id for c in calls]

    def test_rejects_zero_workers(self) -> None:
        plan = plan_batch([_call("c1")], lambda c: "safe")
        with pytest.raises(ValueError):
            execute_batch(plan, _resolver({}), max_workers=0)


class TestSafetyPoint:
    def test_on_group_done_is_called_per_group(self) -> None:
        calls = [
            _call("c1", SAFE),
            _call("c2", EXCL),
            _call("c3", SAFE),
        ]
        tools = {
            SAFE: FakeToolDef(fn=lambda **_: "ok"),
            EXCL: FakeToolDef(fn=lambda **_: "ok", concurrency="exclusive"),
        }
        plan = plan_batch(calls, lambda c: tools[c.name].concurrency)
        seen: list[GroupReport] = []

        execute_batch(plan, _resolver(tools), on_group_done=seen.append)

        assert [r.index for r in seen] == [0, 1, 2]
        assert [r.is_last for r in seen] == [False, False, True]
        assert seen[0].results == {"c1": plan.results["c1"]}

    def test_committed_hook_stops_remaining_groups_and_fills_3e(self) -> None:
        """D7：组间中断 → 停止剩余组，并为全部未回填调用补 3e。"""
        calls = [
            _call("c1", SAFE),
            _call("c2", EXCL),
            _call("c3", SAFE),
        ]
        tools = {
            SAFE: FakeToolDef(fn=lambda **_: "ok"),
            EXCL: FakeToolDef(fn=lambda **_: "ok", concurrency="exclusive"),
        }
        plan = plan_batch(calls, lambda c: tools[c.name].concurrency)
        placeholders: list[str] = []

        def hook(report: GroupReport) -> Committed | None:
            if report.index != 0 or report.is_last:
                return None
            msgs = fill_placeholders(plan, set(report.plan.results), "user_abort")
            placeholders.extend(m.tool_call_id for m in msgs)
            return Committed(reason="user_abort")

        results = execute_batch(plan, _resolver(tools), on_group_done=hook)

        assert list(results) == ["c1"]
        assert placeholders == ["c2", "c3"]

    def test_no_hook_runs_all_groups(self) -> None:
        calls = [_call("c1", SAFE), _call("c2", EXCL)]
        tools = {
            SAFE: FakeToolDef(fn=lambda **_: "ok"),
            EXCL: FakeToolDef(fn=lambda **_: "ok", concurrency="exclusive"),
        }
        plan = plan_batch(calls, lambda c: tools[c.name].concurrency)
        results = execute_batch(plan, _resolver(tools))
        assert list(results) == ["c1", "c2"]


class TestPlanResultCoupling:
    def test_execute_returns_same_objects_as_plan_results(self) -> None:
        tools = {SAFE: FakeToolDef(fn=lambda **_: "ok")}
        plan = plan_batch([_call("c1")], lambda c: "safe")
        results = execute_batch(plan, _resolver(tools))
        assert results["c1"] is plan.results["c1"]
        assert isinstance(results["c1"], ExecResult)
        assert isinstance(plan, BatchPlan)
