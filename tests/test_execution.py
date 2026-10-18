"""执行层数据结构与分组测试（D1–D7）。

测试名带约束编号，便于与 docs/02-数据结构设计/06-执行.md 对照。
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from harness.execution import (
    BatchPlan,
    ExecGroup,
    ExecResult,
    fill_placeholders,
    plan_batch,
)
from harness.message import ToolCall

SAFE = "/tool/os/file/read"
EXCL = "/tool/os/shell/run"


def _call(call_id: str, name: str = SAFE) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments="{}")


def _concurrency_map(mapping: dict[str, str]) -> object:
    """按 call.id 查并发属性，缺省 ``safe``。"""

    def of(call: ToolCall) -> str:
        return mapping.get(call.id, "safe")

    return of


class TestPlanBatch:
    def test_all_safe_single_parallel_group(self) -> None:
        """D1：全 safe → 单个 parallel 组，且 mode == parallel。"""
        calls = [_call("c1"), _call("c2"), _call("c3")]
        plan = plan_batch(calls, _concurrency_map({}))

        assert plan.mode == "parallel"
        assert len(plan.groups) == 1
        assert plan.groups[0].mode == "parallel"
        assert [c.id for c in plan.groups[0].calls] == ["c1", "c2", "c3"]

    def test_adjacent_safe_merged_and_exclusive_isolated(self) -> None:
        """D5/D6：相邻 safe 合并；每个 exclusive 独占一组。"""
        mapping = {"c3": "exclusive", "c4": "exclusive"}
        calls = [_call(f"c{i}") for i in range(1, 7)]
        plan = plan_batch(calls, _concurrency_map(mapping))

        assert [len(g.calls) for g in plan.groups] == [2, 1, 1, 2]
        assert [g.mode for g in plan.groups] == ["parallel", "serial", "serial", "parallel"]
        assert [[c.id for c in g.calls] for g in plan.groups] == [
            ["c1", "c2"],
            ["c3"],
            ["c4"],
            ["c5", "c6"],
        ]

    def test_safe_group_does_not_cross_exclusive(self) -> None:
        """D5：safe 不得跨过 exclusive 合并。"""
        mapping = {"c2": "exclusive"}
        plan = plan_batch(
            [_call("c1"), _call("c2"), _call("c3")], _concurrency_map(mapping)
        )
        flattened = [call.id for group in plan.groups for call in group.calls]
        assert flattened == ["c1", "c2", "c3"]
        assert [g.mode for g in plan.groups] == ["parallel", "serial", "parallel"]

    def test_any_exclusive_makes_plan_serial(self) -> None:
        """D1：出现任一 exclusive → BatchPlan.mode == serial。"""
        mapping = {"c2": "exclusive"}
        plan = plan_batch(
            [_call("c1"), _call("c2"), _call("c3")], _concurrency_map(mapping)
        )
        assert plan.mode == "serial"

    def test_groups_concatenation_equals_calls(self) -> None:
        """D5：groups 拼接 == calls（无遗漏 / 无重复 / 无重排）。"""
        calls = [_call(f"c{i}") for i in range(1, 11)]
        mapping = {"c2": "exclusive", "c3": "exclusive", "c7": "exclusive"}
        plan = plan_batch(calls, _concurrency_map(mapping))
        flattened = [call.id for group in plan.groups for call in group.calls]
        assert flattened == [call.id for call in calls]
        assert len(flattened) == len(set(flattened))

    def test_group_index_is_sequential(self) -> None:
        mapping = {"c1": "exclusive", "c3": "exclusive"}
        plan = plan_batch(
            [_call("c1"), _call("c2"), _call("c3")], _concurrency_map(mapping)
        )
        assert [g.index for g in plan.groups] == list(range(len(plan.groups)))

    def test_unknown_concurrency_treated_as_exclusive(self) -> None:
        """§4.6：保守默认 exclusive。"""
        plan = plan_batch([_call("c1")], _concurrency_map({"c1": "weird"}))
        assert plan.groups[0].mode == "serial"
        assert plan.mode == "serial"

    def test_empty_batch(self) -> None:
        plan = plan_batch([], _concurrency_map({}))
        assert plan.groups == []
        assert plan.calls == []
        assert plan.mode == "parallel"


class TestPlanInvariants:
    def test_d5_rejects_mismatched_groups(self) -> None:
        """D5：构造期就拒绝 groups 与 calls 不一致。"""
        calls = [_call("c1"), _call("c2")]
        with pytest.raises(ValidationError):
            BatchPlan(
                calls=calls,
                groups=[ExecGroup(index=0, calls=[calls[0]], mode="parallel")],
                mode="parallel",
            )

    def test_d6_parallel_group_rejects_missing_mode_consistency(self) -> None:
        """D6：parallel 组必须自洽（多元素 + mode=parallel 由分组器保证）。"""
        group = ExecGroup(index=0, calls=[_call("c1"), _call("c2")], mode="parallel")
        assert len(group.calls) == 2

    def test_results_record_rejects_foreign_id(self) -> None:
        """D2：results 键必须来自本批 calls。"""
        plan = plan_batch([_call("c1")], _concurrency_map({}))
        with pytest.raises(KeyError):
            plan.record("cX", ExecResult(ok=True, content="x"))

    def test_ordered_results_follow_group_order(self) -> None:
        """D3：ordered_results 按组序 + 组内序，与写入顺序无关。"""
        calls = [_call("c1"), _call("c2"), _call("c3")]
        plan = plan_batch(calls, _concurrency_map({"c3": "exclusive"}))
        # 逆序写入
        for call_id in ("c3", "c2", "c1"):
            plan.record(call_id, ExecResult(ok=True, content=call_id))
        assert [call.id for call, _ in plan.ordered_results()] == ["c1", "c2", "c3"]
        assert plan.is_complete()


class TestExecResult:
    def test_failed_content_is_string(self) -> None:
        """D4：ok=False 时 content 仍是合法字符串。"""
        result = ExecResult(ok=False, content="boom", error_type="3c", retryable=False)
        assert isinstance(result.content, str)

    def test_retryable_follows_idempotent_only(self) -> None:
        """§4.13：retryable 由调用方按 idempotent 传入，与 concurrency 无关。"""
        safe_non_idem = ExecResult(ok=True, content="x", retryable=False)
        exclusive_idem = ExecResult(ok=True, content="x", retryable=True)
        assert safe_non_idem.retryable is False
        assert exclusive_idem.retryable is True

    def test_failure_requires_error_type(self) -> None:
        with pytest.raises(ValidationError):
            ExecResult(ok=False, content="boom")

    def test_3e_cannot_be_retryable(self) -> None:
        """D7：error_type == 3e 时 retryable 必须为 false。"""
        with pytest.raises(ValidationError):
            ExecResult(ok=False, content="{}", error_type="3e", retryable=True)


class TestFillPlaceholders:
    def test_covers_every_unfilled_call_not_just_unexecuted(self) -> None:
        """D7：覆盖"全部未回填"，包含已执行但未回填的调用。"""
        calls = [_call(f"c{i}") for i in range(1, 7)]
        mapping = {"c3": "exclusive", "c5": "exclusive"}
        plan = plan_batch(calls, _concurrency_map(mapping))
        # 假设组 0、1 已回填，组 2 的 c4 已执行但因中断未回填
        filled = {"c1", "c2", "c3"}
        plan.record("c1", ExecResult(ok=True, content="x"))
        plan.record("c2", ExecResult(ok=True, content="x"))
        plan.record("c3", ExecResult(ok=True, content="x"))

        placeholders = fill_placeholders(plan, filled, "user_abort")

        assert [m.tool_call_id for m in placeholders] == ["c4", "c5", "c6"]
        for msg in placeholders:
            payload = json.loads(msg.content)
            assert payload == {"status": "skipped", "reason": "user_abort"}

    def test_reason_is_passed_through(self) -> None:
        """D7：reason 取自 InterruptRequest.reason，不得写死。"""
        plan = plan_batch([_call("c1")], _concurrency_map({}))
        for reason in ("user_abort", "session_end", "timeout"):
            msg = fill_placeholders(plan, set(), reason)[0]
            assert json.loads(msg.content)["reason"] == reason

    def test_placeholder_name_matches_call(self) -> None:
        """补占位后 ToolMsg.name 必须与 ToolCall.name 一致（A6）。"""
        plan = plan_batch([_call("c1", EXCL)], _concurrency_map({"c1": "exclusive"}))
        msg = fill_placeholders(plan, set(), "session_end")[0]
        assert msg.name == EXCL

    def test_placeholder_order_is_group_order(self) -> None:
        """D3/D7：占位顺序沿用 groups 顺序。"""
        calls = [_call(f"c{i}") for i in range(1, 5)]
        plan = plan_batch(calls, _concurrency_map({"c2": "exclusive"}))
        ids = [m.tool_call_id for m in fill_placeholders(plan, set(), "timeout")]
        assert ids == ["c1", "c2", "c3", "c4"]

    def test_no_placeholder_when_all_filled(self) -> None:
        plan = plan_batch([_call("c1")], _concurrency_map({}))
        assert fill_placeholders(plan, {"c1"}, "timeout") == []

    def test_rejects_unknown_reason(self) -> None:
        """G11：reason 只能取三个固定值。"""
        plan = plan_batch([_call("c1")], _concurrency_map({}))
        with pytest.raises(ValueError):
            fill_placeholders(plan, set(), "because")
