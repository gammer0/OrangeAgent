"""执行层数据模型与批次分组（D 组）。

对应文档：docs/02-数据结构设计/06-执行.md

本模块只负责**数据结构与分组**，不执行任何工具：

* ``ExecResult``  单次调用结果（D4：失败时 ``content`` 仍为字符串）
* ``ExecGroup``   子批次，一次可提交的原子单位
* ``BatchPlan``   本批全部调用 + 分组 + 结果槽
* ``plan_batch``  分组规则（D1/D5/D6）
* ``fill_placeholders`` 组间中断时为本批**全部未回填**调用补 ``3e``（D7）

约束归属：

* D1 → ``plan_batch`` 的 ``mode`` 推导
* D2 → ``BatchPlan`` 构造期校验
* D5 → ``plan_batch`` 的拼接自检
* D6 → ``ExecGroup`` 构造期校验
* D7 → ``fill_placeholders`` 覆盖范围 + ``retryable=False``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.message import ToolCall, ToolMsg

__all__ = [
    "ERROR_TYPES",
    "ABORT_REASON_ALLOWED",
    "ExecResult",
    "ExecGroup",
    "BatchPlan",
    "plan_batch",
    "fill_placeholders",
    "skipped_content",
    "error_content",
]

ErrorType = Literal["1", "2", "3a", "3b", "3c", "3d", "3e"]
"""异常编号，见 docs/01-系统架构设计.md §4.9。"""

ERROR_TYPES: tuple[str, ...] = ("1", "2", "3a", "3b", "3c", "3d", "3e")

Mode = Literal["serial", "parallel"]

ABORT_REASON_ALLOWED: tuple[str, ...] = ("user_abort", "session_end", "timeout")
"""``InterruptRequest.reason`` 的固定取值（G11）。"""


def _error_payload(error_type: str, message: str) -> str:
    """构造 ``ok=False`` 时的 ``content``。

    D4 要求失败时 ``content`` 仍是合法字符串；这里统一序列化为 JSON 对象，
    既保证是字符串，也便于模型解析。
    """
    return json.dumps(
        {"error_type": error_type, "message": message}, ensure_ascii=False
    )


def error_content(error_type: str, message: str) -> str:
    """公开的错误 ``content`` 构造器（供执行器 / 参数校验复用）。"""
    return _error_payload(error_type, message)


def skipped_content(reason: str) -> str:
    """构造 ``3e`` 占位 ``content``（§6.3 固定值）。"""
    return json.dumps(
        {"status": "skipped", "reason": reason}, ensure_ascii=False
    )


class ExecResult(BaseModel):
    """``ExecResult``：单次工具调用的结果。

    Attributes:
        ok: 是否成功。
        content: 已序列化结果；``ok=False`` 时仍是合法字符串（D4）。
        error_type: 异常编号；成功时为 ``None``。
        retryable: **只由** ``ToolDef.idempotent`` 决定（§4.13），与 ``concurrency`` 无关；
            ``3e`` 恒为 ``False``（D7）。
        duration_ms: 执行耗时，毫秒。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    content: str
    error_type: ErrorType | None = None
    retryable: bool = False
    duration_ms: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _check_consistency(self) -> ExecResult:
        if not isinstance(self.content, str):  # D4
            raise ValueError("D4: content 必须是字符串")
        if self.ok and self.error_type is not None:
            raise ValueError("ok=True 时 error_type 必须为 None")
        if not self.ok and self.error_type is None:
            raise ValueError("ok=False 时必须给出 error_type")
        if self.error_type == "3e" and self.retryable:  # D7
            raise ValueError("D7: error_type == '3e' 时 retryable 必须为 false")
        return self


class ExecGroup(BaseModel):
    """``ExecGroup``：一次可提交的原子单位。

    Attributes:
        index: 在本批中的序号，从 0 起。
        calls: 组内调用，有序。
        mode: 组内执行模式；``"parallel"`` 时元素必须全部为 ``safe``（D6）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: Annotated[int, Field(ge=0)]
    calls: Annotated[list[ToolCall], Field(min_length=1)]
    mode: Mode


class BatchPlan(BaseModel):
    """``BatchPlan``：一个批次的完整执行计划。

    构造后不可变（§6.2：分组一次成型，执行期不可变）；``results`` 由执行器
    通过 ``record`` 逐项写入，写入顺序不影响回填顺序（D3）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: list[ToolCall] = Field(default_factory=list)
    groups: list[ExecGroup] = Field(default_factory=list)
    mode: Mode = "parallel"
    results: dict[str, ExecResult] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_groups_cover_calls(self) -> BatchPlan:
        """D5：groups 拼接后与 calls 完全一致（无遗漏、无重复、无重排）。"""
        flattened = [call.id for group in self.groups for call in group.calls]
        expected = [call.id for call in self.calls]
        if flattened != expected:
            raise ValueError(
                f"D5: groups 拼接与 calls 不一致；groups={flattened} calls={expected}"
            )
        for position, group in enumerate(self.groups):
            if group.index != position:
                raise ValueError(
                    f"group.index 必须与位置一致：位置 {position} 实际 {group.index}"
                )
        return self

    # -- results 是执行期唯一可变的部分（D2 / D3） -----------------------------

    def record(self, call_id: str, result: ExecResult) -> None:
        """写入一条结果；键必须属于本批 ``calls``（D2）。"""
        if not any(call.id == call_id for call in self.calls):
            raise KeyError(f"D2: {call_id!r} 不属于本批调用")
        self.results[call_id] = result

    def is_complete(self) -> bool:
        """本批是否已全部回填（D2 的完整性判定）。"""
        return len(self.results) == len(self.calls)

    def ordered_results(self) -> list[tuple[ToolCall, ExecResult]]:
        """按组序 + 组内序返回 (call, result)，不按完成顺序（D3）。"""
        ordered: list[tuple[ToolCall, ExecResult]] = []
        for group in self.groups:
            for call in group.calls:
                result = self.results.get(call.id)
                if result is not None:
                    ordered.append((call, result))
        return ordered


ConcurrencyOf = Callable[[ToolCall], str]
"""依赖注入：把 ``ToolCall`` 映射到 ``"safe"`` / ``"exclusive"``。"""


def plan_batch(calls: list[ToolCall], concurrency_of: ConcurrencyOf) -> BatchPlan:
    """按并发属性把一个批次切成 ``ExecGroup`` 序列（§6.2）。

    规则：

    * 相邻 ``safe`` 合并为一组 ``parallel``（不跨 ``exclusive``）；
    * 每个 ``exclusive`` 独占一组 ``serial``；
    * 只要出现任一 ``exclusive``，``BatchPlan.mode = "serial"``（D1）；
    * 全 ``safe`` → 单个 ``parallel`` 组；空批次 → 无组、``mode="parallel"``。

    Args:
        calls: 本批全部调用，顺序即 provider 返回顺序。
        concurrency_of: 调用 → 并发属性；返回值不是 ``"safe"`` 时按 ``exclusive`` 处理
            （保守默认，§4.6）。

    Returns:
        不可变的 ``BatchPlan``，满足 D5。
    """
    groups: list[ExecGroup] = []
    pending_safe: list[ToolCall] = []

    def _flush_safe() -> None:
        if not pending_safe:
            return
        groups.append(
            ExecGroup(index=len(groups), calls=list(pending_safe), mode="parallel")
        )
        pending_safe.clear()

    for call in calls:
        if concurrency_of(call) == "safe":
            pending_safe.append(call)
        else:
            _flush_safe()
            groups.append(
                ExecGroup(index=len(groups), calls=[call], mode="serial")
            )

    _flush_safe()

    plan_mode: Mode = "parallel" if all(group.mode == "parallel" for group in groups) else "serial"
    return BatchPlan(calls=list(calls), groups=groups, mode=plan_mode)


def fill_placeholders(
    plan: BatchPlan,
    filled_ids: set[str] | frozenset[str],
    reason: str,
) -> list[ToolMsg]:
    """为**本批全部未回填**调用补 ``3e`` 占位（D7）。

    注意覆盖范围是"未回填"而非"未执行"：已执行但结果尚未回填的调用同样要补，
    否则下一轮 LLM 会因 ``tool_calls`` 未闭合而 400。返回顺序沿用
    ``groups`` 顺序 + 组内顺序（D3）。

    Args:
        plan: 当前批次的执行计划。
        filled_ids: 已经有 ``ExecResult`` 的 ``tool_call_id`` 集合。
        reason: ``InterruptRequest.reason``，原样写入 ``content.reason``，不得写死。

    Returns:
        占位 ``ToolMsg`` 列表，顺序稳定。

    Raises:
        ValueError: ``reason`` 不在 G11 允许的三个取值内。
    """
    if reason not in ABORT_REASON_ALLOWED:
        raise ValueError(
            f"G11: reason 只能取 {ABORT_REASON_ALLOWED}，实际 {reason!r}"
        )

    content = skipped_content(reason)
    placeholders: list[ToolMsg] = []
    for group in plan.groups:
        for call in group.calls:
            if call.id in filled_ids:
                continue
            placeholders.append(
                ToolMsg(tool_call_id=call.id, name=call.name, content=content)
            )
    return placeholders
