"""执行器（Executor）：按 ``BatchPlan`` 执行并回填结果。

对应文档：docs/02-数据结构设计/06-执行.md、docs/01-系统架构设计.md §4.6/§4.6.1

边界（明确不做的事）：

* **不消费 ``Inbox``**：安全点检查由 ``on_group_done`` 回调交给编排层完成；
* **不实现重试 / 退避**：属于异常层（§4.9），本模块只标注 ``retryable``；
* **不访问 ``tool_tree``**：工具定义通过 ``resolver`` 依赖注入。
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from harness.args_validate import parse_call_arguments
from harness.execution import BatchPlan, ExecGroup, ExecResult, error_content
from harness.message import ToolCall
from harness.truncate import DEFAULT_MAX_CHARS, truncate_result

__all__ = [
    "DEFAULT_MAX_WORKERS",
    "Committed",
    "ExecToolDef",
    "GroupDoneHook",
    "GroupReport",
    "execute_batch",
]

DEFAULT_MAX_WORKERS = 4


@runtime_checkable
class ExecToolDef(Protocol):
    """执行所需的工具定义子集（避免本模块硬依赖 tool 层的 ``ToolDef``）。

    Attributes:
        fn: 工具实现，签名 ``fn(**kwargs) -> Any``。
        concurrency: ``"safe"`` / ``"exclusive"``；不是 ``"safe"`` 一律按 ``exclusive``。
        idempotent: 是否幂等；**唯一**决定 ``ExecResult.retryable``（§4.13）。
    """

    fn: Callable[..., Any]
    concurrency: str
    idempotent: bool


@dataclass(frozen=True, slots=True)
class GroupReport:
    """一次组完成后传给编排层的信息（安全点检查用）。

    Attributes:
        index: 组序号。
        group: 该组本身。
        results: 该组已产出的结果，键为 ``tool_call_id``。
        is_last: 是否为本批最后一组（② 与 ③ 是否重合）。
        plan: 批次计划，便于回调判断整体进度。
    """

    index: int
    group: ExecGroup
    results: dict[str, ExecResult]
    is_last: bool
    plan: BatchPlan


@dataclass(frozen=True, slots=True)
class Committed:
    """编排层的安全点结论：因中断已为全部未回填调用补完占位。

    Attributes:
        reason: 触发中断的 ``InterruptRequest.reason``，原样透传。
    """

    reason: str


GroupDoneHook = Callable[[GroupReport], Committed | None]
"""组完成回调；返回 ``Committed`` 表示本组已提交且剩余组应停止执行。"""


def _content_of(value: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """把工具返回值转成 ``ToolMsg.content`` 所需的字符串（D4）。

    ponytail: C1 截断在此接入——它是工具结果进入 ``ToolMsg`` 的唯一出口，
    放这里才能同时覆盖成功与失败两条路径。
    """
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    return truncate_result(value, max_chars=max_chars)


def _failed_content(exc: BaseException, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """``3c`` 的 ``content``：错误类型 + 消息 + traceback（仍是合法字符串）。"""
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return truncate_result(
        error_content("3c", f"{type(exc).__name__}: {exc}\n{detail}"),
        max_chars=max_chars,
    )


def _run_one(
    call: ToolCall,
    resolver: Callable[[ToolCall], ExecToolDef],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ExecResult:
    """执行单个调用；任何异常都收敛为 ``3c``（§4.6 Executor 职责）。"""
    started = time.perf_counter()
    tool: ExecToolDef | None = None
    try:
        tool = resolver(call)
        value = tool.fn(**parse_call_arguments(call))
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExecResult(
            ok=True,
            content=_content_of(value, max_chars),
            error_type=None,
            retryable=tool.idempotent,
            duration_ms=elapsed,
        )
    except BaseException as exc:  # noqa: BLE001 - 工具异常必须被捕获，否则批次留下半截
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExecResult(
            ok=False,
            content=_failed_content(exc, max_chars),
            error_type="3c",
            retryable=bool(getattr(tool, "idempotent", False)),
            duration_ms=elapsed,
        )


def _run_group(
    group: ExecGroup,
    resolver: Callable[[ToolCall], ExecToolDef],
    max_workers: int,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, ExecResult]:
    """执行一个组，返回 ``{tool_call_id: ExecResult}``。

    ``parallel`` 组用线程池并发；``serial`` 组按组内顺序逐个执行。
    ``exclusive`` 组按 D6 恒为单元素，串行分支同样兼容多元素。
    """
    if group.mode == "parallel" and len(group.calls) > 1:
        workers = max(1, min(max_workers, len(group.calls)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                (call, pool.submit(_run_one, call, resolver, max_chars))
                for call in group.calls
            ]
            return {call.id: future.result() for call, future in futures}

    return {call.id: _run_one(call, resolver, max_chars) for call in group.calls}


def execute_batch(
    plan: BatchPlan,
    resolver: Callable[[ToolCall], ExecToolDef],
    on_group_done: GroupDoneHook | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, ExecResult]:
    """按 ``groups`` 顺序执行整批，并把结果写回 ``plan.results``。

    结果字典的**插入顺序**严格等于"组序 + 组内序"（D3），与并发完成顺序无关：
    每个组执行完毕后按组内 ``calls`` 顺序写入。

    Args:
        plan: 批次计划（分组不可变，仅 ``results`` 被填充）。
        resolver: ``ToolCall -> ExecToolDef`` 依赖注入。
        on_group_done: 组完成回调（安全点）。返回 :class:`Committed` 时立即停止
            后续组；本模块**不自行补占位**，由编排层在回调中调用
            :func:`harness.execution.fill_placeholders`。
        max_workers: 并行组的线程数上限，默认 4。
        max_chars: 单条工具结果的字符上限（C1），默认 4000。

    Returns:
        与 ``plan.calls`` 一一对应的 ``{tool_call_id: ExecResult}``（D2）。

    Raises:
        ValueError: ``max_workers < 1``。
    """
    if max_workers < 1:
        raise ValueError("max_workers 必须 >= 1")

    total = len(plan.groups)

    for position, group in enumerate(plan.groups):
        group_results = _run_group(group, resolver, max_workers, max_chars)
        # D3：按组内 calls 顺序写入，不按完成顺序
        for call in group.calls:
            plan.record(call.id, group_results[call.id])

        if on_group_done is None:
            continue

        report = GroupReport(
            index=group.index,
            group=group,
            results=group_results,
            is_last=position == total - 1,
            plan=plan,
        )
        if on_group_done(report) is not None:
            break

    # D2：正常情况下 results 键与 calls 一一对应；中断时由编排层补 3e 后闭合。
    return dict(plan.results)
