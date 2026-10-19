"""Executor: run a ``BatchPlan`` and backfill results.

See docs/02-data-structures/06-execution.md and docs/01-architecture.md §4.6/§4.6.1.

Boundaries (things this module explicitly does not do):

* **Does not consume ``Inbox``**: safe-point checks are delegated to the
  orchestration layer via the ``on_group_done`` callback;
* **Does not retry or back off**: that belongs to the error layer (§4.9); this
  module only marks ``retryable``;
* **Does not touch ``tool_tree``**: tool definitions arrive through the
  ``resolver`` dependency injection.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from orange.args_validate import validate_arguments
from orange.execution import BatchPlan, ExecGroup, ExecResult, error_content
from orange.message import ToolCall
from orange.truncate import DEFAULT_MAX_CHARS, truncate_result

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
    """The subset of a tool definition needed to execute it.

    Kept minimal so this module does not hard-depend on the tool layer's
    ``ToolDef``.

    Attributes:
        fn: Tool implementation with signature ``fn(**kwargs) -> Any``.
        concurrency: ``"safe"`` / ``"exclusive"``; anything other than
            ``"safe"`` is treated as ``exclusive``.
        idempotent: Whether the tool is idempotent; the **only** input to
            ``ExecResult.retryable`` (§4.13).
        schema: Argument JSON Schema; absent or an empty object means
            "no constraints".
    """

    fn: Callable[..., Any]
    concurrency: str
    idempotent: bool
    schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GroupReport:
    """Information handed to the orchestration layer after a group completes (for safe points).

    Attributes:
        index: Group index.
        group: The group itself.
        results: Results produced by this group, keyed by ``tool_call_id``.
        is_last: Whether this is the last group of the batch (i.e. whether ② and
            ③ coincide).
        plan: The batch plan, so callbacks can judge overall progress.
    """

    index: int
    group: ExecGroup
    results: dict[str, ExecResult]
    is_last: bool
    plan: BatchPlan


@dataclass(frozen=True, slots=True)
class Committed:
    """The orchestration layer's safe-point verdict: placeholders are already filled.

    Returned when an interrupt has caused every unfilled call to receive a
    placeholder.

    Attributes:
        reason: The ``InterruptRequest.reason`` that triggered the interrupt,
            passed through verbatim.
    """

    reason: str


GroupDoneHook = Callable[[GroupReport], Committed | None]
"""Group-completion callback; returning ``Committed`` means this group is already
committed and the remaining groups must not run."""


def _content_of(value: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Convert a tool return value into the string needed for ``ToolMsg.content`` (D4).

    Note: C1 truncation is wired in here -- this is the single exit through
    which a tool result becomes a ``ToolMsg``, so placing it here covers both
    the success and failure paths.
    """
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    return truncate_result(value, max_chars=max_chars)


def _failed_content(exc: BaseException, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """``content`` for ``3c``: error type + message + traceback (still a valid string)."""
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
    """Execute one call: validate arguments first (gates ①-③), then invoke (gate ④).

    Validating **before** entering ``fn`` matters:

    * Argument errors map to ``3d`` (the model can self-correct) rather than the
      execution exception ``3c``;
    * ``3d`` content is structured JSON (``stage`` + ``errors[]``) so the model
      sees exactly which field is missing or mistyped; a ``3c`` traceback is
      noise to the model;
    * Missing required fields or type errors must never be silently passed to
      ``fn`` -- otherwise built-in defaults inside the tool could produce a
      result that looks successful but is semantically wrong.

    ``3d`` is never retried (J4); only exceptions raised inside ``fn`` map to
    ``3c``.
    """
    started = time.perf_counter()
    tool: ExecToolDef | None = None
    try:
        tool = resolver(call)

        # Gates ①-③: JSON parsing -> schema validation -> coercion and default filling
        validation, failure = validate_arguments(
            call.arguments, getattr(tool, "schema", {}) or {}
        )
        if failure is not None:
            elapsed = int((time.perf_counter() - started) * 1000)
            return failure.model_copy(update={"duration_ms": elapsed})
        assert validation is not None  # mutually exclusive with failure

        # Gate ④: invoke the implementation
        value = tool.fn(**validation.params)
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExecResult(
            ok=True,
            content=_content_of(value, max_chars),
            error_type=None,
            retryable=tool.idempotent,
            duration_ms=elapsed,
        )
    except BaseException as exc:  # noqa: BLE001 - must catch; else the batch is left half-done
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
    """Execute one group and return ``{tool_call_id: ExecResult}``.

    ``parallel`` groups use a thread pool; ``serial`` groups run in intra-group
    order. Per D6 an ``exclusive`` group always has a single element, but the
    serial branch tolerates multiple elements too.
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
    """Execute the whole batch in ``groups`` order, writing results back into ``plan.results``.

    The **insertion order** of the result dict is exactly "group order then
    intra-group order" (D3), independent of completion order: each group writes
    its results in intra-group ``calls`` order once it finishes.

    Args:
        plan: Batch plan (grouping is immutable; only ``results`` is filled).
        resolver: ``ToolCall -> ExecToolDef`` dependency injection.
        on_group_done: Group-completion callback (safe point). Returning
            :class:`Committed` stops the remaining groups immediately; this
            module **does not fill placeholders itself** -- the orchestration
            layer calls :func:`orange.execution.fill_placeholders` inside the
            callback.
        max_workers: Worker cap for parallel groups, default 4.
        max_chars: Per-result character cap (C1), default 4000.

    Returns:
        ``{tool_call_id: ExecResult}`` matching ``plan.calls`` one-to-one (D2).

    Raises:
        ValueError: ``max_workers < 1``.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    total = len(plan.groups)

    for position, group in enumerate(plan.groups):
        group_results = _run_group(group, resolver, max_workers, max_chars)
        # D3: write in intra-group calls order, not completion order
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

    # D2: normally results matches calls one-to-one; on interrupt the
    # orchestration layer closes the gap by filling 3e placeholders.
    return dict(plan.results)
