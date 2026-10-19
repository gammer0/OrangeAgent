"""Execution-layer data models and batch grouping (group D).

See docs/02-data-structures/06-execution.md.

This module handles **data structures and grouping only** and executes no
tools:

* ``ExecResult``        result of one call (D4: ``content`` remains a string on failure)
* ``ExecGroup``         a sub-batch, the atomic unit that can be submitted at once
* ``BatchPlan``         all calls of this batch + grouping + result slots
* ``plan_batch``        grouping rules (D1/D5/D6)
* ``fill_placeholders`` backfill ``3e`` for **every unfilled** call in the batch
  when an interrupt arrives between groups (D7)

Constraint ownership:

* D1 -> ``mode`` derivation in ``plan_batch``
* D2 -> construction-time validation of ``BatchPlan``
* D5 -> concatenation self-check in ``plan_batch``
* D6 -> construction-time validation of ``ExecGroup``
* D7 -> coverage of ``fill_placeholders`` + ``retryable=False``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orange.message import ToolCall, ToolMsg

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
"""Error identifiers; see docs/01-architecture.md §4.9."""

ERROR_TYPES: tuple[str, ...] = ("1", "2", "3a", "3b", "3c", "3d", "3e")

Mode = Literal["serial", "parallel"]

ABORT_REASON_ALLOWED: tuple[str, ...] = ("user_abort", "session_end", "timeout")
"""Fixed set of ``InterruptRequest.reason`` values (G11)."""


def _error_payload(error_type: str, message: str) -> str:
    """Build ``content`` for ``ok=False``.

    D4 requires ``content`` to still be a valid string on failure; we serialize
    a JSON object so it is both a string and easy for the model to parse.
    """
    return json.dumps(
        {"error_type": error_type, "message": message}, ensure_ascii=False
    )


def error_content(error_type: str, message: str) -> str:
    """Public error ``content`` builder (reused by the executor and argument validation)."""
    return _error_payload(error_type, message)


def skipped_content(reason: str) -> str:
    """Build the ``3e`` placeholder ``content`` (fixed value per §6.3)."""
    return json.dumps(
        {"status": "skipped", "reason": reason}, ensure_ascii=False
    )


class ExecResult(BaseModel):
    """``ExecResult``: the result of one tool call.

    Attributes:
        ok: Whether the call succeeded.
        content: Serialized result; still a valid string when ``ok=False`` (D4).
        error_type: Error identifier; ``None`` on success.
        retryable: Determined **only** by ``ToolDef.idempotent`` (§4.13) and
            unrelated to ``concurrency``; always ``False`` for ``3e`` (D7).
        duration_ms: Execution duration in milliseconds.
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
            raise ValueError("D4: content must be a string")
        if self.ok and self.error_type is not None:
            raise ValueError("error_type must be None when ok=True")
        if not self.ok and self.error_type is None:
            raise ValueError("error_type is required when ok=False")
        if self.error_type == "3e" and self.retryable:  # D7
            raise ValueError("D7: retryable must be false when error_type == '3e'")
        return self


class ExecGroup(BaseModel):
    """``ExecGroup``: the atomic unit that can be submitted at once.

    Attributes:
        index: Position within this batch, starting at 0.
        calls: Calls in the group, ordered.
        mode: Execution mode within the group; ``"parallel"`` requires every
            element to be ``safe`` (D6).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: Annotated[int, Field(ge=0)]
    calls: Annotated[list[ToolCall], Field(min_length=1)]
    mode: Mode


class BatchPlan(BaseModel):
    """``BatchPlan``: the complete execution plan for one batch.

    Immutable after construction (§6.2: grouping is formed once and cannot
    change during execution); ``results`` is written item by item via
    ``record``, and write order never affects backfill order (D3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: list[ToolCall] = Field(default_factory=list)
    groups: list[ExecGroup] = Field(default_factory=list)
    mode: Mode = "parallel"
    results: dict[str, ExecResult] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_groups_cover_calls(self) -> BatchPlan:
        """D5: concatenating groups equals calls exactly (no omission, duplicate or reorder)."""
        flattened = [call.id for group in self.groups for call in group.calls]
        expected = [call.id for call in self.calls]
        if flattened != expected:
            raise ValueError(
                f"D5: concatenated groups do not match calls; "
                f"groups={flattened} calls={expected}"
            )
        for position, group in enumerate(self.groups):
            if group.index != position:
                raise ValueError(
                    f"group.index must match its position: "
                    f"position {position} has {group.index}"
                )
        return self

    # -- results is the only mutable part during execution (D2 / D3) -----------

    def record(self, call_id: str, result: ExecResult) -> None:
        """Record one result; the key must belong to this batch's ``calls`` (D2)."""
        if not any(call.id == call_id for call in self.calls):
            raise KeyError(f"D2: {call_id!r} does not belong to this batch")
        self.results[call_id] = result

    def is_complete(self) -> bool:
        """Whether every result has been backfilled (D2 completeness check)."""
        return len(self.results) == len(self.calls)

    def ordered_results(self) -> list[tuple[ToolCall, ExecResult]]:
        """Return (call, result) in group order then intra-group order (D3).

        Never in completion order.
        """
        ordered: list[tuple[ToolCall, ExecResult]] = []
        for group in self.groups:
            for call in group.calls:
                result = self.results.get(call.id)
                if result is not None:
                    ordered.append((call, result))
        return ordered


ConcurrencyOf = Callable[[ToolCall], str]
"""Injected mapper from ``ToolCall`` to ``"safe"`` / ``"exclusive"``."""


def plan_batch(calls: list[ToolCall], concurrency_of: ConcurrencyOf) -> BatchPlan:
    """Split one batch into a sequence of ``ExecGroup`` by concurrency (§6.2).

    Rules:

    * Adjacent ``safe`` calls merge into one ``parallel`` group (never spanning
      an ``exclusive``);
    * Each ``exclusive`` call occupies its own ``serial`` group;
    * If any ``exclusive`` appears, ``BatchPlan.mode = "serial"`` (D1);
    * All ``safe`` -> one ``parallel`` group; empty batch -> no groups,
      ``mode="parallel"``.

    Args:
        calls: Every call in this batch, in provider-return order.
        concurrency_of: Call -> concurrency attribute; anything other than
            ``"safe"`` is treated as ``exclusive`` (conservative default, §4.6).

    Returns:
        An immutable ``BatchPlan`` satisfying D5.
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
    """Backfill a ``3e`` placeholder for **every unfilled** call in the batch (D7).

    Note the coverage is "unfilled", not "unexecuted": a call that ran but has
    no result yet must be backfilled too, otherwise the next LLM round fails
    with 400 because ``tool_calls`` is left open. Output order follows
    ``groups`` order then intra-group order (D3).

    Args:
        plan: The execution plan of the current batch.
        filled_ids: Set of ``tool_call_id`` values that already have an ``ExecResult``.
        reason: ``InterruptRequest.reason``, written verbatim into
            ``content.reason``; must never be hardcoded.

    Returns:
        The placeholder ``ToolMsg`` list, in stable order.

    Raises:
        ValueError: ``reason`` is not one of the three values allowed by G11.
    """
    if reason not in ABORT_REASON_ALLOWED:
        raise ValueError(
            f"G11: reason must be one of {ABORT_REASON_ALLOWED}, got {reason!r}"
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
