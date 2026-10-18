"""Jev routing: tool-branch selection with low-confidence fallback (§4.4).

See:

* docs/01-architecture.md §4.4 (enable conditions, Jev spec, fall back on failure)
* docs/02-data-structures/08-runtime-state.md (``RouteDecision`` and F2)

Design notes:

* **Routing is an optimization, not the critical path** (I4): any exception, low
  confidence or timeout must degrade losslessly to "expose everything" and must
  never block the main flow.
* **``state`` carries the query only, never tool output** -- this prevents
  historical tool results from skewing the routing decision.
* **The main model performs routing on our behalf**: no external Jev service.
  We send one tiny selection request over the same model channel, list the
  candidate branch numbers in the prompt, and ask the model to reply with a
  single number. When the main model is unavailable we fall back immediately
  instead of retrying.

Enable conditions (design doc §4.4): ``tool_count >= 30`` and
``branch_count >= 6`` and ``history_tokens > 4000``. If any condition fails we
skip routing -- searching a tree of few tools is pure inference loss.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from harness.message import AssistantMsg, Message, UserMsg
from harness.routing import RouteDecision

__all__ = [
    "ROUTE_ENABLE_TOOL_COUNT",
    "ROUTE_ENABLE_BRANCH_COUNT",
    "ROUTE_ENABLE_HISTORY_TOKENS",
    "CONFIDENCE_THRESHOLD",
    "JevRouter",
    "should_route",
    "build_route_prompt",
    "parse_route_reply",
]

#: Config item ``JEV_ROUTE_ENABLE_THRESHOLD``.
ROUTE_ENABLE_TOOL_COUNT = 30
#: Minimum branch count (design doc §4.4).
ROUTE_ENABLE_BRANCH_COUNT = 6
#: Minimum history tokens (design doc §4.4).
ROUTE_ENABLE_HISTORY_TOKENS = 4000
#: Fall back to the full tree below this confidence (design doc §4.4).
CONFIDENCE_THRESHOLD = 0.7

#: Extract a number from the model reply; tolerates ``B3`` / ``3`` / ``branch 3``.
_BRANCH_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])(?:B)?(\d{1,3})(?![0-9])")


def should_route(
    *,
    tool_count: int,
    branch_count: int,
    history_tokens: int,
) -> bool:
    """Whether the routing enable conditions hold (all three thresholds met)."""
    return (
        tool_count >= ROUTE_ENABLE_TOOL_COUNT
        and branch_count >= ROUTE_ENABLE_BRANCH_COUNT
        and history_tokens > ROUTE_ENABLE_HISTORY_TOKENS
    )


def build_route_prompt(query: str, branches: Sequence[str]) -> str:
    """Build the routing-selection prompt.

    Carries **only the query and candidate branch names**, never any tool output
    (guards against JPDH).

    Args:
        query: The user's request text for this turn.
        branches: Candidate top-level branch paths.

    Returns:
        Prompt text usable directly as a single ``UserMsg``.

    Note:
        The prompt text is model-facing and intentionally stays in Chinese.
    """
    listing = "\n".join(f"{index}. {path}" for index, path in enumerate(branches, start=1))
    return (
        "你是一个工具路由分类器。请从下列候选分支中选出与用户请求最相关的一个。\n"
        "只输出编号（一个阿拉伯数字），不要解释，不要输出其它任何内容。\n\n"
        f"候选分支：\n{listing}\n\n"
        f"用户请求：{query}"
    )


def parse_route_reply(reply: str, branches: Sequence[str]) -> RouteDecision:
    """Parse a model reply into a ``RouteDecision``.

    Parse failures and out-of-range numbers always yield the fallback decision
    (F2: ``fallback=True`` when ``branch is None``).

    Args:
        reply: Raw reply text from the model.
        branches: Candidate branches, aligned with the prompt numbering (1-based).

    Returns:
        A selection with ``confidence=1.0`` for a valid number; otherwise the
        fallback decision.
    """
    match = _BRANCH_PATTERN.search(reply.strip())
    if match is None:
        return RouteDecision.full(confidence=0.0)

    index = int(match.group(1))
    if not 1 <= index <= len(branches):
        return RouteDecision.full(confidence=0.0)

    return RouteDecision(branch=branches[index - 1], confidence=1.0, fallback=False)


@dataclass
class JevRouter:
    """A router that delegates the decision to the main model.

    Attributes:
        chat: Injected single-turn model call with signature
            ``(messages: list[Message]) -> AssistantMsg``.
        confidence_threshold: Fall back to the full tree below this confidence.
    """

    chat: Callable[[list[Message]], AssistantMsg]
    confidence_threshold: float = CONFIDENCE_THRESHOLD

    def route(
        self,
        query: str,
        branches: Sequence[str],
        *,
        history_tokens: int = ROUTE_ENABLE_HISTORY_TOKENS,
        tool_count: int = ROUTE_ENABLE_TOOL_COUNT,
    ) -> RouteDecision:
        """Select a tool branch; any failure falls back to the full tree (I4).

        Args:
            query: The user's request text.
            branches: Candidate top-level branch paths.
            history_tokens: Current history token count (for the enable check).
            tool_count: Current total tool count (for the enable check).

        Returns:
            A :class:`RouteDecision`; when the enable conditions fail or the call
            fails, ``branch is None`` and ``fallback is True``.
        """
        if not branches:
            return RouteDecision.full(confidence=0.0)

        if not should_route(
            tool_count=tool_count,
            branch_count=len(branches),
            history_tokens=history_tokens,
        ):
            # Enable conditions not met: skip routing, equivalent to full exposure
            return RouteDecision.full(confidence=0.0)

        prompt = build_route_prompt(query, branches)
        try:
            reply = self.chat([UserMsg(content=prompt)])
        except Exception:
            # A routing failure must not block the main flow (I4)
            return RouteDecision.full(confidence=0.0)

        decision = parse_route_reply(reply.content or "", branches)
        if decision.confidence < self.confidence_threshold:
            return RouteDecision.full(confidence=decision.confidence)
        return decision

    def visible_branches(
        self,
        decision: RouteDecision,
        branches: Sequence[str],
    ) -> list[str]:
        """Filter branches by the decision; a fallback returns everything (F2 consumer).

        Args:
            decision: Routing result.
            branches: All candidate branches.

        Returns:
            The branch list to expose to the model.
        """
        if decision.fallback or decision.branch is None:
            return list(branches)
        return [decision.branch]


def parse_route_json(text: str) -> RouteDecision | None:
    """Try to parse the reply as ``{"branch": ..., "confidence": ...}``.

    A tolerance path only: the model occasionally returns JSON instead of a
    bare number.

    Args:
        text: Model reply.

    Returns:
        A decision on success, otherwise ``None``.
    """
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    branch = payload.get("branch")
    confidence = payload.get("confidence", 0.0)
    if not isinstance(branch, str) or not isinstance(confidence, (int, float)):
        return None
    confidence = max(0.0, min(1.0, float(confidence)))
    if branch is None:
        return RouteDecision.full(confidence=confidence)
    return RouteDecision(branch=branch, confidence=confidence, fallback=False)
