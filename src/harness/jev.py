"""Jev 路由：工具分支选择 + 低置信回退（§4.4）。

对应文档：

* docs/01-系统架构设计.md §4.4（启用条件、Jev 规格、失败即回退）
* docs/02-数据结构设计/08-运行时状态.md（``RouteDecision`` 与 F2）

设计要点：

* **路由是优化，不是关键路径**（I4）：任何异常、低置信、超时都必须无损退化为
  "全量暴露"，绝不阻断主流程。
* **``state`` 只放 query，不放工具输出**——防止历史工具结果干扰路由判定。
* **由主模型代为执行路由**：不引入外部 Jev 服务。用同一条模型通道发一个
  极小的选择请求，把候选分支编号列进 prompt，要求模型只回一个编号。
  主模型不可用时直接回退，不做二次尝试。

启用条件（文档 §4.4）：``tool_count >= 30`` 且 ``branch_count >= 6`` 且
``history_tokens > 4000``。三个条件任一不满足就不路由——少量工具查树纯亏推理。
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

#: 文档配置项 ``JEV_ROUTE_ENABLE_THRESHOLD``。
ROUTE_ENABLE_TOOL_COUNT = 30
#: 分支数下限（文档 §4.4）。
ROUTE_ENABLE_BRANCH_COUNT = 6
#: 历史 token 下限（文档 §4.4）。
ROUTE_ENABLE_HISTORY_TOKENS = 4000
#: 置信度低于该值即回退全量（文档 §4.4）。
CONFIDENCE_THRESHOLD = 0.7

#: 模型回复中提取编号：容忍 ``B3`` / ``3`` / ``branch 3`` 等写法。
_BRANCH_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9])(?:B)?(\d{1,3})(?![0-9])")


def should_route(
    *,
    tool_count: int,
    branch_count: int,
    history_tokens: int,
) -> bool:
    """是否满足路由启用条件（三个阈值同时满足）。"""
    return (
        tool_count >= ROUTE_ENABLE_TOOL_COUNT
        and branch_count >= ROUTE_ENABLE_BRANCH_COUNT
        and history_tokens > ROUTE_ENABLE_HISTORY_TOKENS
    )


def build_route_prompt(query: str, branches: Sequence[str]) -> str:
    """构造路由选择提示。

    **只放 query 与候选分支名**，不放任何工具输出（防 JPDH）。

    Args:
        query: 用户本轮请求文本。
        branches: 候选一级分支路径列表。

    Returns:
        可直接作为单条 ``UserMsg`` 发给模型的提示文本。
    """
    listing = "\n".join(f"{index}. {path}" for index, path in enumerate(branches, start=1))
    return (
        "你是一个工具路由分类器。请从下列候选分支中选出与用户请求最相关的一个。\n"
        "只输出编号（一个阿拉伯数字），不要解释，不要输出其它任何内容。\n\n"
        f"候选分支：\n{listing}\n\n"
        f"用户请求：{query}"
    )


def parse_route_reply(reply: str, branches: Sequence[str]) -> RouteDecision:
    """解析模型回复为 ``RouteDecision``。

    解析失败或越界一律返回回退决策（F2：``branch is None`` 时 ``fallback=True``）。

    Args:
        reply: 模型原始回复文本。
        branches: 候选分支列表，与提示中的编号一一对应（从 1 起）。

    Returns:
        有效编号返回 ``confidence=1.0`` 的选择；否则返回回退决策。
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
    """由主模型代为执行的路由器。

    Attributes:
        chat: 依赖注入的单轮模型调用，签名为
            ``(messages: list[Message]) -> AssistantMsg``。
        confidence_threshold: 低于该置信度回退全量。
        history_tokens_provider: 可选回调，返回当前历史 token 数；
            省略时由调用方直接传入 ``history_tokens``。
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
        """选择工具分支；任何失败都回退全量（I4）。

        Args:
            query: 用户请求文本。
            branches: 候选一级分支路径列表。
            history_tokens: 当前历史 token 数（用于启用条件判定）。
            tool_count: 当前工具总数（用于启用条件判定）。

        Returns:
            :class:`RouteDecision`；不满足启用条件或调用失败时
            ``branch is None`` 且 ``fallback is True``。
        """
        if not branches:
            return RouteDecision.full(confidence=0.0)

        if not should_route(
            tool_count=tool_count,
            branch_count=len(branches),
            history_tokens=history_tokens,
        ):
            # 未达启用条件：不做路由，等价于全量（保守）
            return RouteDecision.full(confidence=0.0)

        prompt = build_route_prompt(query, branches)
        try:
            reply = self.chat([UserMsg(content=prompt)])
        except Exception:
            # 路由失败不得阻断主流程（I4）
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
        """按决策筛选分支：回退时返回全量（F2 的消费端）。

        Args:
            decision: 路由结果。
            branches: 全部候选分支。

        Returns:
            需要暴露给模型的分支列表。
        """
        if decision.fallback or decision.branch is None:
            return list(branches)
        return [decision.branch]


def parse_route_json(text: str) -> RouteDecision | None:
    """尝试把回复解析为 ``{"branch": ..., "confidence": ...}`` 形式。

    仅作为容错路径：模型偶尔会返回 JSON 而不是纯编号。

    Args:
        text: 模型回复。

    Returns:
        解析成功返回决策，否则 ``None``。
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
