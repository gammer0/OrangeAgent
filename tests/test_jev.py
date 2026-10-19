"""Jev 路由测试：启用条件、主模型代为选择、失败回退（I4 / F2）。"""

from __future__ import annotations

from typing import Any

from orange.jev import (
    CONFIDENCE_THRESHOLD,
    JevRouter,
    build_route_prompt,
    parse_route_json,
    parse_route_reply,
    should_route,
)
from orange.message import AssistantMsg

BRANCHES = [f"/skill/mmd/branch{i}" for i in range(1, 9)]


def _router(reply: str | Exception) -> JevRouter:
    def chat(messages: list[Any]) -> AssistantMsg:
        if isinstance(reply, Exception):
            raise reply
        return AssistantMsg(content=reply)

    return JevRouter(chat=chat)


class TestEnableConditions:
    def test_all_thresholds_required(self) -> None:
        assert should_route(tool_count=30, branch_count=6, history_tokens=4001)
        assert not should_route(tool_count=29, branch_count=6, history_tokens=4001)
        assert not should_route(tool_count=30, branch_count=5, history_tokens=4001)
        assert not should_route(tool_count=30, branch_count=6, history_tokens=4000)

    def test_below_threshold_returns_fallback(self) -> None:
        """少量工具查树纯亏推理，直接回退全量。"""
        decision = _router("1").route("问题", BRANCHES[:3], tool_count=3)
        assert decision.branch is None
        assert decision.fallback


class TestPromptConstruction:
    def test_prompt_contains_only_query_and_branches(self) -> None:
        prompt = build_route_prompt("读取配置", BRANCHES[:3])
        assert "读取配置" in prompt
        for branch in BRANCHES[:3]:
            assert branch in prompt
        # 防 JPDH：提示中不得出现工具输出标记
        assert "tool_result" not in prompt
        assert "ExecResult" not in prompt

    def test_branches_are_numbered_from_one(self) -> None:
        prompt = build_route_prompt("q", ["/a", "/b"])
        assert "1. /a" in prompt
        assert "2. /b" in prompt


class TestReplyParsing:
    def test_plain_number(self) -> None:
        decision = parse_route_reply("3", BRANCHES)
        assert decision.branch == BRANCHES[2]
        assert decision.confidence == 1.0
        assert not decision.fallback

    def test_tolerates_prefix_and_noise(self) -> None:
        for reply in ("B2", "分支 2", "答案是 2。", " 2 "):
            assert parse_route_reply(reply, BRANCHES).branch == BRANCHES[1]

    def test_out_of_range_falls_back(self) -> None:
        decision = parse_route_reply("99", BRANCHES)
        assert decision.branch is None
        assert decision.fallback

    def test_unparsable_falls_back(self) -> None:
        decision = parse_route_reply("我不知道", BRANCHES)
        assert decision.branch is None
        assert decision.fallback

    def test_json_reply_supported(self) -> None:
        decision = parse_route_json('{"branch": "/a/b", "confidence": 0.9}')
        assert decision is not None
        assert decision.branch == "/a/b"

    def test_json_invalid_returns_none(self) -> None:
        assert parse_route_json("not json") is None
        assert parse_route_json('{"branch": 1}') is None


class TestRouterBehaviour:
    def test_selects_branch_when_enabled(self) -> None:
        router = _router("2")
        decision = router.route(
            "问题", BRANCHES, tool_count=50, history_tokens=10000
        )
        assert decision.branch == BRANCHES[1]
        assert not decision.fallback

    def test_model_exception_falls_back_I4(self) -> None:
        router = _router(RuntimeError("模型挂了"))
        decision = router.route("q", BRANCHES, tool_count=50, history_tokens=10000)
        assert decision.branch is None
        assert decision.fallback

    def test_empty_branches_falls_back(self) -> None:
        decision = _router("1").route("q", [], tool_count=50, history_tokens=10000)
        assert decision.branch is None

    def test_low_confidence_falls_back(self) -> None:
        router = JevRouter(chat=lambda _m: AssistantMsg(content="x"), confidence_threshold=0.5)
        decision = router.route("q", BRANCHES, tool_count=50, history_tokens=10000)
        assert decision.confidence < CONFIDENCE_THRESHOLD
        assert decision.fallback

    def test_visible_branches_full_on_fallback_F2(self) -> None:
        router = _router("1")
        fallback = router.route("q", BRANCHES, tool_count=1)
        assert router.visible_branches(fallback, BRANCHES) == BRANCHES

        chosen = router.route("q", BRANCHES, tool_count=50, history_tokens=10000)
        assert router.visible_branches(chosen, BRANCHES) == [BRANCHES[0]]
