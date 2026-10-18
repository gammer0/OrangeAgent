"""路由结果测试（F2）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.routing import RouteDecision


class TestRouteDecision:
    def test_accepts_branch_without_fallback(self) -> None:
        decision = RouteDecision(branch="/tool/fs", confidence=0.9)
        assert decision.branch == "/tool/fs"
        assert decision.fallback is False

    def test_accepts_none_branch_with_fallback(self) -> None:
        decision = RouteDecision(branch=None, confidence=0.2, fallback=True)
        assert decision.fallback is True

    def test_f2_none_branch_requires_fallback(self) -> None:
        with pytest.raises(ValidationError):
            RouteDecision(branch=None, confidence=0.2)

    def test_f2_default_construction_rejected(self) -> None:
        """默认 branch=None 且 fallback=False，必须被拒绝（F2）。"""
        with pytest.raises(ValidationError):
            RouteDecision(confidence=1.0)

    def test_f2_error_message_mentions_rule(self) -> None:
        with pytest.raises(ValidationError) as exc:
            RouteDecision(branch=None, confidence=0.0)
        assert "F2" in str(exc.value)

    def test_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            RouteDecision(branch="/a", confidence=1.5)
        with pytest.raises(ValidationError):
            RouteDecision(branch="/a", confidence=-0.1)

    def test_immutable_and_forbids_extra(self) -> None:
        decision = RouteDecision(branch="/a", confidence=0.5)
        with pytest.raises(ValidationError):
            decision.branch = "/b"
        with pytest.raises(ValidationError):
            RouteDecision(branch="/a", confidence=0.5, extra_field=1)

    def test_full_helper(self) -> None:
        decision = RouteDecision.full(confidence=0.3)
        assert decision.branch is None
        assert decision.fallback is True

    def test_full_helper_still_checks_f2(self) -> None:
        with pytest.raises(ValidationError):
            RouteDecision.full(fallback=False)
