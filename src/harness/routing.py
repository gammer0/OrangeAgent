"""Routing results (group F, F2).

See docs/02-data-structures/08-runtime-state.md.

This module **defines data structures only**; it does not implement the Jev
routing algorithm (that belongs to the ingress layer).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["RouteDecision"]


class RouteDecision(BaseModel):
    """Jev routing result.

    F2: when ``branch is None``, ``fallback`` must be ``True`` (fall back to the
    full tool tree).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    fallback: bool = False

    @model_validator(mode="after")
    def _check_fallback(self) -> RouteDecision:
        if self.branch is None and not self.fallback:
            raise ValueError("F2: fallback must be True when branch is None")
        return self

    @classmethod
    def full(cls, *, confidence: float = 1.0, fallback: bool = True) -> RouteDecision:
        """Build a full-fallback decision (``branch=None``; F2 requires ``fallback=True``)."""
        return cls(branch=None, confidence=confidence, fallback=fallback)
