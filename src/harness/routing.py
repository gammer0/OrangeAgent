"""路由结果（F 组，F2）。

对应文档：docs/02-数据结构设计/08-运行时状态.md

本模块**只定义数据结构**，不实现 Jev 路由算法（算法属接入层）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["RouteDecision"]


class RouteDecision(BaseModel):
    """Jev 路由结果。

    F2：``branch is None`` 时 ``fallback`` 必须为 ``True``（回退全量工具树）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    fallback: bool = False

    @model_validator(mode="after")
    def _check_fallback(self) -> RouteDecision:
        if self.branch is None and not self.fallback:
            raise ValueError("F2: branch 为 None 时 fallback 必须为 True")
        return self

    @classmethod
    def full(cls, *, confidence: float = 1.0, fallback: bool = True) -> RouteDecision:
        """构造回退全量的决策（``branch=None``，F2 要求 ``fallback=True``）。"""
        return cls(branch=None, confidence=confidence, fallback=fallback)
