"""模型层运行时：限流与健康状态（ADT-10，K 组）。

对应文档：docs/02-数据结构设计/10-模型层.md

多会话共享同一个 ``ReverseProxy``，因此限流窗口与健康状态都是**全局**的。

* K1：只做全局速率窗口，不做每会话并发限制。
* K2：每个 ``RateEvent`` 必须携带 ``session_id``。
* K3：窗口淘汰只按 ``ts`` 滑出，不按会话。
* K4：``ProviderHealth`` 全局单例，退避与故障转移对所有会话生效。
* K5：``state != "healthy"`` 期间的失败不重复累加 ``consecutive_failures``。

所有状态读写均由 ``threading.Lock`` 保护。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RateEvent",
    "RateWindow",
    "Decision",
    "ProviderState",
    "ProviderHealth",
    "get_provider_health",
    "reset_provider_health_registry",
]

Decision = Literal["allow", "retry_after", "fail_fast"]
ProviderState = Literal["healthy", "degraded", "down"]


# ------------------------------------------------------------------- 限流


class RateEvent(BaseModel):
    """窗口内的单次请求记录。

    ``tokens`` 为请求前估算；``actual_tokens`` 由 :meth:`RateWindow.record_actual`
    在响应后回填。
    """

    model_config = ConfigDict(extra="forbid")

    ts: float
    session_id: Annotated[str, Field(min_length=1)]  # K2
    tokens: int
    actual_tokens: int | None = None


class RateWindow:
    """滑动窗口限速器（全局一份）。

    线程安全：所有读写都在 ``self._lock`` 内完成。
    """

    def __init__(self, max_requests: int, max_tokens: int, window_s: int = 60) -> None:
        max_requests = int(max_requests)
        max_tokens = int(max_tokens)
        window_s = int(window_s)
        if max_requests <= 0:
            raise ValueError("max_requests 必须为正整数")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须为正整数")
        if window_s <= 0:
            raise ValueError("window_s 必须为正整数")
        self.window_s = window_s
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self._events: deque[RateEvent] = deque()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 视图

    @property
    def events(self) -> list[RateEvent]:
        """窗口内事件的快照，按 ``ts`` 升序。"""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ---------------------------------------------------------------- 淘汰

    def _evict_locked(self, now: float) -> None:
        """滑出 ``ts <= now - window_s`` 的事件（K3：只按 ts，不按会话）。"""
        cutoff = now - self.window_s
        events = self._events
        while events and events[0].ts <= cutoff:
            events.popleft()

    def _used_tokens_locked(self) -> int:
        """窗口内已占用 token 总数（回填优先使用 actual_tokens）。"""
        total = 0
        for event in self._events:
            total += event.actual_tokens if event.actual_tokens is not None else event.tokens
        return total

    # ---------------------------------------------------------------- 获取

    def try_acquire(self, session_id: str, tokens: int, now: float) -> bool:
        """尝试占用一个请求名额；窗口已满则拒绝并返回 ``False``。

        拒绝时不写入事件，因此不会污染窗口。

        Args:
            session_id: 发起请求的会话（K2）。
            tokens: 请求前估算的 token 数。
            now: 当前时间戳（Unix 秒）。
        """
        if not session_id:
            raise ValueError("K2: session_id 不得为空")
        if tokens < 0:
            raise ValueError("tokens 不得为负")
        with self._lock:
            self._evict_locked(now)
            if len(self._events) + 1 > self.max_requests:
                return False
            if self._used_tokens_locked() + tokens > self.max_tokens:
                return False
            self._events.append(RateEvent(ts=now, session_id=session_id, tokens=tokens))
            return True

    # ---------------------------------------------------------------- 回填

    def record_actual(self, session_id: str, actual_tokens: int, now: float | None = None) -> bool:
        """把某会话**最早一条**未回填事件的 ``actual_tokens`` 回填（K2）。

        回填按 ``session_id`` 匹配，因此并发会话的记录不会互相串写。
        未找到可回填事件时返回 ``False``。

        Args:
            session_id: 会话 id。
            actual_tokens: 响应后得到的真实 token 数。
            now: 当前时间戳；传入时会先滑出窗口再回填。
        """
        if not session_id:
            raise ValueError("K2: session_id 不得为空")
        if actual_tokens < 0:
            raise ValueError("actual_tokens 不得为负")
        with self._lock:
            if now is not None:
                self._evict_locked(now)
            for event in self._events:
                if event.session_id == session_id and event.actual_tokens is None:
                    event.actual_tokens = actual_tokens
                    return True
            return False


# --------------------------------------------------------------- 健康状态

#: K4：全局单例注册表。键为 provider 名，所有会话共享同一实例。
_REGISTRY: dict[str, ProviderHealth] = {}
_REGISTRY_LOCK = threading.Lock()


class ProviderHealth:
    """单个 provider 的全局健康状态（K4：全局单例）。

    单例语义：一个 provider 名在整个进程内只对应一份状态，
    请通过 :func:`get_provider_health` 获取，**不要**自行构造实例，
    否则退避与故障转移不会对所有会话生效。
    """

    def __init__(
        self, provider: str, *, failure_threshold: int = 3, backoff_s: float = 30.0
    ) -> None:
        if not provider:
            raise ValueError("provider 不得为空")
        self.provider = provider
        self.state: ProviderState = "healthy"
        self.consecutive_failures = 0
        self.retry_after = 0.0
        self.switched_to: str | None = None
        self.failure_threshold = max(1, int(failure_threshold))
        self.backoff_s = float(backoff_s)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 判定

    def can_send(self, now: float) -> Decision:
        """发送前检查。

        Returns:
            ``"allow"`` 正常发送；``"retry_after"`` 需等待 ``retry_after``；
            ``"fail_fast"`` 直接失败不重试。
        """
        with self._lock:
            if self.state == "healthy":
                return "allow"
            if self.state == "degraded":
                return "allow" if now >= self.retry_after else "retry_after"
            return "fail_fast"

    # ---------------------------------------------------------------- 迁移

    def on_success(self) -> None:
        """一次成功调用：清零失败计数并恢复 ``healthy``。"""
        with self._lock:
            self.consecutive_failures = 0
            self.state = "healthy"
            self.retry_after = 0.0
            self.switched_to = None

    def on_failure(self, now: float, threshold: int | None = None) -> ProviderState:
        """一次失败调用，返回迁移后的状态。

        K5：``state != "healthy"`` 期间不重复累加 ``consecutive_failures``，
        但会延长 ``retry_after``，使退避持续生效。

        Args:
            now: 当前时间戳。
            threshold: 本次调用的失败阈值，缺省用实例配置。
        """
        limit = max(1, int(threshold if threshold is not None else self.failure_threshold))
        with self._lock:
            if self.state == "down":
                # K5：down 期间不累加；仅保持 fail-fast
                return self.state
            if self.state == "degraded":
                # K5：不重复累加，但刷新退避截止时间
                self.retry_after = now + self.backoff_s
                return self.state
            self.consecutive_failures += 1
            if self.consecutive_failures >= limit:
                self.state = "degraded"
                self.retry_after = now + self.backoff_s
            return self.state

    def mark_down(self, switched_to: str | None = None) -> None:
        """显式降级为 ``down``（例如故障转移已选定目标）。"""
        with self._lock:
            self.state = "down"
            self.switched_to = switched_to

    # ---------------------------------------------------------------- 快照

    def snapshot(self) -> dict[str, object]:
        """线程安全的只读快照。"""
        with self._lock:
            return {
                "provider": self.provider,
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "retry_after": self.retry_after,
                "switched_to": self.switched_to,
            }

    def __repr__(self) -> str:
        data = self.snapshot()
        return (
            f"ProviderHealth(provider={data['provider']!r}, state={data['state']!r}, "
            f"consecutive_failures={data['consecutive_failures']!r})"
        )


def get_provider_health(
    provider: str, *, failure_threshold: int = 3, backoff_s: float = 30.0
) -> ProviderHealth:
    """获取（必要时创建）provider 的全局单例健康状态（K4）。

    同一 ``provider`` 名重复调用返回**同一对象**；首次创建时的
    ``failure_threshold`` / ``backoff_s`` 生效，后续参数被忽略。
    """
    with _REGISTRY_LOCK:
        health = _REGISTRY.get(provider)
        if health is None:
            health = ProviderHealth(
                provider, failure_threshold=failure_threshold, backoff_s=backoff_s
            )
            _REGISTRY[provider] = health
        return health


def reset_provider_health_registry() -> None:
    """清空全局注册表（仅供测试使用）。"""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
