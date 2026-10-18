"""会话层（H 组）。

对应文档：docs/02-数据结构设计/02-会话层.md

结构层次::

    SessionRegistry
    ├── sessions: dict[str, Session]
    ├── lock: threading.Lock
    └── policy: EvictionPolicy

跨分支耦合说明（重要）
--------------------
会话层需要持有 ``Watermark`` / ``TreeQueryState`` / ``DiscoveredTools`` 三个
**其他组**的会话级结构。为避免在本分支引入上下文层（F / B 组）模块的导入依赖，
本模块只定义 ``Watermark`` 的**最小占位**与 ``TreeQueryState`` / ``DiscoveredTools``
的**协议接口**（``Protocol``）：

* ``Watermark`` 占位只包含文档 §8.1 的三个字段（``capacity`` / ``compressed_at`` /
  ``last_estimate``）及序列化辅助，**不包含**水位判定逻辑（F1）。
* ``TreeQueryState`` / ``DiscoveredTools`` 用 ``Protocol`` 描述字段形状，
  运行时不需要导入上下文层/能力层实现，装配时可传入任一等价对象。

``SessionTreeQuery.clear_delivered`` / ``DiscoveredTools.clear`` 未在 B 组文档中
写明方法名，这里通过 ``_reset_discovered`` 采用「先尝试 ``clear()``，否则回退构造
新实例」的宽松策略，保证 H8 / H14 的语义可落地且不绑死 B 组 API。
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from harness.inbox import Inbox
from harness.message import Message

__all__ = [
    "MAX_RESIDENT",
    "IDLE_TIMEOUT_S",
    "SWEEP_INTERVAL_S",
    "SessionStatus",
    "Watermark",
    "TreeQueryStateProtocol",
    "DiscoveredToolsProtocol",
    "Session",
    "EvictionPolicy",
    "SessionRecord",
    "SessionRegistry",
    "new_session_id",
]

#: 驱逐策略默认值（文档 §2.1）
MAX_RESIDENT = 20
IDLE_TIMEOUT_S = 1800
SWEEP_INTERVAL_S = 60

SessionStatus = Literal["active", "idle", "ended"]


def new_session_id() -> str:
    """生成全局唯一 ``session_id``（路由键）。"""
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
# 跨分支依赖的最小占位
# ----------------------------------------------------------------------


@dataclass
class Watermark:
    """上下文水位记录的**最小占位**（字段同文档 §8.1）。

    真实实现位于上下文层（F 组），本分支只保留数据结构以避免跨分支导入耦合。
    水位判定（``JudgeResult``）与 ``compressed_at`` 的只增语义（F1）不在本类实现。
    """

    capacity: int = 0
    compressed_at: set[int] = field(default_factory=set)
    last_estimate: int = 0

    def to_json(self) -> dict[str, Any]:
        """序列化为可写入 ``sessions.watermark`` 的 JSON 兼容字典。"""
        return {
            "capacity": self.capacity,
            "compressed_at": sorted(self.compressed_at),
            "last_estimate": self.last_estimate,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Watermark | None:
        """从 ``sessions.watermark`` 反序列化；``None`` 原样返回。"""
        if data is None:
            return None
        raw_compressed = data.get("compressed_at") or []
        return cls(
            capacity=int(data.get("capacity", 0)),
            compressed_at={int(level) for level in raw_compressed},
            last_estimate=int(data.get("last_estimate", 0)),
        )


@runtime_checkable
class TreeQueryStateProtocol(Protocol):
    """``TreeQueryState`` 的结构性接口（B 组）。

    恢复/卸载后必须置空 ``delivered``（H8 / H14），因此本层只依赖该字段可变。
    """

    delivered: set[str]


@runtime_checkable
class DiscoveredToolsProtocol(Protocol):
    """``DiscoveredTools`` 的结构性接口（B 组）。

    恢复/卸载后必须置空清单（H8 / H14）。实现类应提供 ``clear()``，否则由
    ``reset_discovered`` 回退为构造新实例。
    """

    entries: dict[str, Any]

    def clear(self) -> None:  # pragma: no cover - 结构性声明
        ...


def reset_tree_query(tree_query: TreeQueryStateProtocol) -> None:
    """置空 ``TreeQueryState.delivered``（H8 / H14）。"""
    tree_query.delivered.clear()


def reset_discovered(discovered: DiscoveredToolsProtocol) -> Any:
    """置空 ``DiscoveredTools``（H8 / H14）。

    优先调用 ``clear()``；实现类若未提供，则返回一个空清单的新实例供调用方替换。
    返回值仅供“回退构造”路径使用，正常路径返回 ``None``。
    """
    clear = getattr(discovered, "clear", None)
    if callable(clear):
        clear()
        return None

    # 回退路径：构造一个同类型新实例，并尽力清空其 entries 容器。
    replacement = type(discovered)()
    entries = getattr(replacement, "entries", None)
    if isinstance(entries, dict):
        entries.clear()
    return replacement


# ----------------------------------------------------------------------
# 会话
# ----------------------------------------------------------------------


@dataclass
class Session:
    """单个会话的运行时状态。

    隔离边界（H1）：``messages`` / ``inbox`` / ``watermark`` / ``tree_query``
    均须为会话独占对象，不得跨会话共享引用。

    ``tree_version`` 在创建时固定（H15）：工具树后续挂载/卸载只影响新会话。
    """

    session_id: str
    working_dir: str
    messages: list[Message] = field(default_factory=list)
    inbox: Inbox = field(default_factory=Inbox)
    watermark: Watermark = field(default_factory=Watermark)
    tree_query: TreeQueryStateProtocol = field(default_factory=lambda: _default_tree_query())
    tree_version: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    status: SessionStatus = "idle"

    def touch(self) -> None:
        """刷新 ``last_activity``，用于驱逐排序。"""
        self.last_activity = time.time()

    def busy(self) -> None:
        """标记会话正在推理：``active`` 会话不会被 ``sweep`` 卸载（H10）。"""
        self.status = "active"
        self.touch()

    def release(self) -> None:
        """推理结束，会话回到 ``idle``，重新可被驱逐。"""
        if self.status != "ended":
            self.status = "idle"
        self.touch()


def _default_tree_query() -> TreeQueryStateProtocol:
    """构造默认的空 ``TreeQueryState`` 占位对象（避免导入 B 组实现）。"""

    class _DefaultTreeQuery:
        def __init__(self) -> None:
            self.delivered: set[str] = set()

    return _DefaultTreeQuery()


@dataclass
class EvictionPolicy:
    """会话驱逐策略（文档 §2.1 / §2.4）。"""

    max_resident: int = MAX_RESIDENT
    idle_timeout_s: int = IDLE_TIMEOUT_S
    sweep_interval_s: int = SWEEP_INTERVAL_S


@dataclass
class SessionRecord:
    """``sessions`` 表的持久化行（SQLite 投影）。"""

    session_id: str
    working_dir: str
    created_at: float
    updated_at: float
    status: SessionStatus
    watermark: Watermark | None = None


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------


class SessionRegistry:
    """多会话注册表。

    线程模型（H2）：所有读写持 ``self._lock``；临界区**只做字典操作**，
    不得包含 LLM 调用或工具执行。
    """

    def __init__(self, policy: EvictionPolicy | None = None) -> None:
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.policy = policy or EvictionPolicy()
        # 后台清扫线程（S2）：懒启动，默认不运行
        self._sweeper: threading.Thread | None = None
        self._sweeper_stop = threading.Event()
        self._sweeper_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 基本读写
    # ------------------------------------------------------------------

    def add(self, session: Session) -> Session:
        """登记会话（持锁）。同 id 已存在时覆盖。"""
        with self.lock:
            self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """按 id 取会话（持锁）；不存在返回 ``None``。"""
        with self.lock:
            return self.sessions.get(session_id)

    def remove(self, session_id: str) -> Session | None:
        """从内存移除会话（持锁），返回被移除对象；不存在返回 ``None``。

        仅丢弃内存引用，不触碰数据库（H13）。
        """
        with self.lock:
            return self.sessions.pop(session_id, None)

    def end(self, session_id: str) -> Session | None:
        """标记会话 ``ended`` 并立即从内存移除（H12）。

        Returns:
            被移除的会话对象；不存在返回 ``None``。
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            session.status = "ended"
            return self.sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self.lock:
            return len(self.sessions)

    def __contains__(self, session_id: object) -> bool:
        with self.lock:
            return session_id in self.sessions

    # ------------------------------------------------------------------
    # 驱逐
    # ------------------------------------------------------------------

    def _evictable(self, session: Session, now: float) -> bool:
        """会话是否可被卸载：不活跃（``idle``）且无待处理消息（H10）。"""
        return session.status == "idle" and session.inbox.is_empty()

    def sweep(self, now: float | None = None) -> list[str]:
        """执行一次驱逐扫描（文档 §2.4）。

        三段规则，均在**同一把锁**内完成（H11“卸载须持锁”）：

        1. 持锁移除全部 ``ended`` 会话（H12）；
        2. **空闲超时**：``now - last_activity > idle_timeout_s`` 且可卸载者，
           一律卸载。这是唯一不依赖会话数量的回收路径；
        3. **容量溢出**：若驻留数仍超过 ``max_resident``，按 ``last_activity``
           升序继续卸载可卸载会话，直到满足上限或无可卸载对象。

        可卸载（H10）= ``status == "idle"`` 且 ``inbox`` 为空。
        ``active`` 会话（正在推理）永不卸载，因此驻留数可能短暂超过
        ``max_resident`` —— 上限是目标而非硬约束。

        Args:
            now: 逻辑当前时间（秒）。默认取 ``time.time()``；测试可注入。

        Returns:
            被卸载（含 ``ended`` 移除）的 ``session_id`` 列表。
        """
        current = time.time() if now is None else now
        evicted: list[str] = []

        with self.lock:
            for session_id in [sid for sid, s in self.sessions.items() if s.status == "ended"]:
                self.sessions.pop(session_id, None)
                evicted.append(session_id)

            # ① 空闲超时淘汰（与容量无关）
            timed_out = [
                s
                for s in self.sessions.values()
                if self._evictable(s, current)
                and current - s.last_activity > self.policy.idle_timeout_s
            ]
            for session in timed_out:
                self.sessions.pop(session.session_id, None)
                evicted.append(session.session_id)

            # ② 容量溢出淘汰（LRU）
            overflow = len(self.sessions) - self.policy.max_resident
            if overflow > 0:
                candidates = sorted(
                    (s for s in self.sessions.values() if self._evictable(s, current)),
                    key=lambda s: s.last_activity,
                )
                for session in candidates[:overflow]:
                    self.sessions.pop(session.session_id, None)
                    evicted.append(session.session_id)

        return evicted

    def session_ids(self) -> list[str]:
        """当前内存驻留的全部会话 id（快照，持锁）。"""
        with self.lock:
            return list(self.sessions.keys())

    # ------------------------------------------------------------------
    # 周期清扫（S2）
    # ------------------------------------------------------------------

    def start_sweeper(self) -> None:
        """启动后台清扫线程，按 ``policy.sweep_interval_s`` 周期调用 ``sweep()``。

        幂等：已在运行时直接返回。线程为守护线程，不阻止进程退出；
        ``stop_sweeper()`` 会尽快唤醒并结束它（``Event.wait`` 可被立即打断，
        因此不会为等待一个完整周期而阻塞关闭）。
        """
        with self._sweeper_lock:
            if self._sweeper is not None and self._sweeper.is_alive():
                return

            self._sweeper_stop.clear()
            thread = threading.Thread(
                target=self._sweeper_loop,
                name="harness-session-sweeper",
                daemon=True,
            )
            self._sweeper = thread
            thread.start()

    def stop_sweeper(self, timeout: float = 2.0) -> None:
        """通知后台清扫线程退出并等待其结束（幂等）。"""
        with self._sweeper_lock:
            thread = self._sweeper
            self._sweeper = None
            self._sweeper_stop.set()

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def sweeping(self) -> bool:
        """后台清扫线程是否正在运行。"""
        with self._sweeper_lock:
            return self._sweeper is not None and self._sweeper.is_alive()

    def _sweeper_loop(self) -> None:
        """清扫线程主体：等待间隔 → 扫描，直到被要求停止。"""
        interval = max(1, int(self.policy.sweep_interval_s))
        # 首轮延后一个完整周期，避免服务刚启动就因空表扫描产生噪音
        while not self._sweeper_stop.wait(interval):
            # 清扫是尽力而为，任何异常都不应拖垮服务或结束线程
            with contextlib.suppress(Exception):
                self.sweep()
