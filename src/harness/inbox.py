"""接入层双通道（G 组）。

对应文档：docs/02-数据结构设计/01-接入层.md

结构层次::

    Inbox
    ├── control: deque[InterruptRequest]   立即插入通道，容量无限
    ├── chat: deque[UserInput]             等待插入通道，FIFO，容量上限 100
    └── lock: threading.Lock               保护两通道读写

本模块只做**队列语义**，不实现 ``AgentLoop``：

* 安全点的「消费什么」由调用方决定，这里只提供语义化的两个入口
  ``drain_at_full_safe_point``（① / ③，G4）与
  ``take_control_at_sub_batch``（②，G2）。
* 丢弃通知通过注入的 ``on_drop`` 回调完成，避免对 ``trace`` 模块的硬依赖
  （G8 / G12 要求写 ``TraceEvent``，由上层把 ``TraceEvent`` 写入逻辑注入进来）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CHAT_CAPACITY",
    "INTERRUPT_REASONS",
    "DropReason",
    "DropNotice",
    "UserInput",
    "InterruptRequest",
    "Inbox",
]

#: ``chat`` 通道容量上限（文档 §1.3）
CHAT_CAPACITY = 100

#: ``reason`` 的三个合法取值（G11）
INTERRUPT_REASONS: tuple[str, str, str] = ("user_abort", "session_end", "timeout")

#: 入队丢弃原因：G8（超容量）/ G12（空文本）
DropReason = Literal["overflow", "empty_text"]

#: 丢弃通知回调签名。上层收到后写 ``TraceEvent``。
DropCallback = Callable[["DropNotice"], None]


def _now() -> float:
    return time.time()


@dataclass(frozen=True, slots=True)
class UserInput:
    """``chat`` 通道消息元素。

    Attributes:
        text: 原始文本，不做改写；必须非空（空串在 ``Inbox.put_chat`` 处被拒绝，G12）。
        channel: 来源通道，``stdin``（终端）或 ``ws``（WebSocket）。
        ts: 入队时刻 Unix 时间戳。
    """

    text: str
    channel: Literal["stdin", "ws"]
    ts: float = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class InterruptRequest:
    """``control`` 通道中断信号。

    不变量（G11）：

    * ``reason`` 只能取 ``user_abort`` / ``session_end`` / ``timeout``；
    * 除 ``user_abort`` 外，``text`` 必须为 ``None``。

    Attributes:
        reason: 中断来源。``user_abort`` 由 ``Reader`` 写入，``session_end`` 由
            ``StepController`` 写入，``timeout`` 由 ``Executor`` 写入。
        text: 附带文本；非空时在全部补占位（3e）完成后作为 ``UserMsg`` 插入（G3）。
        ts: 入队时刻 Unix 时间戳。
    """

    reason: Literal["user_abort", "session_end", "timeout"]
    text: str | None = None
    ts: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        # 运行期双保险：Literal 只对静态检查生效，绕过类型检查的调用也要拦住。
        if self.reason not in INTERRUPT_REASONS:
            raise ValueError(f"G11: reason 只能取 {INTERRUPT_REASONS}，实际为 {self.reason!r}")
        if self.reason != "user_abort" and self.text is not None:
            raise ValueError(
                f"G11: reason={self.reason!r} 时 text 必须为 None，实际为 {self.text!r}"
            )


@dataclass(frozen=True, slots=True)
class DropNotice:
    """一次入队丢弃的通知内容，供上层构造 ``TraceEvent``。

    Attributes:
        channel: 被丢弃消息所属通道，恒为 ``"chat"``（``control`` 无容量上限）。
        reason: ``overflow``（G8）或 ``empty_text``（G12）。
        dropped: 被丢弃的元素；``empty_text`` 时 ``None``。
        text: 被丢弃消息的文本；``empty_text`` 时为原始空文本。
        ts: 丢弃发生时刻。
    """

    channel: Literal["chat"]
    reason: DropReason
    dropped: UserInput | None
    text: str
    ts: float = field(default_factory=_now)


class Inbox:
    """会话级双通道收件箱。

    线程模型（G5 / G6）：

    * 写入方可为多个（``Reader`` / ``StepController`` / ``Executor``），全部持 ``lock``；
    * 消费方仅主线程，同样持 ``lock``；
    * 临界区**只做队列操作**，不得在锁内调用工具或模型。
    """

    def __init__(
        self,
        capacity: int = CHAT_CAPACITY,
        on_drop: DropCallback | None = None,
    ) -> None:
        """初始化空收件箱。

        Args:
            capacity: ``chat`` 通道容量上限，默认 100。
            on_drop: 丢弃通知回调（G8 / G12）。回调在**锁外**调用，避免上层
                写 ``TraceEvent`` 的 IO 阻塞写入方。
        """
        self._control: deque[InterruptRequest] = deque()
        self._chat: deque[UserInput] = deque()
        self._lock = threading.Lock()
        self._capacity = capacity
        self._on_drop = on_drop

    # ------------------------------------------------------------------
    # 只读观测（持锁，供注册表卸载判定使用）
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """``chat`` 通道容量上限。"""
        return self._capacity

    def is_empty(self) -> bool:
        """两通道是否均为空（持锁）。"""
        with self._lock:
            return not self._control and not self._chat

    def __len__(self) -> int:
        """两通道元素总数（持锁）。"""
        with self._lock:
            return len(self._control) + len(self._chat)

    def __repr__(self) -> str:
        with self._lock:
            return f"Inbox(control={len(self._control)}, chat={len(self._chat)})"

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def put_chat(self, item: UserInput) -> bool:
        """写入 ``chat`` 通道（FIFO）。

        空文本拒绝入队并通知（G12）；超容量丢弃最旧元素并通知（G8）。

        Args:
            item: 用户输入。

        Returns:
            是否成功入队。
        """
        notice: DropNotice | None = None
        accepted = True

        with self._lock:
            if not item.text:
                accepted = False
                notice = DropNotice(
                    channel="chat", reason="empty_text", dropped=None, text=item.text
                )
            else:
                if len(self._chat) >= self._capacity:
                    dropped = self._chat.popleft()
                    notice = DropNotice(
                        channel="chat", reason="overflow", dropped=dropped, text=dropped.text
                    )
                self._chat.append(item)

        self._emit(notice)
        return accepted

    def put_control(self, item: InterruptRequest) -> None:
        """写入 ``control`` 通道（容量无限）。"""
        with self._lock:
            self._control.append(item)

    # ------------------------------------------------------------------
    # 消费
    # ------------------------------------------------------------------

    def take_control(self) -> InterruptRequest | None:
        """取走首个 ``control`` 信号并清空剩余（G7）。

        Returns:
            首个信号；通道为空时返回 ``None``。
        """
        with self._lock:
            if not self._control:
                return None
            first = self._control.popleft()
            self._control.clear()
            return first

    def drain_chat(self) -> list[UserInput]:
        """取走全部 ``chat`` 消息，保持 FIFO 顺序。"""
        with self._lock:
            items = list(self._chat)
            self._chat.clear()
            return items

    # ------------------------------------------------------------------
    # 安全点语义封装（不实现 AgentLoop）
    # ------------------------------------------------------------------

    def drain_at_full_safe_point(self) -> tuple[InterruptRequest | None, list[UserInput]]:
        """① LLM 返回后 / ③ 全批完成的完整安全点消费（G4）。

        两通道均有消息时，先处理 ``control``，再 ``drain`` ``chat``。

        Returns:
            ``(control 信号或 None, 全部 chat 消息)``。
        """
        interrupt = self.take_control()
        return interrupt, self.drain_chat()

    def take_control_at_sub_batch(self) -> InterruptRequest | None:
        """② 组间子批次安全点消费（G2）。

        只取 ``control``；``chat`` 消息留在队列中，既不出队也不丢弃。
        """
        return self.take_control()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _emit(self, notice: DropNotice | None) -> None:
        """在锁外投递丢弃通知。"""
        if notice is not None and self._on_drop is not None:
            self._on_drop(notice)
