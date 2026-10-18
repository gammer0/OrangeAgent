"""轨迹日志（ADT-7，E 组）。

对应文档：docs/02-数据结构设计/07-轨迹.md

一个会话一个 JSONL 文件；每个事件一行、立即落盘。

* E1：每步结束即追加并 flush（可选 fsync），不缓存到最后统一写。
* E2：每行必须是完整可解析的 JSON 对象（失败则整行不写）。
* E3：每条事件必带 ``session_id``，且文件按会话隔离。

线程模型：``TraceLogger`` 内部持锁，多线程写同一文件时不会交错半行；
不同会话写不同文件，互不阻塞。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "TraceKind",
    "TraceEvent",
    "TraceLogger",
    "TRACE_DIR_NAME",
    "trace_file_path",
]

#: 轨迹子目录名：``<trace_root>/<session_id>.jsonl``
TRACE_DIR_NAME = "traces"

TraceKind = Literal[
    "llm_call",
    "tool_exec",
    "compress",
    "inject",
    "error",
    "abort",
    "memory_extract",
]


def trace_file_path(trace_root: str | os.PathLike[str], session_id: str) -> Path:
    """返回会话轨迹文件路径 ``<trace_root>/<session_id>.jsonl``。

    文件名对 ``session_id`` 做安全化（仅保留字母数字与 ``-_.``），
    避免路径穿越（E3：按会话隔离，不允许跑到别的目录）。
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in session_id)
    if not safe:
        raise ValueError("E3: session_id 不得为空")
    return Path(trace_root) / f"{safe}.jsonl"


class TraceEvent(BaseModel):
    """一条轨迹事件（ADT-7）。

    ``payload`` 结构见文档 §7.2；本模型不逐 kind 校验 payload 字段，
    以便向前兼容（新增字段不破坏旧日志解析）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=0)]
    ts: float
    kind: TraceKind
    payload: dict[str, Any] = Field(default_factory=dict)

    # --------------------------------------------------------------- 序列化

    def to_json_line(self) -> str:
        """序列化为单行 JSON（不含换行符）；E2 保证可解析。"""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_json_line(cls, line: str) -> TraceEvent:
        """从一行 JSON 反序列化（用于日志回读与测试）。"""
        return cls.model_validate(json.loads(line))


class TraceLogger:
    """会话级轨迹写入器：一会话一份 JSONL（E1/E2/E3）。

    每次 ``log`` 都追加一整行并 flush，因此**其它进程/测试立即读到**（E1）。
    锁粒度为实例，多个实例指向同一文件时由操作系统追加语义 + 单次
    ``write`` 调用保证整行不被撕裂；同一实例内多线程完全安全。
    """

    def __init__(
        self,
        trace_root: str | os.PathLike[str],
        session_id: str,
        *,
        fsync: bool = False,
        clock: Any = time.time,
    ) -> None:
        if not session_id:
            raise ValueError("E3: session_id 不得为空")
        self._session_id = session_id
        self._path = trace_file_path(trace_root, session_id)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._clock = clock
        self._lock = threading.Lock()

    # --------------------------------------------------------------- 属性

    @property
    def session_id(self) -> str:
        """所属会话 id（E3）。"""
        return self._session_id

    @property
    def path(self) -> Path:
        """JSONL 文件路径。"""
        return self._path

    # --------------------------------------------------------------- 写入

    def log(self, step: int, kind: TraceKind, payload: dict[str, Any] | None = None) -> TraceEvent:
        """追加一条事件并立即落盘，返回写入的事件对象。

        Args:
            step: ReAct 步号。
            kind: 事件类型。
            payload: 事件载荷，字段见文档 §7.2。
        """
        event = TraceEvent(
            session_id=self._session_id,
            step=step,
            ts=float(self._clock()),
            kind=kind,
            payload=payload or {},
        )
        line = event.to_json_line() + "\n"
        # 单次 write 写完整行；持锁保证同一实例内顺序与不交错
        with self._lock, self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        return event

    # ------------------------------------------------------- 7 个便捷方法

    def log_llm_call(
        self,
        step: int,
        *,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        tool_call_count: int,
    ) -> TraceEvent:
        """记录一次 LLM 调用。"""
        return self.log(
            step,
            "llm_call",
            {
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_ms": duration_ms,
                "tool_call_count": tool_call_count,
            },
        )

    def log_tool_exec(
        self,
        step: int,
        *,
        path: str,
        tool_id: str | None,
        arguments: Any,
        ok: bool,
        error_type: str | None,
        duration_ms: float,
    ) -> TraceEvent:
        """记录一次工具执行。"""
        return self.log(
            step,
            "tool_exec",
            {
                "path": path,
                "tool_id": tool_id,
                "arguments": arguments,
                "ok": ok,
                "error_type": error_type,
                "duration_ms": duration_ms,
            },
        )

    def log_compress(
        self,
        step: int,
        *,
        level: int,
        before_tokens: int,
        after_tokens: int,
        summary_chars: int,
    ) -> TraceEvent:
        """记录一次压缩。"""
        return self.log(
            step,
            "compress",
            {
                "level": level,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "summary_chars": summary_chars,
            },
        )

    def log_inject(
        self, step: int, *, channel: Literal["chat", "control"], content_chars: int
    ) -> TraceEvent:
        """记录一次外部注入。"""
        return self.log(step, "inject", {"channel": channel, "content_chars": content_chars})

    def log_error(
        self,
        step: int,
        *,
        error_type: str,
        message: str,
        retry_count: int,
    ) -> TraceEvent:
        """记录一次错误。"""
        return self.log(
            step,
            "error",
            {"error_type": error_type, "message": message, "retry_count": retry_count},
        )

    def log_abort(self, step: int, *, reason: str, coalesced: bool) -> TraceEvent:
        """记录一次中断。"""
        return self.log(step, "abort", {"reason": reason, "coalesced": coalesced})

    def log_memory_extract(
        self, step: int, *, status: str, entries_added: int, duration_ms: float
    ) -> TraceEvent:
        """记录一次结算提取结果。"""
        return self.log(
            step,
            "memory_extract",
            {"status": status, "entries_added": entries_added, "duration_ms": duration_ms},
        )

    # --------------------------------------------------------------- 回读

    def read_events(self) -> list[TraceEvent]:
        """回读本会话已落盘的全部事件（每行必须可解析，E2）。"""
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [TraceEvent.from_json_line(line) for line in lines if line.strip()]
