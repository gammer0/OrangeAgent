"""记忆层（ADT-5，C 组）。

对应文档：docs/02-数据结构设计/05-记忆.md

设计要点：

* 记忆以**文件**为存储单位，不是内存对象。
* 工作记忆 ``<workdir>/.agent/WORK.md`` 为自由文本，系统不解析其结构（C1）。
* 长期记忆 ``<global>/.agent/MEMORY.md`` 全系统唯一，只追加合并（C7）。
* 所有写入走「临时文件 + ``os.replace``」保证原子性（C4）。
* 本模块只提供**可在后台线程调用的纯函数/类**，不含线程调度：
  后台运行由调用方负责（C5 的「失败不影响主循环」由调用方隔离）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.exceptions import HarnessError

__all__ = [
    "AGENT_DIR_NAME",
    "WORK_MEMORY_FILENAME",
    "LONG_TERM_MEMORY_FILENAME",
    "EXTRACT_PRINCIPLE_FILENAME",
    "work_memory_path",
    "long_term_memory_path",
    "extract_principle_path",
    "read_working_memory",
    "write_working_memory_atomic",
    "MemoryEntry",
    "ProjectMemory",
    "MemoryExtraction",
    "MemoryTransitionError",
    "MemoryExtractionStatus",
    "extraction_transition",
]

# --------------------------------------------------------------------------- 路径

#: 目录名：``<workdir>/.agent/`` 与 ``<global>/.agent/``
AGENT_DIR_NAME = ".agent"
#: 工作记忆文件名（每个工作目录一份，多会话共享，C4）
WORK_MEMORY_FILENAME = "WORK.md"
#: 长期记忆文件名（全系统唯一，C7）
LONG_TERM_MEMORY_FILENAME = "MEMORY.md"
#: 提取原则文件名（全系统唯一，C7）
EXTRACT_PRINCIPLE_FILENAME = "extract.md"


def work_memory_path(workdir: str | os.PathLike[str]) -> Path:
    """返回工作记忆路径 ``<workdir>/.agent/WORK.md``（C1/C4）。"""
    return Path(workdir) / AGENT_DIR_NAME / WORK_MEMORY_FILENAME


def long_term_memory_path(global_dir: str | os.PathLike[str]) -> Path:
    """返回长期记忆路径 ``<global>/.agent/MEMORY.md``（C7）。

    ``global_dir`` 即全局目录；该路径不随工作目录变化（C7）。
    """
    return Path(global_dir) / AGENT_DIR_NAME / LONG_TERM_MEMORY_FILENAME


def extract_principle_path(global_dir: str | os.PathLike[str]) -> Path:
    """返回提取原则路径 ``<global>/.agent/extract.md``（C7）。"""
    return Path(global_dir) / AGENT_DIR_NAME / EXTRACT_PRINCIPLE_FILENAME


# ----------------------------------------------------------------- 原子写（C4）


#: 替换失败时的重试次数与间隔（Windows 上并发读者持有句柄会报 WinError 5）
_REPLACE_RETRIES = 40
_REPLACE_BACKOFF_S = 0.002


def _replace_with_retry(src: Path, dst: Path) -> None:
    """``os.replace``，遇 Windows 共享冲突时短暂重试。

    POSIX 上 ``replace`` 不会被读者阻塞，通常一次成功；Windows 上若另一
    线程正持有目标文件句柄，会抛 ``PermissionError``（WinError 5/32）。
    这里做有限重试，仍然失败则原样抛出（不掩盖真实错误）。
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写入文本：同目录临时文件 + flush + fsync + ``os.replace``。

    ``os.replace`` 在同一文件系统上是原子的，因此并发读者只会看到
    「旧完整内容」或「新完整内容」，不会看到半截文件（C4）。
    临时文件失败时会被清理，不污染目录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_text_with_retry(path: Path) -> str | None:
    """读取文本；文件不存在返回 ``None``。

    Windows 上读取与别人 ``os.replace`` 撞车时可能抛 ``PermissionError``
    （WinError 13/5），短暂重试即可。
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)
    return None


# ----------------------------------------------------------- 工作记忆（C1/C4）


def read_working_memory(workdir: str | os.PathLike[str]) -> str:
    """读取 ``<workdir>/.agent/WORK.md`` 的原文。

    文件不存在时返回空串：工作记忆没有结构，缺失即「空」（C1）。
    不做任何解析、不裁剪、不改写换行。
    """
    text = _read_text_with_retry(work_memory_path(workdir))
    return "" if text is None else text


def write_working_memory_atomic(workdir: str | os.PathLike[str], text: str) -> Path:
    """原子覆盖写入 ``<workdir>/.agent/WORK.md``，内容原样落盘（C1/C4）。

    Returns:
        实际写入的路径。
    """
    path = work_memory_path(workdir)
    _atomic_write_text(path, text)
    return path


# --------------------------------------------------------------- 长期记忆（C7）

MemoryText = Annotated[str, Field(min_length=1)]


class MemoryEntry(BaseModel):
    """一条**跨项目通用**结论（C8）。

    ``source_workdir`` 仅用于溯源，不注入模型上下文（C6）。
    """

    model_config = ConfigDict(extra="forbid")

    text: MemoryText
    source_workdir: str = ""
    updated_at: float = 0.0

    @model_validator(mode="after")
    def _normalise(self) -> MemoryEntry:
        # 去掉首尾空白后再判定合并语义，避免 "x" 与 " x " 重复累积
        object.__setattr__(self, "text", self.text.strip())
        if not self.text:
            raise ValueError("C8: MemoryEntry.text 不得为空")
        return self


class ProjectMemory(BaseModel):
    """长期记忆容器：全系统唯一一份（C7）。

    唯一性由路径常量表达：所有工作目录共用
    :func:`long_term_memory_path` 返回的同一个 ``MEMORY.md``，
    本类不做多实例管理。

    写入语义是**只追加合并**：``merge`` 不删除已有条目，也不会
    在末尾覆盖；相同 ``text`` 视为同一条（不重复追加）。
    """

    model_config = ConfigDict(extra="forbid")

    entries: list[MemoryEntry] = Field(default_factory=list)
    updated_at: float = 0.0

    # -------------------------------------------------------------- 追加合并

    def merge(self, new_entries: list[MemoryEntry], now: float) -> int:
        """把 ``new_entries`` 合并进 ``entries``，返回实际新增条数。

        相同 ``text`` 的条目视为已存在：不重复追加，也不改动原有条目
        （保持 ``source_workdir`` / ``updated_at`` 的首次来源）。
        """
        existing = {entry.text for entry in self.entries}
        added = 0
        for entry in new_entries:
            if entry.text in existing:
                continue
            existing.add(entry.text)
            self.entries.append(entry)
            added += 1
        if added:
            self.updated_at = now
        return added

    # -------------------------------------------------------------- 持久化

    @staticmethod
    def load(path: str | os.PathLike[str]) -> ProjectMemory:
        """从 ``MEMORY.md`` 读取长期记忆；文件不存在时返回空容器。"""
        file = Path(path)
        raw = _read_text_with_retry(file)
        if raw is None or not raw.strip():
            return ProjectMemory()
        return ProjectMemory.model_validate_json(raw)

    def save_atomic(self, path: str | os.PathLike[str]) -> Path:
        """原子写回 ``MEMORY.md``（临时文件 + ``os.replace``）。"""
        file = Path(path)
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2)
        _atomic_write_text(file, payload + "\n")
        return file


# ------------------------------------------------------------ 结算提取（C5）

MemoryExtractionStatus = Literal["pending", "running", "done", "failed"]

#: 合法状态迁移：pending → running → done / failed
_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"done", "failed"}),
    "done": frozenset(),
    "failed": frozenset(),
}


class MemoryTransitionError(HarnessError):
    """``MemoryExtraction`` 非法状态迁移。"""

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(f"非法的提取状态迁移: {source} -> {target}")


def extraction_transition(
    extraction: MemoryExtraction, target: MemoryExtractionStatus
) -> MemoryExtraction:
    """执行一次状态迁移并返回新对象；非法迁移抛 :class:`MemoryTransitionError`。"""
    if target not in _LEGAL_TRANSITIONS[extraction.status]:
        raise MemoryTransitionError(extraction.status, target)
    return extraction.model_copy(update={"status": target})


class MemoryExtraction(BaseModel):
    """结算提取任务（用户显式触发的后台任务，不阻塞主循环，C5）。

    状态机：``pending → running → done | failed``；
    ``done`` / ``failed`` 为终态，不可再迁移。
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str
    principle_path: str
    target_path: str
    status: MemoryExtractionStatus = "pending"

    @property
    def is_terminal(self) -> bool:
        """是否已到终态。"""
        return self.status in ("done", "failed")

    def can_transition_to(self, target: MemoryExtractionStatus) -> bool:
        """判断迁移是否合法。"""
        return target in _LEGAL_TRANSITIONS[self.status]

    def transition_to(self, target: MemoryExtractionStatus) -> MemoryExtraction:
        """迁移到 ``target``，非法迁移抛 :class:`MemoryTransitionError`。

        Returns:
            迁移后的新实例（原实例不被修改）。
        """
        return extraction_transition(self, target)


#: 解析提取任务时的转换函数签名（供后台线程回调使用）
ExtractionRunner = Callable[[MemoryExtraction], MemoryExtraction]
