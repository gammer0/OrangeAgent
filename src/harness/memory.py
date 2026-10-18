"""Memory layer (ADT-5, group C).

See docs/02-data-structures/05-memory.md.

Design notes:

* Memory is stored as **files**, not in-memory objects.
* Working memory ``<workdir>/.agent/WORK.md`` is free text; the system never
  parses its structure (C1).
* Long-term memory ``<global>/.agent/MEMORY.md`` is system-wide unique and
  append-merge only (C7).
* Every write goes through "temp file + ``os.replace``" for atomicity (C4).
* This module only provides **pure functions/classes safe to call from a
  background thread**; it performs no scheduling. Running in the background is
  the caller's responsibility (C5's "failure does not affect the main loop" is
  isolated by the caller).
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
    "ensure_working_memory",
    "WORK_MEMORY_TEMPLATE",
    "MemoryEntry",
    "ProjectMemory",
    "MemoryExtraction",
    "MemoryTransitionError",
    "MemoryExtractionStatus",
    "extraction_transition",
]

# ------------------------------------------------------------------------ paths

#: Directory name: ``<workdir>/.agent/`` and ``<global>/.agent/``
AGENT_DIR_NAME = ".agent"
#: Working-memory file name (one per working directory, shared across sessions, C4)
WORK_MEMORY_FILENAME = "WORK.md"
#: Long-term memory file name (system-wide unique, C7)
LONG_TERM_MEMORY_FILENAME = "MEMORY.md"
#: Extraction-principle file name (system-wide unique, C7)
EXTRACT_PRINCIPLE_FILENAME = "extract.md"


def work_memory_path(workdir: str | os.PathLike[str]) -> Path:
    """Return the working-memory path ``<workdir>/.agent/WORK.md`` (C1/C4)."""
    return Path(workdir) / AGENT_DIR_NAME / WORK_MEMORY_FILENAME


def long_term_memory_path(global_dir: str | os.PathLike[str]) -> Path:
    """Return the long-term memory path ``<global>/.agent/MEMORY.md`` (C7).

    ``global_dir`` is the global directory; this path does not vary with the
    working directory (C7).
    """
    return Path(global_dir) / AGENT_DIR_NAME / LONG_TERM_MEMORY_FILENAME


def extract_principle_path(global_dir: str | os.PathLike[str]) -> Path:
    """Return the extraction-principle path ``<global>/.agent/extract.md`` (C7)."""
    return Path(global_dir) / AGENT_DIR_NAME / EXTRACT_PRINCIPLE_FILENAME


# ------------------------------------------------------------- atomic write (C4)


#: Retry count and interval when replacement fails (on Windows a concurrent
#: reader holding a handle raises WinError 5)
_REPLACE_RETRIES = 40
_REPLACE_BACKOFF_S = 0.002


def _replace_with_retry(src: Path, dst: Path) -> None:
    """``os.replace`` with a brief retry on Windows sharing violations.

    On POSIX ``replace`` is never blocked by readers and usually succeeds on the
    first try; on Windows another thread holding a handle to the target raises
    ``PermissionError`` (WinError 5/32). We retry a bounded number of times and
    re-raise if it still fails (never masking the real error).
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
    """Write text atomically: temp file in the same directory + flush + fsync + ``os.replace``.

    ``os.replace`` is atomic within one filesystem, so concurrent readers see
    either "the old complete content" or "the new complete content", never a
    half-written file (C4). The temp file is cleaned up on failure and never
    pollutes the directory.
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
    """Read text; return ``None`` when the file does not exist.

    On Windows a read racing with another ``os.replace`` can raise
    ``PermissionError`` (WinError 13/5); a brief retry is enough.
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


# ------------------------------------------------------- working memory (C1/C4)


def read_working_memory(workdir: str | os.PathLike[str]) -> str:
    """Read the raw text of ``<workdir>/.agent/WORK.md``.

    A missing file returns the empty string: working memory has no structure, so
    absent means empty (C1). Nothing is parsed, trimmed or re-wrapped.
    """
    text = _read_text_with_retry(work_memory_path(workdir))
    return "" if text is None else text


def write_working_memory_atomic(workdir: str | os.PathLike[str], text: str) -> Path:
    """Atomically overwrite ``<workdir>/.agent/WORK.md`` with the exact text (C1/C4).

    Returns:
        The path actually written.
    """
    path = work_memory_path(workdir)
    _atomic_write_text(path, text)
    return path


#: Initial content when creating working memory.
#:
#: Deliberately minimal: just a skeleton, no imposed structure. Working memory
#: is free text (C1) that the system never parses, so the template must not imply
#: any format constraints.
#:
#: Note: this text is model-visible, so it intentionally stays in Chinese.
WORK_MEMORY_TEMPLATE = """# 工作记忆

> 本文件由 Harness 自动创建，同工作目录的多个会话共享。
> 用 /tool/memory/work/{read,append,replace} 读写；系统不解析其结构。

## 当前状态

（尚未记录）

## 关键决定

（尚未记录）

## 待办与风险

（尚未记录）
"""


def ensure_working_memory(
    workdir: str | os.PathLike[str],
    *,
    template: str | None = None,
) -> tuple[Path, bool]:
    """Ensure the working-memory file exists, creating it from the template if absent.

    The system prompt tells the model "this project has ``.agent/WORK.md``" but
    nothing ever created it -- the model's first read returned "file not found",
    contradicting the prompt. This function supplies that precondition during
    service assembly.

    **Creates only when missing**: never overwrites existing content. Working
    memory is shared across sessions, so overwriting would erase conclusions
    another session just wrote.

    Args:
        workdir: Working directory.
        template: Initial content; defaults to :data:`WORK_MEMORY_TEMPLATE`.

    Returns:
        ``(path, created)``. Returns ``(path, False)`` when it already exists.

    Raises:
        ValueError: ``workdir`` is not a valid directory.
        OSError: Creation failed (permissions, ...); the caller decides whether
            to degrade.
    """
    base = Path(workdir)
    if not base.is_dir():
        raise ValueError(f"working directory does not exist or is not a directory: {base}")

    path = work_memory_path(base)
    if path.is_file():
        return path, False

    # Use the atomic path: share the temp-file + replace flow with normal writes
    # so concurrent creation never leaves a half-written file.
    _atomic_write_text(path, template if template is not None else WORK_MEMORY_TEMPLATE)
    return path, True


# --------------------------------------------------------- long-term memory (C7)

MemoryText = Annotated[str, Field(min_length=1)]


class MemoryEntry(BaseModel):
    """One **cross-project** conclusion (C8).

    ``source_workdir`` is for provenance only and is never injected into model
    context (C6).
    """

    model_config = ConfigDict(extra="forbid")

    text: MemoryText
    source_workdir: str = ""
    updated_at: float = 0.0

    @model_validator(mode="after")
    def _normalise(self) -> MemoryEntry:
        # Trim before deciding merge semantics, so "x" and " x " never accumulate
        object.__setattr__(self, "text", self.text.strip())
        if not self.text:
            raise ValueError("C8: MemoryEntry.text must not be empty")
        return self


class ProjectMemory(BaseModel):
    """Long-term memory container: exactly one system-wide (C7).

    Uniqueness is expressed by the path constant: every working directory shares
    the same ``MEMORY.md`` returned by :func:`long_term_memory_path`, and this
    class manages no multiple instances.

    Write semantics are **append-merge only**: ``merge`` never deletes existing
    entries and never overwrites at the end; identical ``text`` counts as the
    same entry (no duplicate append).
    """

    model_config = ConfigDict(extra="forbid")

    entries: list[MemoryEntry] = Field(default_factory=list)
    updated_at: float = 0.0

    # ---------------------------------------------------------- append-merge

    def merge(self, new_entries: list[MemoryEntry], now: float) -> int:
        """Merge ``new_entries`` into ``entries``; returns the number actually added.

        Entries with identical ``text`` count as already present: no duplicate
        append and no modification of the existing entry (preserving the first
        ``source_workdir`` / ``updated_at``).
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

    # --------------------------------------------------------------- storage

    @staticmethod
    def load(path: str | os.PathLike[str]) -> ProjectMemory:
        """Load long-term memory from ``MEMORY.md``; a missing file yields an empty container."""
        file = Path(path)
        raw = _read_text_with_retry(file)
        if raw is None or not raw.strip():
            return ProjectMemory()
        return ProjectMemory.model_validate_json(raw)

    def save_atomic(self, path: str | os.PathLike[str]) -> Path:
        """Atomically write back ``MEMORY.md`` (temp file + ``os.replace``)."""
        file = Path(path)
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2)
        _atomic_write_text(file, payload + "\n")
        return file


# ------------------------------------------------------- settlement extraction (C5)

MemoryExtractionStatus = Literal["pending", "running", "done", "failed"]

#: Legal transitions: pending -> running -> done / failed
_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"done", "failed"}),
    "done": frozenset(),
    "failed": frozenset(),
}


class MemoryTransitionError(HarnessError):
    """An illegal ``MemoryExtraction`` state transition."""

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(f"illegal extraction state transition: {source} -> {target}")


def extraction_transition(
    extraction: MemoryExtraction, target: MemoryExtractionStatus
) -> MemoryExtraction:
    """Perform one state transition and return the new object; illegal transitions raise."""
    if target not in _LEGAL_TRANSITIONS[extraction.status]:
        raise MemoryTransitionError(extraction.status, target)
    return extraction.model_copy(update={"status": target})


class MemoryExtraction(BaseModel):
    """A settlement-extraction task (a background task the user triggers explicitly; never blocks the loop, C5).

    State machine: ``pending -> running -> done | failed``; ``done`` and
    ``failed`` are terminal and cannot transition further.
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str
    principle_path: str
    target_path: str
    status: MemoryExtractionStatus = "pending"

    @property
    def is_terminal(self) -> bool:
        """Whether the state is terminal."""
        return self.status in ("done", "failed")

    def can_transition_to(self, target: MemoryExtractionStatus) -> bool:
        """Whether the transition is legal."""
        return target in _LEGAL_TRANSITIONS[self.status]

    def transition_to(self, target: MemoryExtractionStatus) -> MemoryExtraction:
        """Transition to ``target``; illegal transitions raise :class:`MemoryTransitionError`.

        Returns:
            The new instance (the original is not modified).
        """
        return extraction_transition(self, target)


#: Signature of the conversion function for extraction tasks (for background
#: thread callbacks)
ExtractionRunner = Callable[[MemoryExtraction], MemoryExtraction]
