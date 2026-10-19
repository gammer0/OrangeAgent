"""记忆层测试（C 组：C1/C2/C3/C4/C6/C7/C8）。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.memory import (
    LONG_TERM_MEMORY_FILENAME,
    WORK_MEMORY_FILENAME,
    MemoryEntry,
    MemoryExtraction,
    MemoryTransitionError,
    ProjectMemory,
    extract_principle_path,
    long_term_memory_path,
    read_working_memory,
    work_memory_path,
    write_working_memory_atomic,
)
from harness.paths import harness_home

# ------------------------------------------------------------------ 工作记忆


class TestWorkingMemory:
    def test_c1_work_memory_path_is_project_level(self, tmp_path: Path) -> None:
        """工作记忆是项目级的，直接位于工作目录根部。"""
        assert work_memory_path(tmp_path) == tmp_path / "WORK.md"
        assert WORK_MEMORY_FILENAME == "WORK.md"
        # 不在子目录里，也与全局数据目录无关
        assert work_memory_path(tmp_path).parent == tmp_path
        assert work_memory_path(tmp_path).parent != harness_home()

    def test_c1_freeform_text_roundtrip_verbatim(self, tmp_path: Path) -> None:
        text = "# 标题\n\n- 条目一\n  - 缩进\n\n任意格式 **不解析**\n"
        write_working_memory_atomic(tmp_path, text)
        assert read_working_memory(tmp_path) == text

    def test_c1_missing_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert not work_memory_path(tmp_path).exists()
        assert read_working_memory(tmp_path) == ""

    def test_c4_write_creates_only_the_file(self, tmp_path: Path) -> None:
        path = write_working_memory_atomic(tmp_path, "x")
        assert path == tmp_path / "WORK.md"
        assert path.exists()
        # 不得创建 .agent 子目录
        assert not (tmp_path / ".agent").exists()

    def test_c4_atomic_write_keeps_original_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """写入过程中失败时原文件保持完整（tempfile + os.replace）。"""
        original = "原始内容\n" * 100
        write_working_memory_atomic(tmp_path, original)

        def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            raise OSError("模拟 rename 失败")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            write_working_memory_atomic(tmp_path, "新内容")

        assert read_working_memory(tmp_path) == original
        # 临时文件被清理，目录内只剩 WORK.md
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "WORK.md"]
        assert leftovers == []

    def test_c4_no_partial_file_visible_during_write(self, tmp_path: Path) -> None:
        """写入新内容时不会出现半截文件：replace 前旧内容始终可读。"""
        write_working_memory_atomic(tmp_path, "old")
        seen: set[str] = set()
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                seen.add(read_working_memory(tmp_path))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for i in range(50):
                write_working_memory_atomic(tmp_path, f"new-{i}" * 200)
        finally:
            stop.set()
            thread.join()

        assert {"old"} <= seen
        assert seen <= {"old"} | {f"new-{i}" * 200 for i in range(50)}


# ---------------------------------------------------------------- 长期记忆


class TestProjectMemory:
    def test_c7_paths_are_global_constants(self, tmp_path: Path) -> None:
        a = long_term_memory_path(tmp_path)
        b = long_term_memory_path(tmp_path)
        assert a == b == harness_home() / "MEMORY.md"
        assert LONG_TERM_MEMORY_FILENAME == "MEMORY.md"
        assert extract_principle_path(tmp_path) == harness_home() / "extract.md"
        # 不随工作目录变化（C7）
        assert long_term_memory_path(Path("/elsewhere")) == a

    def test_append_only_merge(self) -> None:
        mem = ProjectMemory()
        mem.merge([MemoryEntry(text="原则一", source_workdir="/a", updated_at=1.0)], now=1.0)
        mem.merge([MemoryEntry(text="原则二", source_workdir="/b", updated_at=2.0)], now=2.0)
        assert [e.text for e in mem.entries] == ["原则一", "原则二"]
        assert mem.updated_at == 2.0

    def test_identical_text_not_duplicated(self) -> None:
        mem = ProjectMemory()
        mem.merge([MemoryEntry(text="同一结论", source_workdir="/a", updated_at=1.0)], now=1.0)
        added = mem.merge(
            [MemoryEntry(text="同一结论", source_workdir="/b", updated_at=9.0)], now=9.0
        )
        assert added == 0
        assert len(mem.entries) == 1
        # 首次来源保留，不被后来的项目覆盖（C6 溯源）
        assert mem.entries[0].source_workdir == "/a"
        assert mem.updated_at == 1.0

    def test_multiple_workdirs_share_one_entries_set_c7(self, tmp_path: Path) -> None:
        path = long_term_memory_path(tmp_path)
        mem = ProjectMemory.load(path)
        mem.merge([MemoryEntry(text="A 项目结论", source_workdir="/a", updated_at=1.0)], now=1.0)
        mem.save_atomic(path)

        # 另一个工作目录结算，读写的仍是同一份文件（C7）
        mem2 = ProjectMemory.load(long_term_memory_path(tmp_path))
        mem2.merge([MemoryEntry(text="B 项目结论", source_workdir="/b", updated_at=2.0)], now=2.0)
        mem2.save_atomic(long_term_memory_path(tmp_path))

        final = ProjectMemory.load(path)
        assert [e.text for e in final.entries] == ["A 项目结论", "B 项目结论"]

    def test_save_load_roundtrip_and_atomic(self, tmp_path: Path) -> None:
        path = long_term_memory_path(tmp_path)
        mem = ProjectMemory()
        mem.merge([MemoryEntry(text="中文结论", source_workdir="/a", updated_at=1.0)], now=1.0)
        mem.save_atomic(path)
        assert json.loads(path.read_text(encoding="utf-8"))["entries"][0]["text"] == "中文结论"
        assert ProjectMemory.load(path).entries == mem.entries

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert ProjectMemory.load(tmp_path / "nope.md").entries == []

    def test_c8_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryEntry(text="   ")

    def test_text_is_stripped_for_merge_semantics(self) -> None:
        mem = ProjectMemory()
        mem.merge([MemoryEntry(text=" 结论 ")], now=1.0)
        assert mem.merge([MemoryEntry(text="结论")], now=2.0) == 0

    def test_atomic_save_keeps_old_file_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = long_term_memory_path(tmp_path)
        ProjectMemory().save_atomic(path)
        before = path.read_text(encoding="utf-8")

        def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            raise OSError("rename 失败")

        monkeypatch.setattr(os, "replace", boom)
        mem = ProjectMemory()
        mem.merge([MemoryEntry(text="新条目")], now=1.0)
        with pytest.raises(OSError):
            mem.save_atomic(path)
        assert path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------- 结算提取


def _extraction(tmp_path: Path) -> MemoryExtraction:
    return MemoryExtraction(
        source_path=str(tmp_path / ".agent" / "WORK.md"),
        principle_path=str(tmp_path / "global" / ".agent" / "extract.md"),
        target_path=str(tmp_path / "global" / ".agent" / "MEMORY.md"),
    )


class TestMemoryExtraction:
    def test_initial_status_is_pending(self, tmp_path: Path) -> None:
        assert _extraction(tmp_path).status == "pending"

    def test_legal_transitions(self, tmp_path: Path) -> None:
        ext = _extraction(tmp_path)
        running = ext.transition_to("running")
        assert running.status == "running"
        assert ext.status == "pending"  # 原对象不被修改
        for terminal in ("done", "failed"):
            assert running.transition_to(terminal).status == terminal

    @pytest.mark.parametrize("target", ["done", "failed", "pending"])
    def test_illegal_transition_from_pending(self, tmp_path: Path, target: str) -> None:
        with pytest.raises(MemoryTransitionError):
            _extraction(tmp_path).transition_to(target)  # type: ignore[arg-type]

    def test_illegal_transition_from_terminal(self, tmp_path: Path) -> None:
        done = _extraction(tmp_path).transition_to("running").transition_to("done")
        assert done.is_terminal
        with pytest.raises(MemoryTransitionError):
            done.transition_to("running")

    def test_can_transition_to_matches_rules(self, tmp_path: Path) -> None:
        ext = _extraction(tmp_path)
        assert ext.can_transition_to("running")
        assert not ext.can_transition_to("done")

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        payload = _extraction(tmp_path).model_dump()
        payload["status"] = "cancelled"
        with pytest.raises(ValidationError):
            MemoryExtraction.model_validate(payload)

    def test_paths_match_documented_layout(self, tmp_path: Path) -> None:
        ext = _extraction(tmp_path)
        assert Path(ext.source_path) == tmp_path / ".agent" / "WORK.md"
        assert Path(ext.principle_path) == tmp_path / "global" / ".agent" / "extract.md"
        assert Path(ext.target_path) == tmp_path / "global" / ".agent" / "MEMORY.md"
