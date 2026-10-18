"""工作记忆专用工具的测试。

这些工具只操作 ``.agent/WORK.md``，不接受路径参数——这是它们的核心约束：
让模型能沉淀记忆，同时不破坏「文件系统只读」的沙箱。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.memory import WORK_MEMORY_TEMPLATE, read_working_memory, work_memory_path
from harness.tools_builtin import (
    TOOL_SPECS,
    build_tool,
    memory_work_append,
    memory_work_read,
    memory_work_replace,
)

PATHS = ("/tool/memory/work/read", "/tool/memory/work/append", "/tool/memory/work/replace")


class TestRegistration:
    @pytest.mark.parametrize("path", PATHS)
    def test_registered(self, path: str) -> None:
        assert path in TOOL_SPECS

    def test_read_is_read_cap(self) -> None:
        assert TOOL_SPECS["/tool/memory/work/read"]["caps"] == ["read"]

    @pytest.mark.parametrize("path", ["/tool/memory/work/append", "/tool/memory/work/replace"])
    def test_write_tools_are_write_cap(self, path: str) -> None:
        assert TOOL_SPECS[path]["caps"] == ["write"]

    @pytest.mark.parametrize("path", PATHS)
    def test_schema_has_no_path_param(self, path: str) -> None:
        """核心约束：模型不能指定文件路径。"""
        props = TOOL_SPECS[path]["schema"]["properties"]
        assert "path" not in props
        assert "workdir" not in props

    @pytest.mark.parametrize("path", PATHS)
    def test_built_tool_rejects_extra_path(self, path: str, tmp_path: Path) -> None:
        entry = build_tool(path, str(tmp_path))
        if path.endswith("read"):
            entry.fn()
            return
        with pytest.raises(TypeError):
            entry.fn(path="../escape.md", content="x")


class TestRead:
    def test_missing_file_message(self, tmp_path: Path) -> None:
        out = memory_work_read(str(tmp_path))
        assert "尚不存在" in out

    def test_reads_content(self, tmp_path: Path) -> None:
        target = work_memory_path(tmp_path)
        target.parent.mkdir(parents=True)
        target.write_text("记忆内容", encoding="utf-8")
        assert "记忆内容" in memory_work_read(str(tmp_path))


class TestAppend:
    def test_creates_file(self, tmp_path: Path) -> None:
        memory_work_append(str(tmp_path), "第一条")
        assert "第一条" in read_working_memory(tmp_path)

    def test_does_not_overwrite(self, tmp_path: Path) -> None:
        memory_work_append(str(tmp_path), "第一条")
        memory_work_append(str(tmp_path), "第二条")
        body = read_working_memory(tmp_path)
        assert "第一条" in body and "第二条" in body

    def test_adds_trailing_newline(self, tmp_path: Path) -> None:
        memory_work_append(str(tmp_path), "无换行结尾")
        memory_work_append(str(tmp_path), "下一行")
        assert "无换行结尾\n下一行\n" in read_working_memory(tmp_path)

    def test_keeps_existing_newline(self, tmp_path: Path) -> None:
        memory_work_append(str(tmp_path), "带换行\n")
        assert read_working_memory(tmp_path) == "带换行\n"

    def test_rejects_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            memory_work_append(str(tmp_path), "   \n  ")

    def test_creates_agent_dir(self, tmp_path: Path) -> None:
        assert not (tmp_path / ".agent").exists()
        memory_work_append(str(tmp_path), "x")
        assert (tmp_path / ".agent").is_dir()

    def test_only_touches_work_memory(self, tmp_path: Path) -> None:
        (tmp_path / "other.md").write_text("别人的文件", encoding="utf-8")
        memory_work_append(str(tmp_path), "记忆")
        assert (tmp_path / "other.md").read_text(encoding="utf-8") == "别人的文件"


class TestReplace:
    def test_replaces_all(self, tmp_path: Path) -> None:
        target = work_memory_path(tmp_path)
        target.parent.mkdir(parents=True)
        target.write_text("旧内容", encoding="utf-8")

        memory_work_replace(str(tmp_path), "# 新内容")
        body = read_working_memory(tmp_path)
        assert "旧内容" not in body
        assert "# 新内容" in body

    def test_ensures_trailing_newline(self, tmp_path: Path) -> None:
        memory_work_replace(str(tmp_path), "没有换行")
        assert read_working_memory(tmp_path) == "没有换行\n"


class TestSystemPrompt:
    def test_mentions_dedicated_tools(self) -> None:
        """系统提示必须指向真实存在的工具，否则模型会去找不存在的路径。"""
        from harness.context import DEFAULT_SYSTEM_REQUIREMENTS

        for path in PATHS:
            assert path in DEFAULT_SYSTEM_REQUIREMENTS, f"提示未提及 {path}"

    def test_mentioned_tools_are_registered(self) -> None:
        from harness.context import DEFAULT_SYSTEM_REQUIREMENTS

        for path in PATHS:
            assert path in TOOL_SPECS
            assert path in DEFAULT_SYSTEM_REQUIREMENTS

    def test_template_exists(self) -> None:
        assert WORK_MEMORY_TEMPLATE.strip()
