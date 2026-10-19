"""工作记忆文件自动创建的测试。

关键约束：**只在缺失时创建，绝不覆盖已有内容**——工作记忆跨会话共享，
覆盖等于抹掉别的会话刚写下的结论。

另：数据目录不在用户工作目录内，见 ``orange.paths``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orange.memory import (
    WORK_MEMORY_TEMPLATE,
    ensure_working_memory,
    read_working_memory,
    work_memory_path,
)
from orange.service import build_service


class TestEnsureWorkingMemory:
    def test_creates_when_missing(self, tmp_path: Path) -> None:
        path, created = ensure_working_memory(tmp_path)
        assert created is True
        assert path == work_memory_path(tmp_path)
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == WORK_MEMORY_TEMPLATE

    def test_creates_work_md_without_agent_dir(self, tmp_path: Path) -> None:
        """直接在工作目录根部创建 WORK.md，不创建 .agent 子目录。"""
        assert not (tmp_path / "WORK.md").exists()
        ensure_working_memory(tmp_path)
        assert (tmp_path / "WORK.md").is_file()
        assert not (tmp_path / ".agent").exists()

    def test_keeps_existing_content(self, tmp_path: Path) -> None:
        """已有内容必须原样保留。"""
        target = work_memory_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("我自己写的内容\n第二行", encoding="utf-8")

        path, created = ensure_working_memory(tmp_path)
        assert created is False
        assert path.read_text(encoding="utf-8") == "我自己写的内容\n第二行"

    def test_keeps_real_looking_memory(self, tmp_path: Path) -> None:
        """真实工作记忆形态（多级标题 + 列表）也不得被模板覆盖。"""
        target = work_memory_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        original = (
            "# 工作记忆\n\n"
            "## 最终结构\n\n"
            "- `config.py`：常量，无逻辑无副作用。\n"
            "- `game.py`：纯逻辑层，可被无头工具直接调用。\n\n"
            "## 验收命令\n\n"
            "- `python -m snake_game.main --headless 200 --seed 7`\n"
        )
        target.write_text(original, encoding="utf-8")

        ensure_working_memory(tmp_path)
        assert target.read_text(encoding="utf-8") == original
        # 模板特征不应出现
        assert "尚未记录" not in target.read_text(encoding="utf-8")

    def test_idempotent(self, tmp_path: Path) -> None:
        ensure_working_memory(tmp_path)
        _, created = ensure_working_memory(tmp_path)
        assert created is False

    def test_custom_template(self, tmp_path: Path) -> None:
        ensure_working_memory(tmp_path, template="# 自定义\n")
        assert work_memory_path(tmp_path).read_text(encoding="utf-8") == "# 自定义\n"

    def test_rejects_missing_workdir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            ensure_working_memory(tmp_path / "nope")

    def test_readable_after_create(self, tmp_path: Path) -> None:
        ensure_working_memory(tmp_path)
        assert "工作记忆" in read_working_memory(tmp_path)

    def test_template_has_no_format_placeholders(self) -> None:
        """模板不该含 ``{name}`` / ``%s`` 这类格式化占位符。

        注意与 ``{read,append,replace}`` 这种花括号列举区分：后者是提示文字，
        不含冒号，不会被 ``str.format`` 吃掉。
        """
        import re

        # {identifier} 或 {identifier:spec} 才算格式化占位
        assert not re.search(r"\{\w+(?:[:!][^}]*)?\}", WORK_MEMORY_TEMPLATE)
        assert "%s" not in WORK_MEMORY_TEMPLATE

    def test_template_is_single_trailing_newline(self) -> None:
        assert WORK_MEMORY_TEMPLATE.endswith("\n")
        assert not WORK_MEMORY_TEMPLATE.endswith("\n\n")


class TestServiceIntegration:
    def test_service_creates_on_startup(self, tmp_path: Path) -> None:
        assert not work_memory_path(tmp_path).exists()
        service = build_service(tmp_path, load_model=False)
        try:
            assert service.work_memory_created is True
            assert service.work_memory == work_memory_path(tmp_path)
            assert (tmp_path / "WORK.md").is_file()
            # 不建 .agent，且全局数据（数据库等）也不落在项目里
            assert not (tmp_path / ".agent").exists()
            assert not (tmp_path / "orange.db").exists()
        finally:
            service.close()

    def test_service_preserves_existing(self, tmp_path: Path) -> None:
        target = work_memory_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("已有结论：别动我", encoding="utf-8")

        service = build_service(tmp_path, load_model=False)
        try:
            assert service.work_memory_created is False
            assert target.read_text(encoding="utf-8") == "已有结论：别动我"
        finally:
            service.close()

    def test_restart_does_not_reset(self, tmp_path: Path) -> None:
        """两次启动之间写的内容必须保留（模拟重启）。"""
        first = build_service(tmp_path, load_model=False)
        first.close()
        work_memory_path(tmp_path).write_text("会话 A 的记录", encoding="utf-8")

        second = build_service(tmp_path, load_model=False)
        try:
            assert second.work_memory_created is False
            assert work_memory_path(tmp_path).read_text(encoding="utf-8") == "会话 A 的记录"
        finally:
            second.close()

    def test_failure_does_not_block_startup(self, tmp_path: Path, monkeypatch) -> None:
        """工作记忆创建失败时应降级，服务仍可启动。"""

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("模拟权限错误")

        monkeypatch.setattr("orange.service.ensure_working_memory", boom)
        service = build_service(tmp_path, load_model=False)
        try:
            assert service.work_memory is None
            assert any("work_memory" in p for p in service.problems)
            # 其他层仍正常
            assert service.tree is not None
        finally:
            service.close()
