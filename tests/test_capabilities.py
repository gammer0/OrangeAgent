"""能力清单测试：内置工具、skill 落盘、清单往返与非法输入拒绝。"""

from __future__ import annotations

import json

import pytest

from harness.capabilities import (
    CapabilityError,
    CapabilityManifest,
    SkillDef,
    build_builtin_tool,
    load_manifest,
    manifest_path,
    save_manifest,
    write_skill_files,
)
from harness.tools_builtin import TOOL_SPECS, build_tool, known_paths


# ---------------------------------------------------------------------------
# 内置工具
# ---------------------------------------------------------------------------


def test_all_builtin_tools_build(tmp_path):
    """每个规格都能构造出合法 ToolDef（B11/B12/B17 不报错）。"""
    for path in known_paths():
        entry = build_tool(path, str(tmp_path))
        assert entry.path == path
        assert entry.fn is not None
        assert entry.description
    assert len(known_paths()) == len(TOOL_SPECS)


def test_write_tool_is_not_idempotent(tmp_path):
    """写类工具不得被标成幂等（B12）。"""
    entry = build_tool("/tool/os/file/write", str(tmp_path))
    assert entry.idempotent is False
    assert entry.concurrency == "exclusive"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/tool/os/file/read", "safe"),
        ("/tool/code/py/run_brief", "exclusive"),
        ("/mcp/fs/write", "exclusive"),
    ],
)
def test_concurrency_derived_from_caps(path, expected, tmp_path):
    """并发性由 caps 推导。"""
    assert build_tool(path, str(tmp_path)).concurrency == expected


def test_file_tools_stay_inside_workdir(tmp_path):
    """路径越界被拒绝。"""
    tool = build_tool("/tool/os/file/read", str(tmp_path))
    with pytest.raises(ValueError):
        tool.fn(path="../escape.txt")


def test_write_then_read_roundtrip(tmp_path):
    """写入后能按行读回，且行号带前缀。"""
    write = build_tool("/tool/os/file/write", str(tmp_path))
    read = build_tool("/tool/os/file/read", str(tmp_path))
    write.fn(path="a/b.py", content="x = 1\ny = 2\n")
    output = read.fn(path="a/b.py")
    assert "共 2 行" in output
    assert "1| x = 1" in output


def test_py_check_reports_syntax_error(tmp_path):
    """语法检查能定位到行号。"""
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    entry = build_tool("/tool/code/py/check", str(tmp_path))
    assert "语法错误" in entry.fn(path="bad.py")


def test_snake_check_flags_missing_entry(tmp_path):
    """验收工具在缺 main.py 时给出问题清单。"""
    project = tmp_path / "snake_game"
    project.mkdir()
    (project / "game.py").write_text("class SnakeGame: pass\n", encoding="utf-8")
    report = json.loads(build_tool("/tool/game/snake/check", str(tmp_path)).fn())
    assert report["结论"] == "未通过"
    assert any("main.py" in item for item in report["问题"])


_MINIMAL_GAME = """\
from __future__ import annotations


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self._snake = [(5, 5)]
        self._direction = "right"
        self._score = 0
        self._over = False

    def move(self, direction: str) -> None:
        if direction == "left":
            return
        self._direction = direction

    def step(self) -> None:
        dx, dy = {"up": (0, -1), "down": (0, 1), "right": (1, 0)}[self._direction]
        head = self._snake[0]
        head = (head[0] + dx, head[1] + dy)
        if not (0 <= head[0] < self.width and 0 <= head[1] < self.height):
            self._over = True
            return
        self._snake = [head]

    def reset(self) -> None:
        self._snake = [(5, 5)]
        self._over = False

    @property
    def score(self) -> int:
        return self._score

    @property
    def snake(self) -> list[tuple[int, int]]:
        return list(self._snake)

    @property
    def food(self) -> tuple[int, int]:
        return (10, 10)

    @property
    def over(self) -> bool:
        return self._over
"""


def test_snake_demo_handles_package_layout(tmp_path):
    """包布局（无 __init__.py、使用相对导入）也能被探针导入。"""
    project = tmp_path / "snake_game"
    project.mkdir()
    (project / "game.py").write_text(_MINIMAL_GAME, encoding="utf-8")
    output = build_tool("/tool/game/snake/demo", str(tmp_path)).fn(
        path="snake_game", steps=50, seed=1
    )
    assert "退出码 0" in output
    assert '"ok": true' in output
    assert "snake_game.game.SnakeGame" in output


def test_snake_demo_reports_missing_class(tmp_path):
    """找不到游戏类时给出可读原因与尝试列表。"""
    project = tmp_path / "empty_game"
    project.mkdir()
    (project / "nothing.py").write_text("x = 1\n", encoding="utf-8")
    output = build_tool("/tool/game/snake/demo", str(tmp_path)).fn(path="empty_game")
    assert "退出码 3" in output
    assert "未找到可实例化的游戏类" in output


# ---------------------------------------------------------------------------
# 清单往返
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """save → load 保持工具与 skill 集合一致。"""
    manifest = CapabilityManifest()
    manifest.add_tool(build_builtin_tool("/tool/os/echo", str(tmp_path)))
    manifest.add_skill(SkillDef(name="demo", description="演示技能", source="builtin"))
    save_manifest(manifest, tmp_path)

    reloaded = load_manifest(tmp_path)
    assert reloaded.tool_paths() == ["/tool/os/echo"]
    assert reloaded.skill_names() == ["demo"]
    assert reloaded.get_tool("/tool/os/echo").fn is not None


def test_manifest_file_is_readable_json(tmp_path):
    """落盘内容是可 diff 的 UTF-8 JSON。"""
    manifest = CapabilityManifest()
    manifest.add_skill(SkillDef(name="中文", description="中文描述"))
    save_manifest(manifest, tmp_path)
    data = json.loads(manifest_path(tmp_path).read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["skills"][0]["name"] == "中文"


def test_duplicate_path_rejected(tmp_path):
    """同一路径不得重复挂载（B1）。"""
    manifest = CapabilityManifest()
    manifest.add_tool(build_builtin_tool("/tool/os/echo", str(tmp_path)))
    with pytest.raises(Exception):
        manifest.add_tool(build_builtin_tool("/tool/os/echo", str(tmp_path)))


def test_duplicate_skill_rejected(tmp_path):
    """同名 skill 不得重复添加。"""
    manifest = CapabilityManifest()
    manifest.add_skill(SkillDef(name="dup", description="a"))
    with pytest.raises(CapabilityError):
        manifest.add_skill(SkillDef(name="dup", description="b"))


def test_skill_path_collides_with_same_name(tmp_path):
    """skill 路径与已有条目冲突时拒绝。"""
    manifest = CapabilityManifest()
    manifest.add_skill(SkillDef(name="x", description="a"))
    with pytest.raises(Exception):
        manifest.add_skill(SkillDef(name="x", description="b", source="other"))


def test_unknown_tool_path_rejected_on_load(tmp_path):
    """清单里出现未知路径时拒绝装载（配置不能变成代码入口）。"""
    target = manifest_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"version": 1, "tools": [{"path": "/tool/evil/rm"}], "skills": []}),
        encoding="utf-8",
    )
    with pytest.raises(CapabilityError):
        load_manifest(tmp_path)


def test_version_mismatch_rejected(tmp_path):
    """版本不符时拒绝。"""
    target = manifest_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"version": 99, "tools": [], "skills": []}), encoding="utf-8")
    with pytest.raises(CapabilityError):
        load_manifest(tmp_path)


def test_missing_manifest_raises(tmp_path):
    """文件不存在时给出明确错误。"""
    with pytest.raises(CapabilityError):
        load_manifest(tmp_path)


# ---------------------------------------------------------------------------
# skill 文件
# ---------------------------------------------------------------------------


def test_skill_reader_returns_body(tmp_path):
    """skill 转成的工具能读回 SKILL.md。"""
    write_skill_files(tmp_path, "demo", {"SKILL.md": "# 标题\n正文"})
    manifest = CapabilityManifest()
    manifest.add_skill(SkillDef(name="demo", description="演示"))
    entry = manifest.to_tooldefs(str(tmp_path))[0]
    assert "正文" in entry.fn()


def test_skill_reader_blocks_escape(tmp_path):
    """skill 工具只能读自己目录内的文件。"""
    write_skill_files(tmp_path, "demo", {"SKILL.md": "x"})
    manifest = CapabilityManifest()
    manifest.add_skill(SkillDef(name="demo", description="演示"))
    entry = manifest.to_tooldefs(str(tmp_path))[0]
    with pytest.raises(ValueError):
        entry.fn(file="../../secret.txt")


def test_skill_illegal_name_rejected():
    """skill 名不得含路径分隔符。"""
    with pytest.raises(CapabilityError):
        SkillDef(name="a/b", description="x")


def test_write_skill_files_rejects_escape(tmp_path):
    """写入文件名不得含分隔符。"""
    with pytest.raises(CapabilityError):
        write_skill_files(tmp_path, "demo", {"../evil.md": "x"})
