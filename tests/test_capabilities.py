"""能力清单测试：工具登记、skill 落盘、清单往返与非法输入拒绝。"""

from __future__ import annotations

import json

import pytest

from orange.capabilities import (
    MANIFEST_VERSION,
    CapabilityError,
    CapabilityManifest,
    SkillDef,
    load_manifest,
    manifest_path,
    save_manifest,
    write_skill_files,
)
from orange.permission import Permission
from orange.tool_tree import ToolDef


def _fake_tool(path: str, caps: list[str] | None = None) -> ToolDef:
    """Build a minimal tool definition without a real implementation.

    The package ships no builtin tools, but the manifest logic still has to
    accept whatever a manifest declares; a stub keeps these tests focused on
    round-tripping rather than on tool behaviour.
    """
    return ToolDef(
        tool_id="t" + path.replace("/", "_"),
        path=path,
        fn=lambda **_: "ok",
        caps=list(caps or ["read"]),
        permission=Permission(src=path.strip("/").split("/")[0], caps=list(caps or ["read"])),
        side_effect_scope="project",
        description=f"stub for {path}",
    )


# ---------------------------------------------------------------------------
# 内置工具
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """save → load 保持工具与 skill 集合一致。

    工具用外来 Python 源，这样重新读回清单时能被真实重建——本部署不随包
    发布内置工具，但「清单往返后工具仍可执行」这一保证必须成立。
    """
    from orange.capabilities import SourceSpec, _build_external_tool

    ext = tmp_path / "ext"
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "echo.py").write_text(
        "def run(text: str = '') -> str:\n    return text\n", encoding="utf-8"
    )
    source = SourceSpec(kind="python", target="./ext/echo.py", callable="run")
    entry = {
        "path": "/tool/os/echo",
        "description": "echo",
        "caps": ["read"],
        "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "source": {"kind": "python", "target": "./ext/echo.py", "callable": "run"},
    }

    manifest = CapabilityManifest()
    manifest.add_tool(_build_external_tool("/tool/os/echo", entry, source, str(tmp_path)))
    manifest.sources["/tool/os/echo"] = source
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
    assert data["version"] == MANIFEST_VERSION
    assert data["skills"][0]["name"] == "中文"


def test_duplicate_path_rejected(tmp_path):
    """同一路径不得重复挂载（B1）。"""
    from orange.tool_tree import ToolTreeError

    manifest = CapabilityManifest()
    manifest.add_tool(_fake_tool("/tool/os/echo"))
    with pytest.raises(ToolTreeError):
        manifest.add_tool(_fake_tool("/tool/os/echo"))


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
    with pytest.raises(CapabilityError):
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
