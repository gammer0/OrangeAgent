"""分组、外来工具、MCP 与权限持久化的测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness.capabilities import (
    CapabilityError,
    CapabilityManifest,
    GroupDef,
    SourceSpec,
    group_of,
    load_manifest,
    normalize_group,
    save_manifest,
)
from harness.mcp_client import McpServerConfig
from harness.permission import Permission, PermissionDenied, visible
from harness.permission_store import (
    load_permissions,
    permissions_path,
    set_grant,
)
from harness.tool_tree import FlatStore, ToolDef, ToolTree, build_tree

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_tool(path: str, caps: list[str] | None = None) -> ToolDef:
    """Build a minimal tool definition for a path, without a real implementation.

    These tests exercise grouping and permission logic, not tool behaviour, so a
    stub keeps them independent of which tools (if any) the package ships.
    """
    return ToolDef(
        tool_id="t" + path.replace("/", "_"),
        path=path,
        fn=lambda **_: "",
        caps=list(caps or ["read"]),
        permission=Permission(src=path.strip("/").split("/")[0], caps=list(caps or ["read"])),
        side_effect_scope="project",
        description=f"stub for {path}",
    )


# ---------------------------------------------------------------------------
# 分组模型
# ---------------------------------------------------------------------------


class TestGroupNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("tool/os", "tool/os"),
            ("/tool/os", "tool/os"),
            ("//tool/os//", "tool/os"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_group(raw) == expected

    @pytest.mark.parametrize("bad", ["foo/bar", "tool/../x", "tool//x", "a"])
    def test_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(CapabilityError):
            normalize_group(bad)

    def test_group_of(self) -> None:
        assert group_of("/tool/os/file/read") == "tool/os/file"
        assert group_of("/tool/os") == "tool"
        assert group_of("/tool") == ""


class TestGroupDef:
    def test_path(self) -> None:
        assert GroupDef(group_id="/tool/os").path == "/tool/os"

    def test_requires_two_segments(self) -> None:
        with pytest.raises(CapabilityError):
            GroupDef(group_id="tool")


class TestManifestGroups:
    def test_add_and_remove(self) -> None:
        manifest = CapabilityManifest()
        manifest.add_group(GroupDef(group_id="tool/mine", description="我的"))
        assert manifest.get_group("/tool/mine") is not None
        with pytest.raises(CapabilityError):
            manifest.add_group(GroupDef(group_id="tool/mine"))
        assert manifest.remove_group("tool/mine") is True

    def test_remove_refuses_when_members(self, tmp_path: Path) -> None:
        manifest = CapabilityManifest()
        manifest.add_group(GroupDef(group_id="tool/mine"))
        _load_ext(manifest, tmp_path, "/tool/mine/x")
        with pytest.raises(CapabilityError):
            manifest.remove_group("tool/mine")
        assert manifest.remove_group("tool/mine", force=True) is True

    def test_group_ids_derived_from_paths(self, tmp_path: Path) -> None:
        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path, "/tool/a/b/c")
        ids = manifest.group_ids()
        assert "tool/a" in ids and "tool/a/b" in ids


# ---------------------------------------------------------------------------
# 外来工具
# ---------------------------------------------------------------------------


def _write_ext(tmp_path: Path, name: str = "hello.py") -> Path:
    target = tmp_path / "ext" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "def run(text='hi', repeat=1, workdir='.'):\n"
        "    return '|'.join([str(text)] * int(repeat)) + ' @' + str(workdir)\n"
        "def other():\n"
        "    return 'other'\n",
        encoding="utf-8",
    )
    return target


def _load_ext(
    manifest: CapabilityManifest, tmp_path: Path, path: str = "/tool/ext/hello"
) -> None:
    _write_ext(tmp_path)
    entry = {
        "path": path,
        "description": "测试外来工具",
        "caps": ["read"],
        "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "source": {"kind": "python", "target": "./ext/hello.py", "callable": "run"},
    }
    from harness.capabilities import _build_external_tool

    source = SourceSpec(kind="python", target="./ext/hello.py", callable="run")
    manifest.add_tool(_build_external_tool(path, entry, source, str(tmp_path)))
    manifest.sources[path] = source


class TestExternalTools:
    def test_roundtrip_manifest(self, tmp_path: Path) -> None:
        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path)
        save_manifest(manifest, tmp_path)

        reloaded = load_manifest(tmp_path)
        assert "/tool/ext/hello" in reloaded.tool_paths()
        source = reloaded.sources["/tool/ext/hello"]
        assert source.target == "./ext/hello.py"
        assert source.callable == "run"

    def test_executes_real_code(self, tmp_path: Path) -> None:
        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path)
        tool = manifest.tools[0]
        out = tool.fn(text="ab", repeat=2)
        assert out.startswith("ab|ab")
        assert str(tmp_path.resolve()) in out

    def test_missing_target_degrades_gracefully(self, tmp_path: Path) -> None:
        """目标缺失时不拖垮整张清单，记入 problems。"""
        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path)
        save_manifest(manifest, tmp_path)
        (tmp_path / "ext" / "hello.py").unlink()

        problems: list[str] = []
        reloaded = load_manifest(tmp_path, problems=problems)
        assert "/tool/ext/hello" not in reloaded.tool_paths()
        assert any("hello" in p for p in problems)

    def test_nested_group_path(self, tmp_path: Path) -> None:
        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path, "/tool/a/b/hello")
        store = FlatStore(manifest.to_tooldefs(str(tmp_path)))
        tree = build_tree(store)
        assert "/tool/a" in tree.paths
        assert "/tool/a/b/hello" in tree.paths
        assert tree.get_node("/tool/a").kind == "abstract"


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


class TestMcp:
    def _config(self) -> McpServerConfig:
        server = FIXTURES / "mcp_echo_server.py"
        return McpServerConfig(
            name="echo", command=sys.executable, args=(str(server),)
        )

    def test_list_tools(self) -> None:
        from harness.mcp_client import list_tools

        tools = list_tools(self._config())
        assert [t["name"] for t in tools] == ["echo", "add"]
        assert tools[0]["inputSchema"]["properties"]

    def test_call_tool(self) -> None:
        from harness.mcp_client import call_tool

        assert call_tool(self._config(), "echo", {"text": "hi"}) == "hi"
        assert call_tool(self._config(), "add", {"a": 2, "b": 5}) == "7"

    def test_unknown_tool_errors(self) -> None:
        from harness.mcp_client import McpError, call_tool

        with pytest.raises(McpError):
            call_tool(self._config(), "nope", {})

    def test_bad_command_errors(self) -> None:
        from harness.mcp_client import McpError, McpServerConfig, list_tools

        bad = McpServerConfig(name="x", command="definitely-not-a-command-xyz")
        with pytest.raises(McpError):
            list_tools(bad)


# ---------------------------------------------------------------------------
# 权限持久化
# ---------------------------------------------------------------------------


class TestPermissionStore:
    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """权限文件在系统数据目录（所有项目共享），每个用例必须用独立目录。"""
        home = tmp_path / "home"
        monkeypatch.setenv("HARNESS_HOME", str(home))
        return home

    def test_default_allows_everything(self, tmp_path: Path) -> None:
        perms = load_permissions(tmp_path, "local")
        assert perms.grants == {}
        assert visible("/tool/os/file/read", perms) is True

    def test_set_grant_persists(self, tmp_path: Path) -> None:
        set_grant(tmp_path, "/tool/os/shell/run", allowed=False, agent_id="local")
        assert permissions_path(tmp_path).is_file()

        perms = load_permissions(tmp_path, "local")
        assert perms.grants["/tool/os/shell/run"] is False
        assert visible("/tool/os/shell/run", perms) is False

    def test_agents_are_isolated(self, tmp_path: Path) -> None:
        set_grant(tmp_path, "/tool/os/shell/run", allowed=False, agent_id="local")
        assert visible("/tool/os/shell/run", load_permissions(tmp_path, "local")) is False
        assert visible("/tool/os/shell/run", load_permissions(tmp_path, "web")) is True

    def test_grant_restores(self, tmp_path: Path) -> None:
        set_grant(tmp_path, "/tool/os/shell/run", allowed=False)
        set_grant(tmp_path, "/tool/os/shell/run", allowed=True)
        perms = load_permissions(tmp_path, "local")
        assert "/tool/os/shell/run" not in perms.grants

    def test_ancestor_denial_hides_subtree(self, tmp_path: Path) -> None:
        set_grant(tmp_path, "/tool/net", allowed=False, agent_id="local")
        perms = load_permissions(tmp_path, "local")
        assert visible("/tool/net", perms) is False
        assert visible("/tool/net/http/get", perms) is False
        assert visible("/tool/os/file/read", perms) is True

    def test_rejects_relative_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            set_grant(tmp_path, "tool/os", allowed=False)

    def test_corrupt_file_degrades(self, tmp_path: Path) -> None:
        target = permissions_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{ not json", encoding="utf-8")
        problems: list[str] = []
        perms = load_permissions(tmp_path, "local", problems=problems)
        assert perms.grants == {}

    def test_save_preserves_other_agents(self, tmp_path: Path) -> None:
        set_grant(tmp_path, "/tool/a", allowed=False, agent_id="local")
        set_grant(tmp_path, "/tool/b", allowed=False, agent_id="web")
        data = json.loads(permissions_path(tmp_path).read_text(encoding="utf-8"))
        assert set(data["agents"]) == {"local", "web"}


class TestPermissionBlocksTree:
    """权限必须真实屏蔽：查询不可见 + 调用被拒。"""

    def _tree(self, tmp_path: Path, grants: dict[str, bool]) -> ToolTree:
        from harness.permission import AgentPermissions

        manifest = CapabilityManifest()
        _load_ext(manifest, tmp_path, "/tool/ext/hello")
        store = FlatStore(manifest.to_tooldefs(str(tmp_path)))
        perms = AgentPermissions(agent_id="t", grants=grants)
        return build_tree(store, perms)

    def test_denied_leaf_hidden_from_index(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path, {"/tool/ext/hello": False})
        assert "/tool/ext/hello" not in tree.tree_index()

    def test_denied_leaf_raises_on_resolve(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path, {"/tool/ext/hello": False})
        with pytest.raises(PermissionDenied) as exc:
            tree.resolve("/tool/ext/hello")
        assert exc.value.error_type == "3a"

    def test_denied_parent_hides_children(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path, {"/tool/ext": False})
        index = tree.tree_index()
        assert "/tool/ext" not in index
        assert "/tool/ext/hello" not in index
        with pytest.raises(PermissionDenied):
            tree.resolve("/tool/ext/hello")


class TestGroupLevelDenial:
    """分组级禁用：拒一个分组 = 禁用其下所有工具。

    这是权限系统最常用的操作——用户想关掉「网络」或「执行」整类能力，
    而不是逐个工具去 deny。
    """

    def _tree_with(self, tmp_path: Path, grants: dict[str, bool]) -> ToolTree:
        """构造含多层级分组的树。

        用测试自带的假工具而不是内置工具：本部署不随包发布任何内置工具，
        但分组级禁用是纯粹的权限逻辑，不应依赖具体工具是否存在。
        """
        from harness.permission import AgentPermissions

        manifest = CapabilityManifest()
        for path in (
            "/tool/os/file/read",
            "/tool/os/file/write",
            "/tool/net/http/get",
            "/tool/net/http/post",
        ):
            manifest.add_tool(_fake_tool(path))
        store = FlatStore(manifest.to_tooldefs(str(tmp_path)))
        return build_tree(store, AgentPermissions(agent_id="t", grants=grants))

    def test_denying_group_hides_all_children(self, tmp_path: Path) -> None:
        tree = self._tree_with(tmp_path, {"/tool/net": False})
        paths = set(tree.paths)
        assert "/tool/net" not in paths
        assert not any(p.startswith("/tool/net/") for p in paths)

    def test_denying_group_leaves_siblings(self, tmp_path: Path) -> None:
        tree = self._tree_with(tmp_path, {"/tool/net": False})
        paths = set(tree.paths)
        assert "/tool/os/file/read" in paths
        assert "/tool/os/file/write" in paths

    def test_denying_group_blocks_calls(self, tmp_path: Path) -> None:
        tree = self._tree_with(tmp_path, {"/tool/net": False})
        for path in ("/tool/net/http/get", "/tool/net/http/post"):
            with pytest.raises(PermissionDenied):
                tree.resolve(path)

    def test_denying_l1_root_blanks_whole_branch(self, tmp_path: Path) -> None:
        tree = self._tree_with(tmp_path, {"/tool": False})
        assert not [p for p in tree.paths if p.startswith("/tool")]
        assert "/tool" not in tree.tree_index()

    def test_denying_deep_group_keeps_ancestors(self, tmp_path: Path) -> None:
        """拒 L3 分组时，L1/L2 祖先仍在（它们本身未被拒）。"""
        tree = self._tree_with(tmp_path, {"/tool/os/file": False})
        paths = set(tree.paths)
        assert "/tool" in paths
        assert "/tool/os" in paths
        assert "/tool/os/file" not in paths
        assert "/tool/os/file/read" not in paths

    def test_ancestor_denial_beats_child_grant(self, tmp_path: Path) -> None:
        """祖先被拒时，子路径显式允许也不能绕过（拒绝优先，更安全）。"""
        tree = self._tree_with(
            tmp_path, {"/tool/net": False, "/tool/net/http/get": True}
        )
        assert "/tool/net/http/get" not in tree.paths
        with pytest.raises(PermissionDenied):
            tree.resolve("/tool/net/http/get")

    def test_group_denial_survives_persistence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """分组拒绝必须能落盘并重新读回。"""
        from harness.permission_store import load_permissions, set_grant

        monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
        set_grant(tmp_path, "/tool/net", allowed=False, agent_id="local")

        perms = load_permissions(tmp_path, "local")
        assert perms.grants.get("/tool/net") is False
        assert visible("/tool/net/http/get", perms) is False
        assert not (tmp_path / ".agent").exists()
