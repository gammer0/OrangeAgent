"""权限模型测试（DAC，B5 / B15 / B16 / B17 / B18）。

对照 ``docs/02-数据结构设计/04-工具与权限.md`` §4.3。
"""

from __future__ import annotations

import pytest

from harness.permission import (
    AgentIdentity,
    AgentPermissions,
    Permission,
    PermissionDenied,
    check_src_matches_path,
    is_absolute_path,
    path_head,
    path_segments,
    visible,
)
from harness.tool_tree import EMPTY_SCHEMA, FlatStore, ToolDef, build_tree, tree_query


class TestNoRecordMeansAllow:
    def test_missing_record_is_allowed_B15(self) -> None:
        perms = AgentPermissions(agent_id="a")
        assert perms.is_allowed("/tool/os/file/read") is True

    def test_false_record_denies_B15(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file/read": False})
        assert perms.is_allowed("/tool/os/file/read") is False
        assert perms.is_denied("/tool/os/file/read") is True

    def test_true_record_allows_B15(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file/read": True})
        assert perms.is_allowed("/tool/os/file/read") is True

    def test_only_exceptions_are_stored_B15(self) -> None:
        """授权表只记例外项：默认表为空。"""
        assert AgentPermissions(agent_id="a").grants == {}


class TestPathKeyStrictness:
    def test_relative_key_rejected_B16(self) -> None:
        with pytest.raises(ValueError, match="B16"):
            AgentPermissions(agent_id="a", grants={"tool/os/file/read": False})

    def test_empty_key_rejected_B16(self) -> None:
        with pytest.raises(ValueError, match="B16"):
            AgentPermissions(agent_id="a", grants={"": False})

    def test_trailing_slash_key_rejected_B16(self) -> None:
        with pytest.raises(ValueError, match="B16"):
            AgentPermissions(agent_id="a", grants={"/tool/os/": False})

    def test_is_allowed_rejects_relative_input_B16(self) -> None:
        perms = AgentPermissions(agent_id="a")
        with pytest.raises(ValueError, match="B16"):
            perms.is_allowed("tool/os")

    def test_prefix_does_not_match_B16(self) -> None:
        """拒绝父路径不等于拒绝子路径，键必须逐字符匹配。"""
        perms = AgentPermissions(agent_id="a", grants={"/tool/os": False})
        assert perms.is_allowed("/tool/os/file/read") is True

    def test_grant_and_revoke_are_functional(self) -> None:
        perms = AgentPermissions(agent_id="a")
        denied = perms.grant("/tool/x", False)
        assert denied.is_allowed("/tool/x") is False
        assert perms.is_allowed("/tool/x") is True  # 原对象不变
        assert denied.revoke("/tool/x").is_allowed("/tool/x") is True


class TestPathHelpers:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/tool", True),
            ("/tool/os/file/read", True),
            ("/tool/a-b_c.d", True),
            ("tool/os", False),
            ("/", False),
            ("//tool", False),
            ("/tool//os", False),
            ("", False),
        ],
    )
    def test_is_absolute_path(self, path: str, expected: bool) -> None:
        assert is_absolute_path(path) is expected

    def test_path_segments(self) -> None:
        assert path_segments("/tool/os/file") == ["tool", "os", "file"]
        assert path_segments("/") == []

    def test_path_head(self) -> None:
        assert path_head("/skill/mmd") == "skill"
        assert path_head("skill/mmd") is None


class TestPermissionModel:
    def test_src_must_match_first_segment_B17(self) -> None:
        check_src_matches_path("skill", "/skill/mmd")
        with pytest.raises(ValueError, match="B17"):
            check_src_matches_path("tool", "/skill/mmd")

    def test_permission_bit_mapping_4_3(self) -> None:
        assert Permission(src="tool").permission_bit == "call"
        assert Permission(src="skill").permission_bit == "read"
        assert Permission(src="mcp").permission_bit == "request"

    def test_caps_reject_duplicates(self) -> None:
        with pytest.raises(ValueError):
            Permission(src="tool", caps=["read", "read"])

    def test_tool_def_rejects_mismatched_src_B17(self) -> None:
        with pytest.raises(ValueError, match="B17"):
            ToolDef(
                tool_id="t1",
                path="/skill/mmd",
                permission=Permission(src="tool"),
            )

    def test_identity_is_owner_only(self) -> None:
        assert AgentIdentity(agent_id="a").role == "owner"
        with pytest.raises(ValueError):
            AgentIdentity(agent_id="a", role="admin")  # type: ignore[arg-type]


class TestVisibilityInTree:
    def _store(self) -> FlatStore:
        return FlatStore(
            [
                ToolDef(
                    tool_id="t1",
                    path="/tool/os/file/read",
                    schema=dict(EMPTY_SCHEMA),
                    caps=["read"],
                    permission=Permission(src="tool", caps=["read"]),
                    description="读取文件",
                ),
                ToolDef(
                    tool_id="t2",
                    path="/tool/os/file/write",
                    schema=dict(EMPTY_SCHEMA),
                    caps=["write"],
                    permission=Permission(src="tool", caps=["write"]),
                    description="写入文件",
                ),
            ]
        )

    def test_denied_leaf_absent_from_tree_B5(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file/write": False})
        tree = build_tree(self._store(), perms)
        assert tree.get_node("/tool/os/file/write") is None
        assert tree.get_node("/tool/os/file/read") is not None

    def test_denied_leaf_absent_from_index_B5(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file/write": False})
        tree = build_tree(self._store(), perms)
        # /tool/os 在 L2 会展示；被拒绝的 write 更深，不在索引里
        assert tree.tree_index().count("/tool/os") >= 1
        assert all("write" not in path for path in tree.paths)

    def test_denied_intermediate_hides_whole_subtree_B5(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file": False})
        tree = build_tree(self._store(), perms)
        assert tree.get_node("/tool/os/file/read") is None
        assert tree.get_node("/tool/os/file/write") is None
        assert tree.get_node("/tool/os") is not None

    def test_denied_node_absent_from_tree_query_B5(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os/file/write": False})
        tree = build_tree(self._store(), perms)
        result = tree_query("/tool/os/file", tree=tree)
        assert [c["path"] for c in result["children"]] == ["/tool/os/file/read"]

    def test_denied_mid_node_absent_from_query_B5(self) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/tool/os": False})
        tree = build_tree(self._store(), perms)
        result = tree_query("/tool", tree=tree)
        assert result["children"] == []
    def test_all_denied_yields_empty_tree_B5(self) -> None:
        perms = AgentPermissions(
            agent_id="a",
            grants={"/tool/os/file/read": False, "/tool/os/file/write": False},
        )
        tree = build_tree(self._store(), perms)
        assert tree.root.children == []


class TestScopeDoesNotAffectPermission_B18:
    def test_scope_is_ignored_by_decision_B18(self) -> None:
        """B18：side_effect_scope 不参与权限判定。"""
        perms = AgentPermissions(agent_id="a")
        for scope in ("session", "project", "global"):
            entry = ToolDef(
                tool_id=f"t_{scope}",
                path=f"/tool/{scope}",
                caps=["write"],
                permission=Permission(src="tool", caps=["write"]),
                side_effect_scope=scope,  # type: ignore[arg-type]
                description="x",
            )
            assert visible(entry.path, perms) is True

    def test_denied_scope_does_not_grant_B18(self) -> None:
        """反向验证：scope=session 也不能绕过 false 记录。"""
        perms = AgentPermissions(agent_id="a", grants={"/tool/a": False})
        assert visible("/tool/a", perms) is False


class TestPermissionDeniedShape:
    def test_error_type_is_3a_B5(self) -> None:
        err = PermissionDenied("/tool/x")
        assert err.error_type == "3a"
        assert err.path == "/tool/x"
        assert "3a" in str(err)
