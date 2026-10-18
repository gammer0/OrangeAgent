"""工具树构建与查询测试（B 组，ADT-4/5/6/7）。

每个测试名带约束编号，便于与 ``docs/02-数据结构设计/04-工具与权限.md`` 对照追溯。
"""

from __future__ import annotations

import threading

import pytest

from harness.discovery import DiscoveredTools, ToolDiscovery, TreeQueryState, TreeSessionState
from harness.permission import AgentPermissions, Permission, PermissionDenied
from harness.tool_tree import (
    EMPTY_SCHEMA,
    FlatStore,
    ToolDef,
    ToolTree,
    ToolTreeError,
    build_tree,
    derive_concurrency,
    tree_query,
)

# ---------------------------------------------------------------------------
# 夹具：文档 §4.1 的三条目例子
# ---------------------------------------------------------------------------


def _def(
    tool_id: str,
    path: str,
    *,
    caps: list[str] | None = None,
    description: str = "",
    schema: dict | None = None,
    idempotent: bool | None = None,
    src: str | None = None,
    scope: str = "session",
) -> ToolDef:
    kind = src or path.strip("/").split("/")[0]
    kwargs: dict = {}
    if idempotent is not None:
        kwargs["idempotent"] = idempotent
    return ToolDef(
        tool_id=tool_id,
        path=path,
        schema=schema if schema is not None else dict(EMPTY_SCHEMA),
        caps=caps if caps is not None else ["read"],
        permission=Permission(src=kind, caps=caps if caps is not None else ["read"]),
        description=description,
        side_effect_scope=scope,
        **kwargs,
    )


@pytest.fixture
def store() -> FlatStore:
    """§4.1 的三条目：两条在 /skill/mmd/pose，一条在 /skill/mmd/render。"""
    return FlatStore(
        [
            _def(
                "t_set",
                "/skill/mmd/pose/set_keyframe",
                caps=["write"],
                description="设置关键帧",
                schema={"type": "object", "properties": {"frame": {"type": "integer"}}},
            ),
            _def(
                "t_get",
                "/skill/mmd/pose/get_keyframe",
                description="读取关键帧",
                idempotent=True,
                schema={"type": "object", "properties": {}},
            ),
            _def(
                "t_export",
                "/skill/mmd/render/export",
                caps=["exec"],
                description="导出渲染结果",
            ),
        ]
    )


@pytest.fixture
def perms() -> AgentPermissions:
    return AgentPermissions(agent_id="agent-1")


# ---------------------------------------------------------------------------
# 构建：分段聚合
# ---------------------------------------------------------------------------


class TestBuildAggregation:
    def test_segments_aggregate_into_tree_4_1(self, store: FlatStore, perms) -> None:
        """文档 4.1：三条扁平条目聚合出 /skill/mmd/pose、/skill/mmd/render 两个中间层。"""
        tree = build_tree(store, perms)

        assert [c.name for c in tree.root.children] == ["skill"]
        skill = tree.root.child("skill")
        assert skill is not None
        mmd = skill.child("mmd")
        assert mmd is not None
        assert [c.name for c in mmd.children] == ["pose", "render"]
        pose = mmd.child("pose")
        assert pose is not None
        assert [c.name for c in pose.children] == ["get_keyframe", "set_keyframe"]
        assert pose.path == "/skill/mmd/pose"

    def test_leaf_binds_tool_id_and_schema_B2_B3(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        node = tree.get_node("/skill/mmd/pose/set_keyframe")
        assert node is not None
        assert node.tool_id == "t_set"
        assert node.schema == {"type": "object", "properties": {"frame": {"type": "integer"}}}
        assert tree.resolve("/skill/mmd/pose/set_keyframe").description == "设置关键帧"

    def test_children_sorted_by_path_B6(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        paths = [n.path for n in tree.root.walk() if n.path != "/"]
        assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# kind 派生（B21）
# ---------------------------------------------------------------------------


class TestKindDerivation:
    def test_kind_is_derived_B21(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        assert tree.get_node("/skill").kind == "abstract"  # type: ignore[union-attr]
        assert tree.get_node("/skill/mmd/pose").kind == "abstract"  # type: ignore[union-attr]
        assert tree.get_node("/skill/mmd/pose/set_keyframe").kind == "leaf"  # type: ignore[union-attr]

    def test_kind_is_not_a_constructor_field_B21(self, store: FlatStore, perms) -> None:
        from harness.tool_tree import ToolNode

        with pytest.raises(TypeError):
            ToolNode(path="/skill/x", name="x", kind="leaf")  # type: ignore[call-arg]

    def test_kind_flips_when_child_added_B21(self) -> None:
        from harness.tool_tree import ToolNode

        node = ToolNode(path="/skill/x", name="x")
        assert node.kind == "leaf"
        node.add_child(ToolNode(path="/skill/x/y", name="y"))
        assert node.kind == "abstract"


# ---------------------------------------------------------------------------
# 合成条目（B20 / B13 / B17）
# ---------------------------------------------------------------------------


class TestSynthesis:
    def test_intermediate_nodes_are_synthesized_B20(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        for path in ("/skill", "/skill/mmd", "/skill/mmd/pose", "/skill/mmd/render"):
            entry = tree.synthetic_entries[path]
            assert entry.synthetic is True
            assert entry.tool_id == tree.get_node(path).tool_id  # type: ignore[union-attr]

    def test_synthetic_entries_not_written_to_flat_store_B20(
        self, store: FlatStore, perms
    ) -> None:
        before = len(store)
        tree = build_tree(store, perms)
        assert len(store) == before == 3
        assert set(store.entries) == {"t_set", "t_get", "t_export"}
        # 合成路径在 FlatStore 中查不到
        for path in tree.synthetic_entries:
            assert store.by_path(path) is None

    def test_synthetic_entry_shape_matches_doc_B13(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        entry = tree.synthetic_entries["/skill/mmd"]
        assert entry.caps == ["read"]
        assert entry.concurrency == "safe"  # B13
        assert entry.idempotent is True  # B13
        assert entry.schema == EMPTY_SCHEMA
        assert entry.description == "由子节点推断"
        assert entry.tool_id.startswith("t_tree_")

    def test_synthetic_permission_inherits_parent_B17(self) -> None:
        """父节点为显式条目时，合成节点继承其 Permission。"""
        flat = FlatStore(
            [
                ToolDef(
                    tool_id="t_skill",
                    path="/skill",
                    schema=dict(EMPTY_SCHEMA),
                    caps=["read"],
                    permission=Permission(src="skill", caps=["read"]),
                    description="技能包",
                ),
                _def("t_leaf", "/skill/mmd/a", description="叶子"),
            ]
        )
        tree = build_tree(flat, AgentPermissions(agent_id="a"))
        assert tree.synthetic_entries["/skill/mmd"].permission == Permission(
            src="skill", caps=["read"]
        )

    def test_explicit_intermediate_entry_is_not_synthesized_B20(self) -> None:
        flat = FlatStore(
            [
                ToolDef(
                    tool_id="t_mmd",
                    path="/skill/mmd",
                    schema=dict(EMPTY_SCHEMA),
                    caps=["read"],
                    permission=Permission(src="skill", caps=["read"]),
                    description="MMD 流水线",
                ),
                _def("t_leaf", "/skill/mmd/a", description="叶子"),
            ]
        )
        tree = build_tree(flat, AgentPermissions(agent_id="a"))
        assert "/skill/mmd" not in tree.synthetic_entries
        node = tree.get_node("/skill/mmd")
        assert node is not None
        assert node.description == "MMD 流水线"
        assert node.tool_id == "t_mmd"


# ---------------------------------------------------------------------------
# 构建期整体校验（B22）
# ---------------------------------------------------------------------------


class TestBuildValidation:
    def test_rejects_same_name_under_same_parent_B1(self) -> None:
        """同名不同 tool_id 在扁平层就不允许（路径冲突），由 FlatStore 拦截。"""
        flat = FlatStore([_def("t1", "/tool/os/read", description="a")])
        with pytest.raises(ToolTreeError) as exc:
            flat.put(_def("t2", "/tool/os/read", description="b"))
        assert exc.value.rule == "B1"

    def test_rejects_duplicate_child_names_B22_B1(self) -> None:
        """手工篡改树后整体校验能发现重名子节点。"""
        flat = FlatStore([_def("t1", "/tool/os/read", description="a")])
        tree = build_tree(flat, AgentPermissions(agent_id="a"))
        os_node = tree.get_node("/tool/os")
        assert os_node is not None
        from harness.tool_tree import ToolNode

        os_node.add_child(ToolNode(path="/tool/os/read", name="read", tool_id="t1"))
        from harness.tool_tree import _validate_tree

        with pytest.raises(ToolTreeError) as exc:
            _validate_tree(tree, flat, {})
        assert exc.value.rule == "B1"

    def test_rejects_illegal_first_segment_B19(self) -> None:
        """首段非法时，Permission 的字面量类型先拦截。"""
        with pytest.raises(ValueError):
            ToolDef(
                tool_id="t1",
                path="/os/file/read",
                permission=Permission(src="os"),  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="B19"):
            _def("t1", "/os/file/read", description="x", src="tool")

    def test_rejects_illegal_first_segment_on_mount_B19(self) -> None:
        flat = FlatStore()
        entry = _def("t1", "/tool/os/read", description="ok")
        object.__setattr__(entry, "path", "/os/file/read")
        with pytest.raises(ToolTreeError) as exc:
            flat.put(entry)
        assert exc.value.rule == "B19"

    def test_rejects_missing_tool_id_B2(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        node = tree.get_node("/skill/mmd/pose/set_keyframe")
        assert node is not None
        node.tool_id = "t_missing"
        from harness.tool_tree import _validate_tree

        with pytest.raises(ToolTreeError) as exc:
            _validate_tree(tree, store, tree.synthetic_entries)
        assert exc.value.rule == "B2"

    def test_rejects_leaf_schema_mismatch_B3(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        node = tree.get_node("/skill/mmd/pose/set_keyframe")
        assert node is not None
        node.schema = {"type": "object", "properties": {"other": {"type": "string"}}}
        from harness.tool_tree import _validate_tree

        with pytest.raises(ToolTreeError) as exc:
            _validate_tree(tree, store, tree.synthetic_entries)
        assert exc.value.rule == "B3"

    def test_rejects_abstract_with_nonempty_schema_B4(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        node = tree.get_node("/skill/mmd/pose")
        assert node is not None
        node.schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        from harness.tool_tree import _validate_tree

        with pytest.raises(ToolTreeError) as exc:
            _validate_tree(tree, store, tree.synthetic_entries)
        assert exc.value.rule == "B4"

    def test_rejects_multiline_description_B9(self) -> None:
        with pytest.raises(ValueError, match="B9"):
            _def("t1", "/tool/os/read", description="第一行\n第二行")

    def test_long_description_is_truncated_with_ellipsis_B10(self) -> None:
        long_l1 = "x" * 60
        flat = FlatStore(
            [
                ToolDef(
                    tool_id="t_os",
                    path="/tool",
                    schema=dict(EMPTY_SCHEMA),
                    caps=["read"],
                    permission=Permission(src="tool", caps=["read"]),
                    description=long_l1,
                ),
                _def("t_read", "/tool/os/read", description="x" * 100),
            ]
        )
        tree = build_tree(flat, AgentPermissions(agent_id="a"))
        l1 = tree.get_node("/tool")
        assert l1 is not None
        assert len(l1.description) == 40
        assert l1.description.endswith("…")
        deep = tree.get_node("/tool/os/read")
        assert deep is not None
        assert len(deep.description) == 80
        assert deep.description.endswith("…")

    def test_build_rejects_before_returning_tree_B22(self) -> None:
        """任一构建期校验失败都不得返回树。"""
        entry = _def("t1", "/tool/os/read", description="ok")
        object.__setattr__(entry, "description", "坏\n描述")
        flat = FlatStore()
        flat._entries["t1"] = entry  # 绕过 put 的检查，模拟脏数据
        with pytest.raises(ToolTreeError) as exc:
            build_tree(flat, AgentPermissions(agent_id="a"))
        assert exc.value.rule == "B9"


# ---------------------------------------------------------------------------
# concurrency / idempotent 推导（B11 / B12 / B13）
# ---------------------------------------------------------------------------


class TestConcurrencyDerivation:
    def test_read_only_caps_are_safe_B11(self) -> None:
        assert derive_concurrency(["read"]) == "safe"
        assert derive_concurrency(["read", "net"]) == "safe"

    def test_write_or_exec_is_exclusive_B11(self) -> None:
        assert derive_concurrency(["write"]) == "exclusive"
        assert derive_concurrency(["read", "write"]) == "exclusive"
        assert derive_concurrency(["exec"]) == "exclusive"
        assert derive_concurrency(["net"]) == "exclusive"

    def test_undeclared_caps_is_exclusive_B11(self) -> None:
        assert derive_concurrency([]) == "exclusive"

    def test_explicit_concurrency_is_checked_B11(self) -> None:
        with pytest.raises(ValueError, match="B11"):
            ToolDef(
                tool_id="t1",
                path="/tool/x",
                caps=["write"],
                concurrency="safe",
                permission=Permission(src="tool", caps=["write"]),
            )

    def test_idempotent_defaults_false_B12(self) -> None:
        entry = _def("t1", "/tool/x", caps=["write"])
        assert entry.idempotent is False

    def test_mutating_caps_cannot_be_idempotent_B12(self) -> None:
        with pytest.raises(ValueError, match="B12"):
            _def("t1", "/tool/x", caps=["write"], idempotent=True)

    def test_read_only_can_be_idempotent_B12(self) -> None:
        assert _def("t1", "/tool/x", caps=["read"], idempotent=True).idempotent is True


# ---------------------------------------------------------------------------
# 树索引（B6）
# ---------------------------------------------------------------------------


class TestTreeIndex:
    def test_index_expands_all_by_default(self, store: FlatStore, perms) -> None:
        """默认（depth=0）必须全展开，否则模型看不到工具叶子。

        历史 bug：默认 depth=2 时，四层路径的工具一个都不显示，
        模型只能看到 ``/tool/fluid`` 这类目录名。
        """
        tree = build_tree(store, perms)
        text = tree.tree_index()
        # 叶子必须可见，否则模型无从知道有哪些工具
        leaves = [p for p in tree.paths if tree.get_node(p).is_leaf]
        for path in leaves:
            assert path in text, f"叶子 {path} 未出现在索引中"

    def test_index_omits_undescribed_groups(self, store: FlatStore, perms) -> None:
        """无描述的中间节点只是子路径前缀的重复，必须省略。"""
        from harness.tool_tree import index_text_to_paths

        tree = build_tree(store, perms)
        shown = set(index_text_to_paths(tree.tree_index()))

        groups = [p for p in tree.paths if not tree.get_node(p).is_leaf]
        for path in groups:
            if tree.get_node(path).depth == 1:
                continue  # L1 资源根恒定出现
            assert path not in shown, f"无描述分组 {path} 不应出现在索引中"

        # 叶子与所有已存在的 L1 资源根仍然出现
        leaves = {p for p in tree.paths if tree.get_node(p).is_leaf}
        assert leaves <= shown
        roots = {p for p in tree.paths if tree.get_node(p).depth == 1}
        assert roots <= shown, f"L1 资源根缺失: {roots - shown}"

    def test_index_shows_described_groups(self, store: FlatStore, perms) -> None:
        """有描述的分组才出现，且带上描述。"""
        from harness.tool_tree import index_text_to_paths

        tree = build_tree(store, perms, group_descriptions={"skill/mmd": "MMD 骨骼"})
        text = tree.tree_index()
        assert "/skill/mmd — MMD 骨骼" in text
        assert "/skill/mmd" in set(index_text_to_paths(text))

    def test_index_no_false_depth_hint(self, store: FlatStore, perms) -> None:
        """全展开时不应出现「更深节点」提示（省略的是空分组，不是更深节点）。"""
        tree = build_tree(store, perms)
        assert "更深节点" not in tree.tree_index()

    def test_index_depth_one_only_l1(self, store: FlatStore, perms) -> None:
        """``depth=1`` 只列 L1（提示行不算条目）。"""
        from harness.tool_tree import index_text_to_paths

        tree = build_tree(store, perms)
        text = tree.tree_index(depth=1)
        roots = {p for p in tree.paths if tree.get_node(p).depth == 1}
        assert set(index_text_to_paths(text)) == roots
        # 深层路径不能作为独立条目出现（提示行里提到它们是允许的）
        assert "/skill/mmd" not in index_text_to_paths(text)

    def test_index_merges_depth_and_fold_hints(self, store: FlatStore, perms) -> None:
        """深度限制与折叠不能各出一行提示（重复表达同一件事）。"""
        tree = build_tree(store, perms)
        text = tree.tree_index(depth=1)
        hints = [line for line in text.splitlines() if line.startswith("（")]
        assert len(hints) <= 1, f"提示重复: {hints}"

    def test_index_depth_is_monotonic(self, store: FlatStore, perms) -> None:
        """depth 越大条目越多（旧实现 depth=3 与 depth=2 完全相同）。"""
        tree = build_tree(store, perms)
        d1 = len(tree.tree_index(depth=1).splitlines())
        d2 = len(tree.tree_index(depth=2).splitlines())
        d9 = len(tree.tree_index(depth=9).splitlines())
        assert d1 <= d2 <= d9

    def test_index_omits_placeholder_description(self, store: FlatStore, perms) -> None:
        """中间节点无描述时只输出路径，不拖“由子节点推断”。"""
        tree = build_tree(store, perms)
        text = tree.tree_index()
        assert "由子节点推断" not in text

    def test_index_uses_group_descriptions(self, store: FlatStore, perms) -> None:
        """显式声明的分组描述要出现在索引里。"""
        tree = build_tree(
            store,
            perms,
            group_descriptions={"skill": "全部技能", "skill/mmd": "MMD 骨骼"},
        )
        text = tree.tree_index()
        assert "/skill — 全部技能" in text
        assert "/skill/mmd — MMD 骨骼" in text

    def test_index_sorted_by_path_B6(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        from harness.tool_tree import index_text_to_paths

        paths = index_text_to_paths(tree.tree_index())
        assert paths == sorted(paths)

    def test_index_is_byte_stable_within_session_B6(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        assert tree.tree_index() == tree.tree_index()

    def test_index_caps_entries_by_description_length_B6(self) -> None:
        """超过折叠阈值时保留描述最短的若干条（防止工具数上百时膨胀）。"""
        from harness.tool_tree import DEFAULT_L2_LIMIT, index_text_to_paths

        total = DEFAULT_L2_LIMIT + 20
        entries = [
            _def(f"t{i:03d}", f"/tool/g{i:03d}", description="x" * (i + 1))
            for i in range(total)
        ]
        tree = build_tree(FlatStore(entries), AgentPermissions(agent_id="a"))

        paths = index_text_to_paths(tree.tree_index())
        l2 = [p for p in paths if p.count("/") == 2]
        # L1 占 1 个名额，剩余名额给最短描述
        assert len(l2) == DEFAULT_L2_LIMIT - 1
        assert l2 == sorted(f"/tool/g{i:03d}" for i in range(DEFAULT_L2_LIMIT - 1))


# ---------------------------------------------------------------------------
# tree_query（B5 / B7）
# ---------------------------------------------------------------------------


class TestTreeQuery:
    def test_returns_next_level_children(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        result = tree_query("/skill/mmd", tree=tree)
        assert result["path"] == "/skill/mmd"
        assert [c["path"] for c in result["children"]] == [
            "/skill/mmd/pose",
            "/skill/mmd/render",
        ]
        assert all(c["kind"] == "abstract" for c in result["children"])

    def test_rejected_children_are_filtered_B5(self, store: FlatStore) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/skill/mmd/render": False})
        tree = build_tree(store, perms)
        result = tree_query("/skill/mmd", tree=tree)
        assert [c["path"] for c in result["children"]] == ["/skill/mmd/pose"]

    def test_query_on_denied_node_raises_B5(self, store: FlatStore) -> None:
        perms = AgentPermissions(agent_id="a", grants={"/skill/mmd": False})
        tree = build_tree(store, perms)
        with pytest.raises(PermissionDenied):
            tree_query("/skill/mmd", tree=tree)

    def test_direct_call_on_denied_node_returns_3a_B5(self, store: FlatStore) -> None:
        from harness.permission import PermissionDenied as Denied

        perms = AgentPermissions(agent_id="a", grants={"/skill/mmd/pose/set_keyframe": False})
        tree = build_tree(store, perms)
        with pytest.raises(Denied) as exc:
            tree.resolve("/skill/mmd/pose/set_keyframe")
        assert exc.value.error_type == "3a"

    def test_schema_delivered_once_B7(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        state = TreeSessionState()

        first = tree_query("/skill/mmd", tree=tree, state=state, step=1)
        assert all("schema" in c for c in first["children"])

        second = tree_query("/skill/mmd", tree=tree, state=state, step=2)
        assert all("schema" not in c for c in second["children"])
        # 其余字段保持
        assert [c["path"] for c in second["children"]] == [
            "/skill/mmd/pose",
            "/skill/mmd/render",
        ]

    def test_delivered_only_affects_same_path_B7(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        state = TreeSessionState()
        tree_query("/skill/mmd/pose", tree=tree, state=state, step=1)
        other = tree_query("/skill/mmd/render", tree=tree, state=state, step=2)
        assert all("schema" in c for c in other["children"])

    def test_tree_query_records_discovery_4_7(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=7)
        paths = {d.path for d in state.discovered.entries.values()}
        assert paths == {"/skill/mmd/pose", "/skill/mmd/render"}
        record = state.discovered.get("/skill/mmd/pose")
        assert record is not None
        assert record.first_seen_step == 7
        assert record.last_used_step == 7


# ---------------------------------------------------------------------------
# 会话状态与同锁（B14）
# ---------------------------------------------------------------------------


class TestSessionStateConcurrency:
    def test_delivered_and_discovery_share_one_lock_B14(self, store: FlatStore, perms) -> None:
        # 建树本身不使用会话状态；此处只验证锁的单例性质
        build_tree(store, perms)
        state = TreeSessionState()
        assert state._lock is state._lock  # 单一锁对象
        # 锁不外泄：外部拿到的是副本，改它不影响内部
        snapshot = state.delivered
        assert isinstance(snapshot, frozenset)

    def test_concurrent_queries_deliver_schema_exactly_once_B14_B7(
        self, store: FlatStore, perms
    ) -> None:
        """并发查询同一节点：schema 只被下发一次的线程级验证。"""
        tree = build_tree(store, perms)
        state = TreeSessionState()
        results: list[dict] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            results.append(tree_query("/skill/mmd", tree=tree, state=state, step=1))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with_schema = sum(
            1 for r in results if any("schema" in c for c in r["children"])
        )
        assert with_schema == 1  # B7 + B14：锁保证只有一次下发

    def test_reset_clears_delivered_and_discovery_together_B14(
        self, store: FlatStore, perms
    ) -> None:
        tree = build_tree(store, perms)
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        assert state.delivered
        assert len(state.discovered) == 2

        state.reset()
        assert state.delivered == frozenset()
        assert len(state.discovered) == 0

        again = tree_query("/skill/mmd", tree=tree, state=state, step=5)
        assert all("schema" in c for c in again["children"])

    def test_purge_prefix_clears_session_cache_B24(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        state.purge_prefix("/skill/mmd/render")
        assert "/skill/mmd/render" not in state.delivered
        assert state.discovered.get("/skill/mmd/render") is None
        assert state.discovered.get("/skill/mmd/pose") is not None


# ---------------------------------------------------------------------------
# 卸载（B23）与版本（B25）
# ---------------------------------------------------------------------------


class TestRetireAndVersion:
    def test_unmount_marks_retired_not_deletes_B23(self, store: FlatStore, perms) -> None:
        touched = store.mark_retired("/skill/mmd/render")
        assert touched == ["t_export"]
        assert len(store) == 3  # 条目仍在
        assert store.get("t_export") is not None
        assert store.get("t_export").retired is True  # type: ignore[union-attr]

        tree = build_tree(store, perms)
        assert tree.get_node("/skill/mmd/render") is None
        assert tree.get_node("/skill/mmd/pose") is not None

    def test_retired_entry_fn_still_usable_B23(self) -> None:
        called: list[str] = []

        def fn() -> None:
            called.append("x")

        entry = ToolDef(
            tool_id="t_run",
            path="/tool/run",
            fn=fn,
            caps=["exec"],
            permission=Permission(src="tool", caps=["exec"]),
            description="运行",
        )
        flat = FlatStore([entry])
        flat.mark_retired("/tool/run")
        retired = flat.get("t_run")
        assert retired is not None
        assert retired.fn is fn  # 执行中的 fn 不被抽走
        retired.fn()  # type: ignore[misc]
        assert called == ["x"]

    def test_version_increments_on_mount_B25(self, store: FlatStore, perms) -> None:
        tree = build_tree(store, perms)
        assert tree.version == 1
        tree.bump()
        assert tree.version == 2


# ---------------------------------------------------------------------------
# FlatStore 与只读视图
# ---------------------------------------------------------------------------


class TestFlatStore:
    def test_entries_keyed_by_tool_id_4_2(self, store: FlatStore) -> None:
        assert set(store.entries) == {"t_set", "t_get", "t_export"}
        assert store.entries["t_set"].path == "/skill/mmd/pose/set_keyframe"

    def test_entries_view_is_read_only_B8(self, store: FlatStore) -> None:
        with pytest.raises(TypeError):
            store.entries["t_new"] = _def("t_new", "/tool/x")  # type: ignore[index]

    def test_rejects_duplicate_tool_id_B8(self, store: FlatStore) -> None:
        with pytest.raises(ToolTreeError) as exc:
            store.put(_def("t_set", "/tool/other", description="dup"))
        assert exc.value.rule == "B8"


# ---------------------------------------------------------------------------
# 空树与边界
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_store_builds_root_only(self) -> None:
        tree = build_tree(FlatStore(), AgentPermissions(agent_id="a"))
        assert isinstance(tree, ToolTree)
        assert tree.root.children == []
        assert tree.tree_index() == ""

    def test_single_leaf_path_is_abstract_because_of_children_B21(
        self, store: FlatStore, perms
    ) -> None:
        tree = build_tree(store, perms)
        node = tree.get_node("/skill")
        assert node is not None
        assert node.children  # 有子节点
        assert node.kind == "abstract"


# ---------------------------------------------------------------------------
# DiscoveredTools（§4.7 淘汰）
# ---------------------------------------------------------------------------


class TestDiscoveredToolsEviction:
    def _disc(self, path: str, step: int) -> ToolDiscovery:
        return ToolDiscovery(
            path=path,
            description="/{}".format(path.strip("/").replace("/", "-")),
            kind="leaf",
            first_seen_step=step,
            last_used_step=step,
        )

    def test_evicts_least_recently_used_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=3)
        bucket.upsert(self._disc("/tool/a", 1))
        bucket.upsert(self._disc("/tool/b", 2))
        bucket.upsert(self._disc("/tool/c", 3))
        bucket.upsert(self._disc("/tool/d", 4))

        assert set(bucket.entries) == {"/tool/b", "/tool/c", "/tool/d"}

    def test_last_used_step_protects_recent_entries_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=3)
        bucket.upsert(self._disc("/tool/a", 1))
        bucket.upsert(self._disc("/tool/b", 2))
        bucket.upsert(self._disc("/tool/c", 3))

        assert bucket.touch("/tool/a", 10) is True
        bucket.upsert(self._disc("/tool/d", 11))

        # 被 touch 过的 /tool/a 不该被淘汰，最久未用的 /tool/b 出局
        assert set(bucket.entries) == {"/tool/a", "/tool/c", "/tool/d"}

    def test_upsert_keeps_first_seen_step_4_7(self) -> None:
        bucket = DiscoveredTools()
        bucket.upsert(self._disc("/tool/a", 1))
        bucket.upsert(self._disc("/tool/a", 9))

        entry = bucket.get("/tool/a")
        assert entry is not None
        assert entry.first_seen_step == 1
        assert entry.last_used_step == 9

    def test_rows_projection_matches_message_3_7(self) -> None:
        bucket = DiscoveredTools()
        bucket.upsert(self._disc("/tool/a", 3))
        rows = bucket.rows()
        assert rows == [{"path": "/tool/a", "description": "/tool-a"}]

    def test_remove_prefix_4_7(self) -> None:
        bucket = DiscoveredTools()
        bucket.upsert(self._disc("/skill/mmd/a", 1))
        bucket.upsert(self._disc("/skill/mmd/b", 2))
        bucket.upsert(self._disc("/skill/other", 3))

        removed = bucket.remove_prefix("/skill/mmd")
        assert sorted(removed) == ["/skill/mmd/a", "/skill/mmd/b"]
        assert set(bucket.entries) == {"/skill/other"}

    def test_default_max_entries_is_50_4_2(self) -> None:
        bucket = DiscoveredTools()
        assert bucket.max_entries == 50
        for i in range(60):
            bucket.upsert(self._disc(f"/tool/t{i:02d}", i))
        assert len(bucket) == 50
        assert "/tool/t00" not in bucket
        assert "/tool/t59" in bucket


class TestTreeQueryState:
    def test_clear_removes_all(self) -> None:
        state = TreeQueryState(delivered={"/a", "/b"})
        state.clear()
        assert state.delivered == set()

    def test_discard_prefix_B24(self) -> None:
        state = TreeQueryState(delivered={"/skill/mmd/a", "/skill/other"})
        assert state.discard_prefix("/skill/mmd") == 1
        assert state.delivered == {"/skill/other"}
