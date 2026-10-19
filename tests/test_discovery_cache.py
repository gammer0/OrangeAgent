"""发现缓存表只保存具体工具（叶子），不保存抽象能力。

抽象节点（``/tool``、``/tool/os``、``/skill``、``/mcp`` 等）不是可调用的能力：
调用它只会再往下展开一层。把它们写进缓存表会占用真实工具的容量，并让每次请求
的临时注入多出仍需继续展开的路径。这条规则对 ``/tool``、``/skill``、``/mcp``
三个分支一致生效。
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.server.sessions import SessionRunner

from harness.tool_tree import FlatStore, build_tree


def _runner(tmp_path: Path) -> tuple[SessionRunner, object]:
    from apps.server.sessions import DemoToolRegistry

    tools = DemoToolRegistry(str(tmp_path))
    runner = SessionRunner(
        session_id="cache",
        workdir=str(tmp_path),
        proxy=object(),  # type: ignore[arg-type]
        tools=tools,
        model="fake",
        capacity=10_000,
    )
    return runner, tools


def _descent_payload(path: str, children: list[tuple[str, str]]) -> str:
    """Build the JSON a descent returns: children with path + kind."""
    return json.dumps(
        {
            "path": path,
            "children": [
                {
                    "path": child_path,
                    "name": child_path.rsplit("/", 1)[-1],
                    "kind": kind,
                    "description": f"desc of {child_path}",
                    "schema": {"type": "object", "properties": {}},
                }
                for child_path, kind in children
            ],
        },
        ensure_ascii=False,
    )


class TestLeafOnlyCache:
    def test_abstract_children_are_not_cached(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        payload = _descent_payload(
            "/tool",
            [
                ("/tool/os", "abstract"),
                ("/tool/net", "abstract"),
                ("/tool/os/file/read", "leaf"),
            ],
        )
        runner._on_descent(payload)

        paths = {row["path"] for row in runner.discovered_rows()}
        assert paths == {"/tool/os/file/read"}

    def test_skill_branch_keeps_only_leaf_skills(self, tmp_path: Path) -> None:
        """/skill 的抽象根不缓存，具体 skill 才缓存。"""
        runner, _ = _runner(tmp_path)
        runner._on_descent(
            _descent_payload(
                "/skill",
                [
                    ("/skill/incremental-delivery", "leaf"),
                    ("/skill/nested-group", "abstract"),
                ],
            )
        )
        assert {r["path"] for r in runner.discovered_rows()} == {"/skill/incremental-delivery"}

    def test_mcp_branch_keeps_only_leaf_tools(self, tmp_path: Path) -> None:
        """MCP 也是同样的规则：/mcp 与 /mcp/workspace 都不进表。"""
        runner, _ = _runner(tmp_path)
        runner._on_descent(
            _descent_payload(
                "/mcp",
                [
                    ("/mcp/workspace", "abstract"),
                    ("/mcp/fs/read", "leaf"),
                ],
            )
        )
        assert {r["path"] for r in runner.discovered_rows()} == {"/mcp/fs/read"}

    def test_all_abstract_descent_caches_nothing(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        runner._on_descent(
            _descent_payload(
                "/tool",
                [("/tool/os", "abstract"), ("/tool/net", "abstract"), ("/tool/memory", "abstract")],
            )
        )
        assert runner.discovered_rows() == []

    def test_malformed_payload_is_ignored(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        runner._on_descent("not json")
        runner._on_descent(json.dumps({"path": "/tool"}))
        runner._on_descent(json.dumps({"path": "/tool", "children": "nope"}))
        assert runner.discovered_rows() == []

    def test_cached_entry_kind_is_always_leaf(self, tmp_path: Path) -> None:
        runner, _ = _runner(tmp_path)
        runner._on_descent(_descent_payload("/tool", [("/tool/os/file/read", "leaf")]))
        entry = runner.discovered.get("/tool/os/file/read")
        assert entry is not None
        assert entry.kind == "leaf"


class TestDescentNotPersisted:
    """下钻回合不写入持久历史：只保留真实工具调用与对话。"""

    def _runner_with_store(self, tmp_path: Path) -> tuple[SessionRunner, object]:
        from apps.server.sessions import DemoToolRegistry

        from harness.session_store import SessionStore

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="persist",
            workdir=str(tmp_path),
            proxy=object(),  # type: ignore[arg-type]
            tools=tools,
            model="fake",
            capacity=10_000,
        )
        store = SessionStore(str(tmp_path))
        store.upsert_session(
            __import__("harness.session", fromlist=["SessionRecord"]).SessionRecord(
                session_id="persist",
                working_dir=str(tmp_path),
                created_at=1.0,
                updated_at=1.0,
                status="idle",
            )
        )
        runner.store = store
        runner.persisted_count = len(runner.messages)
        return runner, store

    def test_flush_writes_whatever_history_holds(self, tmp_path: Path) -> None:
        """持久化不再自行过滤下钻：清理发生在 loop 侧，落盘即所见。

        清理时机由 ``AgentLoop._prune_resolved_descents`` 控制（选到具体工具后
        才删），因此到了 flush 阶段，历史里剩下的就是最终轨迹。
        """
        from harness.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

        runner, store = self._runner_with_store(tmp_path)

        descent = ToolCall(id="d1", name="/tool", arguments="{}")
        real = ToolCall(id="r1", name="/tool/os/file/read", arguments='{"path":"a"}')
        runner._descent_call_ids.add("d1")

        runner.messages.append(UserMsg(content="读文件"))
        runner.messages.append(AssistantMsg(content="", tool_calls=[descent]))
        runner.messages.append(ToolMsg(tool_call_id="d1", name="/tool", content="{}"))
        runner.messages.append(AssistantMsg(content="", tool_calls=[real]))
        runner.messages.append(
            ToolMsg(tool_call_id="r1", name="/tool/os/file/read", content="内容")
        )
        runner.messages.append(AssistantMsg(content="完成", tool_calls=[]))
        runner.persist_pending()

        stored = store.load_messages("persist")
        types = [row.type for row in stored]
        names = [row.name for row in stored if row.type == "tool"]
        assert types == ["user", "assistant", "tool", "assistant", "tool", "assistant"], types
        assert names == ["/tool", "/tool/os/file/read"], names

    def test_pruned_history_persists_without_descents(self, tmp_path: Path) -> None:
        """loop 清理后的历史落盘时不含下钻，只留真实调用。"""
        from harness.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

        runner, store = self._runner_with_store(tmp_path)

        descent = ToolCall(id="d1", name="/tool", arguments="{}")
        real = ToolCall(id="r1", name="/tool/os/file/read", arguments='{"path":"a"}')

        # 模拟 loop 已经清理：下钻那一对已从历史移除
        runner.messages.append(UserMsg(content="读文件"))
        runner.messages.append(AssistantMsg(content="", tool_calls=[real]))
        runner.messages.append(
            ToolMsg(tool_call_id="r1", name="/tool/os/file/read", content="内容")
        )
        runner.messages.append(AssistantMsg(content="完成", tool_calls=[]))
        runner.persist_pending()

        stored = store.load_messages("persist")
        assert [row.type for row in stored] == ["user", "assistant", "tool", "assistant"]
        assert [r.name for r in stored if r.type == "tool"] == ["/tool/os/file/read"]


class TestTreeMatchesCacheRule:
    """用一棵自建工具树交叉验证：缓存规则与树的叶子/抽象划分一致。

    不读真实清单：测试把数据目录隔离到临时路径，依赖外部配置会让用例变得
    环境相关。这里自己声明三个分支，正好覆盖 /tool、/skill、/mcp。
    """

    def _tree(self, tmp_path: Path) -> object:
        from harness.capabilities import CapabilityManifest
        from harness.permission import Permission
        from harness.tool_tree import ToolDef

        def leaf(path: str) -> ToolDef:
            return ToolDef(
                tool_id="t" + path.replace("/", "_"),
                path=path,
                fn=lambda **_: "ok",
                caps=["read"],
                permission=Permission(src=path.strip("/").split("/")[0], caps=["read"]),
                description=f"leaf {path}",
            )

        manifest = CapabilityManifest()
        for path in (
            "/tool/os/file/read",
            "/tool/os/file/write",
            "/skill/alpha",
            "/mcp/workspace/list_dir",
        ):
            manifest.add_tool(leaf(path))
        store = FlatStore(manifest.to_tooldefs(str(tmp_path)))
        return build_tree(store, None)

    def test_abstract_paths_are_disjoint_from_leaves(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path)
        abstract = tree.abstract_paths()
        leaves = {node.path for node in tree.visible_nodes() if node.kind == "leaf"}
        assert abstract, "expected some intermediate nodes"
        assert not (abstract & leaves)

    def test_every_branch_has_abstract_root(self, tmp_path: Path) -> None:
        tree = self._tree(tmp_path)
        abstract = tree.abstract_paths()
        for root in ("/tool", "/skill", "/mcp"):
            has_children = any(p.startswith(root + "/") for p in tree.paths)
            assert has_children, f"fixture should contain {root} children"
            assert root in abstract, f"{root} should be a descent point"
