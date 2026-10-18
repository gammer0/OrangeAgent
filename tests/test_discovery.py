"""发现清单与树查询会话状态测试（§4.6 / §4.7，B7 / B14）。

重点验证：

* schema 只下发一次（B7）；
* ``delivered`` 判定与写入、``DiscoveredTools`` 读写同锁（B14）；
* 压缩时两者同步清空；
* 淘汰按 ``last_used_step`` 升序（§4.7）。
"""

from __future__ import annotations

import threading

import pytest

from harness.discovery import (
    DEFAULT_MAX_DISCOVERED,
    DiscoveredTools,
    ToolDiscovery,
    TreeQueryState,
    TreeSessionState,
)
from harness.permission import AgentPermissions, Permission
from harness.tool_tree import EMPTY_SCHEMA, FlatStore, ToolDef, build_tree, tree_query


def _store() -> FlatStore:
    return FlatStore(
        [
            ToolDef(
                tool_id="t_pose",
                path="/skill/mmd/pose",
                schema=dict(EMPTY_SCHEMA),
                caps=["read"],
                permission=Permission(src="skill", caps=["read"]),
                description="姿态与骨骼调整",
            ),
            ToolDef(
                tool_id="t_render",
                path="/skill/mmd/render",
                schema=dict(EMPTY_SCHEMA),
                caps=["read"],
                permission=Permission(src="skill", caps=["read"]),
                description="渲染输出",
            ),
            ToolDef(
                tool_id="t_set",
                path="/skill/mmd/pose/set_keyframe",
                schema={"type": "object", "properties": {"frame": {"type": "integer"}}},
                caps=["write"],
                permission=Permission(src="skill", caps=["write"]),
                description="设置关键帧",
            ),
        ]
    )


@pytest.fixture
def tree():
    return build_tree(_store(), AgentPermissions(agent_id="a"))


def _record(path: str, step: int, kind: str = "leaf") -> ToolDiscovery:
    return ToolDiscovery(
        path=path,
        description=path.strip("/").replace("/", "-"),
        kind=kind,  # type: ignore[arg-type]
        first_seen_step=step,
        last_used_step=step,
    )


# ---------------------------------------------------------------------------
# ToolDiscovery 基本约束
# ---------------------------------------------------------------------------


class TestToolDiscoveryFields:
    def test_rejects_relative_path(self) -> None:
        with pytest.raises(ValueError):
            _record("skill/mmd", 1)

    def test_rejects_multiline_description(self) -> None:
        with pytest.raises(ValueError):
            ToolDiscovery(
                path="/skill/a",
                description="a\nb",
                kind="leaf",
                first_seen_step=1,
                last_used_step=1,
            )

    def test_description_truncated_to_80_4_2(self) -> None:
        record = ToolDiscovery(
            path="/skill/a",
            description="x" * 100,
            kind="leaf",
            first_seen_step=1,
            last_used_step=1,
        )
        assert len(record.description) == 80
        assert record.description.endswith("…")

    def test_touch_updates_last_used(self) -> None:
        record = _record("/skill/a", 1)
        record.touch(9)
        assert record.first_seen_step == 1
        assert record.last_used_step == 9


# ---------------------------------------------------------------------------
# 清单容量与淘汰（§4.7）
# ---------------------------------------------------------------------------


class TestEvictionByLastUsedStep:
    def test_evicts_ascending_by_last_used_step_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=2)
        bucket.upsert(_record("/skill/a", 1))
        bucket.upsert(_record("/skill/b", 5))
        bucket.upsert(_record("/skill/c", 3))

        # a(1) 最小被淘汰，c(3) 保留
        assert set(bucket.entries) == {"/skill/b", "/skill/c"}

    def test_new_entry_is_kept_even_if_old_first_seen_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=1)
        bucket.upsert(_record("/skill/a", 1))
        bucket.upsert(_record("/skill/b", 2))
        assert set(bucket.entries) == {"/skill/b"}

    def test_touch_promotes_entry_before_eviction_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=2)
        bucket.upsert(_record("/skill/a", 1))
        bucket.upsert(_record("/skill/b", 2))
        bucket.touch("/skill/a", 99)
        bucket.upsert(_record("/skill/c", 100))

        assert set(bucket.entries) == {"/skill/a", "/skill/c"}

    def test_upsert_existing_does_not_evict_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=2)
        bucket.upsert(_record("/skill/a", 1))
        bucket.upsert(_record("/skill/b", 2))
        bucket.upsert(_record("/skill/a", 3))

        assert len(bucket) == 2
        assert set(bucket.entries) == {"/skill/a", "/skill/b"}

    def test_max_entries_default_is_50_4_2(self) -> None:
        assert DiscoveredTools().max_entries == DEFAULT_MAX_DISCOVERED == 50

    def test_overflow_keeps_exactly_max_entries_4_7(self) -> None:
        bucket = DiscoveredTools(max_entries=5)
        for i in range(20):
            bucket.upsert(_record(f"/skill/t{i:02d}", i))
        assert len(bucket) == 5
        assert set(bucket.entries) == {f"/skill/t{i:02d}" for i in range(15, 20)}


# ---------------------------------------------------------------------------
# schema 只下发一次（B7）
# ---------------------------------------------------------------------------


class TestSchemaDeliveredOnce:
    def test_first_query_includes_schema_B7(self, tree) -> None:
        state = TreeSessionState()
        result = tree_query("/skill/mmd/pose", tree=tree, state=state, step=1)
        assert [c.get("schema") for c in result["children"]] == [
            {"type": "object", "properties": {"frame": {"type": "integer"}}}
        ]

    def test_second_query_omits_schema_B7(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd/pose", tree=tree, state=state, step=1)
        result = tree_query("/skill/mmd/pose", tree=tree, state=state, step=2)
        assert "schema" not in result["children"][0]

    def test_omitted_payload_keeps_required_keys_B7(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd/pose", tree=tree, state=state, step=1)
        child = tree_query("/skill/mmd/pose", tree=tree, state=state, step=2)["children"][0]
        assert set(child) == {"path", "name", "kind", "description"}

    def test_unqueried_sibling_still_gets_schema_B7(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd/pose", tree=tree, state=state, step=1)
        render = tree_query("/skill/mmd", tree=tree, state=state, step=2)
        paths = {c["path"]: c for c in render["children"]}
        # /skill/mmd/render 尚未下发过 schema
        assert "schema" in paths["/skill/mmd/render"]
        # /skill/mmd/pose 已下发过
        assert "schema" not in paths["/skill/mmd/pose"]


# ---------------------------------------------------------------------------
# 同锁（B14）
# ---------------------------------------------------------------------------


class TestSameLock:
    def test_state_and_discovered_are_behind_one_lock_B14(self) -> None:
        state = TreeSessionState()
        result = tree_query(
            "/skill/mmd",
            tree=build_tree(_store(), AgentPermissions(agent_id="a")),
            state=state,
            step=1,
        )

        # 同一次调用同时影响 delivered 与 discovered
        assert len(result["children"]) == 2
        assert state.delivered == {"/skill/mmd/pose", "/skill/mmd/render"}
        assert set(state.discovered.entries) == {"/skill/mmd/pose", "/skill/mmd/render"}

    def test_concurrent_queries_single_schema_delivery_B14(self, tree) -> None:
        state = TreeSessionState()
        n = 16
        barrier = threading.Barrier(n)
        payloads: list[dict] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            out = tree_query("/skill/mmd", tree=tree, state=state, step=1)
            with lock:
                payloads.append(out)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(payloads) == n
        full = [p for p in payloads if all("schema" in c for c in p["children"])]
        assert len(full) == 1  # B14：只有一次能看到 schema

    def test_concurrent_discovery_no_lost_update_B14(self, tree) -> None:
        """并发写清单不丢记录（同锁保证）。"""
        state = TreeSessionState()
        paths = ["/skill/mmd/pose", "/skill/mmd/render"]
        barrier = threading.Barrier(20)

        def worker(i: int) -> None:
            barrier.wait()
            tree_query("/skill/mmd", tree=tree, state=state, step=i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert set(state.discovered.entries) == set(paths)
        # last_used_step 为并发查询的最大步号
        assert state.discovered.get(paths[0]).last_used_step == 19  # type: ignore[union-attr]

    def test_record_use_updates_under_lock_B14(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        assert state.record_use("/skill/mmd/pose", 8) is True
        assert state.record_use("/skill/unknown", 8) is False
        assert state.discovered.get("/skill/mmd/pose").last_used_step == 8  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 压缩清空（§4.6 / §4.7）
# ---------------------------------------------------------------------------


class TestCompactionReset:
    def test_reset_clears_both_stores_atomically_4_7(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        assert state.delivered and state.discovered

        state.reset()
        assert state.delivered == frozenset()
        assert len(state.discovered) == 0

    def test_after_reset_schema_is_delivered_again_4_6(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        tree_query("/skill/mmd", tree=tree, state=state, step=2)
        state.reset()

        result = tree_query("/skill/mmd", tree=tree, state=state, step=3)
        assert all("schema" in c for c in result["children"])

    def test_reset_is_idempotent(self, tree) -> None:
        state = TreeSessionState()
        state.reset()
        state.reset()
        assert state.delivered == frozenset()


# ---------------------------------------------------------------------------
# 清单投影（消息 §3.7）
# ---------------------------------------------------------------------------


class TestProjection:
    def test_discovered_rows_are_path_and_description_only_3_7(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        rows = state.discovered_rows()
        assert sorted(rows, key=lambda r: r["path"]) == [
            {"path": "/skill/mmd/pose", "description": "姿态与骨骼调整"},
            {"path": "/skill/mmd/render", "description": "渲染输出"},
        ]

    def test_projection_is_usable_by_to_provider_3_7(self, tree) -> None:
        from harness.message import SystemMsg, UserMsg, to_provider

        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)
        messages = [SystemMsg(content="sys"), UserMsg(content="hi")]
        projected = to_provider(messages, state.discovered_rows())

        assert projected[-1]["role"] == "system"
        assert "/skill/mmd/pose — 姿态与骨骼调整" in projected[-1]["content"]
        # A12：注入不写入 messages
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# 卸载清理（B24）
# ---------------------------------------------------------------------------


class TestPurgeOnUnmount:
    def test_purge_prefix_clears_delivered_and_discovered_B24(self, tree) -> None:
        state = TreeSessionState()
        tree_query("/skill/mmd", tree=tree, state=state, step=1)

        state.purge_prefix("/skill/mmd/render")
        assert "/skill/mmd/render" not in state.delivered
        assert "/skill/mmd/pose" in state.delivered
        assert state.discovered.get("/skill/mmd/render") is None
        assert state.discovered.get("/skill/mmd/pose") is not None

    def test_purge_exact_path_B24(self) -> None:
        state = TreeSessionState(state=TreeQueryState(delivered={"/a", "/a/b"}))
        state.purge_prefix("/a")
        assert state.delivered == frozenset()


# ---------------------------------------------------------------------------
# TreeQueryState 独立行为
# ---------------------------------------------------------------------------


class TestTreeQueryStateStandalone:
    def test_delivered_starts_empty_4_2(self) -> None:
        assert TreeQueryState().delivered == set()

    def test_mark_is_additive(self) -> None:
        state = TreeQueryState()
        state.mark("/a")
        state.mark("/a")
        assert state.delivered == {"/a"}
        assert len(state) == 1
