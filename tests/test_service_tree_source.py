"""工具树数据源回归测试。

历史问题：``HarnessService._load_capability`` 写死 ``FlatStore()`` 空存储，
从不读 ``.agent/capabilities.json``。结果 CLI 看到空树，而 Web 服务按清单
装载了实际工具——两边观测到的能力面完全不一致。
"""

from __future__ import annotations

import json

from harness.capabilities import SkillDef, build_builtin_tool, save_manifest
from harness.capabilities import CapabilityManifest
from harness.service import build_service


def _write_manifest(workdir) -> None:
    manifest = CapabilityManifest()
    manifest.add_tool(build_builtin_tool("/tool/os/file/read", str(workdir)))
    manifest.add_tool(build_builtin_tool("/tool/os/echo", str(workdir)))
    manifest.add_skill(SkillDef(name="demo", description="演示技能"))
    save_manifest(manifest, workdir)


def test_service_tree_reflects_manifest(tmp_path) -> None:
    """有清单时，service 的工具树必须装载清单内容。"""
    _write_manifest(tmp_path)
    service = build_service(tmp_path, load_model=False)
    try:
        assert service.tree is not None
        paths = set(service.tree.paths)
        assert "/tool/os/file/read" in paths
        assert "/tool/os/echo" in paths
        assert "/skill/demo" in paths
        # 中间节点由 build_tree 合成
        assert "/tool" in paths
        assert "/tool/os" in paths
    finally:
        service.close()


def test_service_tree_has_expected_leaf_count(tmp_path) -> None:
    """叶子数应等于清单条目数（2 工具 + 1 skill）。"""
    _write_manifest(tmp_path)
    service = build_service(tmp_path, load_model=False)
    try:
        assert service.flat_store is not None
        assert len(service.flat_store.active_paths()) == 3
    finally:
        service.close()


def test_service_degrades_to_empty_tree_without_manifest(tmp_path) -> None:
    """无清单时退回空树，而不是崩溃（保守降级）。"""
    service = build_service(tmp_path, load_model=False)
    try:
        assert service.tree is not None
        assert service.flat_store is not None
        assert len(service.flat_store.active_paths()) == 0
        # 记录原因，便于运维发现
        assert any("capability" in item for item in service.problems)
    finally:
        service.close()


def test_service_tree_matches_web_loader(tmp_path) -> None:
    """CLI 与 Web 必须看到同一套工具（同一数据源）。"""
    _write_manifest(tmp_path)

    service = build_service(tmp_path, load_model=False)
    try:
        cli_paths = sorted(service.flat_store.active_paths()) if service.flat_store else []
    finally:
        service.close()

    from apps.server.sessions import SessionManager

    class _NullProxy:
        def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("proxy should not be used")

    manager = SessionManager(
        str(tmp_path), _NullProxy(), default_model="x", capacity=1024, max_sessions=1
    )
    web_paths = sorted(manager.tools.store.active_paths())

    assert cli_paths == web_paths
