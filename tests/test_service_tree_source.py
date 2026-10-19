"""工具树数据源回归测试。

历史问题：``OrangeService._load_capability`` 写死 ``FlatStore()`` 空存储，
从不读 ``.agent/capabilities.json``。结果 CLI 看到空树，而 Web 服务按清单
装载了实际工具——两边观测到的能力面完全不一致。
"""

from __future__ import annotations

from typing import Never

from orange.capabilities import (
    CapabilityManifest,
    SkillDef,
    SourceSpec,
    _build_external_tool,
    save_manifest,
)
from orange.service import build_service
from orange.tools_builtin import DEFAULT_PATHS

_EXT_SOURCE = """\
def read(path: str = ".") -> str:
    return f"read {path}"


def echo(text: str = "") -> str:
    return text
"""


def _write_manifest(workdir) -> None:
    """按清单装载两个外来工具 + 一个 skill。

    使用外来 Python 工具而不是内置工具：本部署不随包发布内置工具，但「工具树
    必须来自清单」这一回归点与工具从哪来无关。
    """
    root = workdir / "ext"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tool.py").write_text(_EXT_SOURCE, encoding="utf-8")

    manifest = CapabilityManifest()
    for path, callable_name in (
        ("/tool/os/file/read", "read"),
        ("/tool/os/echo", "echo"),
    ):
        source = SourceSpec(kind="python", target="./ext/tool.py", callable=callable_name)
        entry = {
            "path": path,
            "description": f"testing {path}",
            "caps": ["read"],
            "schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "source": {"kind": "python", "target": "./ext/tool.py", "callable": callable_name},
        }
        manifest.add_tool(_build_external_tool(path, entry, source, str(workdir)))
        manifest.sources[path] = source
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


def test_service_seeds_default_tools_without_manifest(tmp_path) -> None:
    """无清单时自动播种默认工具，而不是给一个空树。

    全新安装没有 capabilities.json。若退回空树，模型将无任何工具可调，
    产品开箱即不可用——所以这里必须播种。
    """
    service = build_service(tmp_path, load_model=False)
    try:
        assert service.tree is not None
        assert service.flat_store is not None
        active = service.flat_store.active_paths()
        assert set(active) == set(DEFAULT_PATHS)
        assert len(active) == 4
        # 播种成功后不应再有 capability 问题
        assert not any("capability" in item for item in service.problems)
    finally:
        service.close()


def test_seeding_does_not_overwrite_existing_manifest(tmp_path) -> None:
    """已有清单时不得被默认工具覆盖。"""
    _write_manifest(tmp_path)
    service = build_service(tmp_path, load_model=False)
    try:
        assert service.flat_store is not None
        # 仍是手写清单的 3 个条目，没有掺入默认 4 个
        assert len(service.flat_store.active_paths()) == 3
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
        def chat(self, *args, **kwargs) -> Never:  # noqa: ANN002, ANN003
            raise AssertionError("proxy should not be used")

    manager = SessionManager(
        str(tmp_path), _NullProxy(), default_model="x", capacity=1024, max_sessions=1
    )
    web_paths = sorted(manager.tools.store.active_paths())

    assert cli_paths == web_paths
