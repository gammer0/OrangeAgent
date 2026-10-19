"""pytest 公共配置。

本机 ``C:\\Users\\USER\\AppData\\Local\\Temp\\pytest-of-USER`` 不可访问
（WinError 5），因此把 ``tmp_path`` 的基准目录重定向到仓库内
``.pytest_tmp``，避免测试依赖系统临时目录权限。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_BASETEMP = Path(__file__).resolve().parent.parent / ".pytest_tmp"


@pytest.fixture(scope="session")
def _tmp_root() -> Path:
    _BASETEMP.mkdir(parents=True, exist_ok=True)
    return _BASETEMP


@pytest.fixture
def tmp_path(_tmp_root: Path, request: pytest.FixtureRequest) -> Path:
    """覆盖内建 ``tmp_path``：每个用例一个独立目录，位于仓库内。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.node.nodeid)
    path = _tmp_root / safe
    for child in path.glob("*"):  # 同一用例名重复运行时保持干净
        if child.is_dir():
            import shutil

            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def _isolated_harness_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把数据目录隔离到用例临时目录。

    生产环境的数据目录在用户系统目录（``%APPDATA%\\Harness``），测试绝不能
    写进去：否则用例之间互相污染，且会在开发者机器上留下垃圾。
    """
    home = tmp_path / "harness_home"
    monkeypatch.setenv("HARNESS_HOME", str(home))
    return home
