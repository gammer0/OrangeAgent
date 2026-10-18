"""pytest 公共配置。

本机 ``C:\\Users\\USER\\AppData\\Local\\Temp\\pytest-of-USER`` 不可访问
（WinError 5），因此把 ``tmp_path`` 的基准目录重定向到仓库内
``.pytest_tmp``，避免测试依赖系统临时目录权限。
"""

from __future__ import annotations

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
