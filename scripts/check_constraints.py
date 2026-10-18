"""约束与依赖检查器。

三件事：

1. 依赖白名单：``pyproject.toml`` 只允许设计文档约定的包。
2. 导入边界：``src/harness`` 不得导入 ``tests``，不得使用框架类库。
3. 约束覆盖：设计文档中的约束编号必须在源码或测试中出现。

用法::

    uv run python scripts/check_constraints.py

退出码 0 表示全部通过，1 表示存在违规。
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "harness"
TESTS = ROOT / "tests"
DOCS = ROOT / "docs"
APPS = ROOT / "apps"

ALLOWED_DEPENDENCIES = {
    "httpx",
    "openai",
    "pydantic",
    "python-dotenv",
    "rich",
    # Web 服务层（仅 apps/ 使用，核心 src/harness 仍禁止导入）
    "fastapi",
    "uvicorn",
}

#: 核心编排层禁止的模块。设计原则：核心零 Web 框架、零编排框架。
BANNED_MODULES = {
    "langchain",
    "langchain_core",
    "llama_index",
    "autogen",
    "crewai",
    "fastapi",
    "celery",
    "django",
    "sqlalchemy",
    "uvicorn",
    "starlette",
}

#: 服务层（apps/）允许的 Web 框架；其余禁止项仍然生效。
ALLOWED_WEB_MODULES_IN_APPS = {"fastapi", "starlette", "uvicorn"}

# 约束编号形如 A1 / A0a / G12 / B25 / 3a；只取大写字母前缀的文档约束
CONSTRAINT_PATTERN = re.compile(r"\b([A-K])(\d+)(?![\d])\b")


def _normalise(name: str) -> str:
    return name.strip().split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()


def check_dependencies() -> list[str]:
    """校验依赖白名单。"""
    problems: list[str] = []
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        name = _normalise(dep).lower()
        if name not in ALLOWED_DEPENDENCIES:
            problems.append(f"未授权的依赖: {dep}")
    return problems


def _iter_python_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)


def _scan_imports(file: Path) -> set[str]:
    """返回文件导入的顶层模块名集合。"""
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def check_import_boundaries() -> list[str]:
    """校验导入边界（按目录区分白名单）。

    * ``src/harness``：核心层，``BANNED_MODULES`` 全部禁止，且不得导入 ``tests``。
    * ``apps/``：服务层，允许 FastAPI/Starlette/Uvicorn，其余编排框架仍禁止，
      且不得直接导入 ``tests``。

    Returns:
        违规描述列表；空列表表示通过。
    """
    problems: list[str] = []

    for file in _iter_python_files(SRC):
        for module in _scan_imports(file):
            if module == "tests":
                problems.append(f"{file.relative_to(ROOT)}: 源码不得导入 tests")
            if module in BANNED_MODULES:
                problems.append(
                    f"{file.relative_to(ROOT)}: 核心层禁止导入 {module}"
                    "（Web 服务请放在 apps/）"
                )

    if APPS.is_dir():
        for file in _iter_python_files(APPS):
            for module in _scan_imports(file):
                if module == "tests":
                    problems.append(f"{file.relative_to(ROOT)}: 应用层不得导入 tests")
                if module in ALLOWED_WEB_MODULES_IN_APPS:
                    continue
                if module in BANNED_MODULES:
                    problems.append(f"{file.relative_to(ROOT)}: 应用层禁止导入 {module}")

    return problems


def collect_doc_constraints() -> set[str]:
    """从设计文档收集全部约束编号。"""
    found: set[str] = set()
    for file in DOCS.rglob("*.md"):
        text = file.read_text(encoding="utf-8")
        for match in CONSTRAINT_PATTERN.finditer(text):
            found.add(f"{match.group(1)}{match.group(2)}")
    return found


def collect_code_constraints() -> set[str]:
    """从源码与测试收集被引用到的约束编号。"""
    found: set[str] = set()
    for base in (SRC, TESTS):
        for file in _iter_python_files(base):
            text = file.read_text(encoding="utf-8")
            for match in CONSTRAINT_PATTERN.finditer(text):
                found.add(f"{match.group(1)}{match.group(2)}")
    return found


def check_constraint_coverage(min_ratio: float = 0.0) -> tuple[list[str], float]:
    """统计约束编号覆盖率。

    Args:
        min_ratio: 要求的最低覆盖率，0 表示只报告不拦截。

    Returns:
        (问题列表, 覆盖率)
    """
    doc = collect_doc_constraints()
    code = collect_code_constraints()
    covered = doc & code
    ratio = len(covered) / len(doc) if doc else 1.0

    problems: list[str] = []
    if ratio < min_ratio:
        missing = ", ".join(sorted(doc - code))
        problems.append(f"约束覆盖率 {ratio:.1%} 低于要求 {min_ratio:.0%}；缺失: {missing}")
    return problems, ratio


def main() -> int:
    """执行全部检查并输出报告。"""
    failures: list[str] = []

    dep_problems = check_dependencies()
    failures.extend(dep_problems)

    boundary_problems = check_import_boundaries()
    failures.extend(boundary_problems)

    coverage_problems, ratio = check_constraint_coverage()
    failures.extend(coverage_problems)

    print(f"[deps]       {'OK' if not dep_problems else 'FAIL'}")
    print(f"[boundaries] {'OK' if not boundary_problems else 'FAIL'}")
    print(f"[coverage]   {ratio:.1%} 的文档约束已被源码/测试引用")

    if failures:
        print("\n发现问题：")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n全部检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
