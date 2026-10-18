"""流体仿真验收工具：物理约束的客观检查。

设计原则：不 import pygame、不需要窗口，直接驱动 ``fluid.core.ParticleSystem``
的无头接口，因此可以在 CI 里跑。

检查项对应 ``.agent/skills/fluid-particle-sim/SKILL.md`` 的「物理要求」：
1. 不可压缩性（平均间距 >= 1.6x 半径）
2. 质量守恒（粒子数不变）
3. 边界与固体（越界数为 0）
4. 重力与静置（动能下降 >= 60%）
5. 无 NaN / 无数值爆炸
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

__all__ = ["fluid_check", "fluid_probe"]

#: 探测脚本：在子进程里导入被测项目并输出 JSON。
_PROBE = r'''
import importlib, json, math, os, sys

sys.path.insert(0, os.getcwd())
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pkg = sys.argv[1] if len(sys.argv) > 1 else "fluid"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500

CANDIDATES = [
    (f"{pkg}.core", "ParticleSystem"),
    (f"{pkg}", "ParticleSystem"),
]

cls = None
used = None
for module_name, cls_name in CANDIDATES:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        continue
    candidate = getattr(module, cls_name, None)
    if candidate is not None:
        cls = candidate
        used = f"{module_name}.{cls_name}"
        break

if cls is None:
    print(json.dumps({"ok": False, "reason": "找不到 ParticleSystem"}, ensure_ascii=False))
    raise SystemExit(3)

W, H = 400, 300
RADIUS = 6.0

def make():
    sysm = cls(W, H, seed=7)
    sysm.add_solid((150.0, 200.0, 120.0, 12.0))   # 一块地板状固体
    sysm.add_water(200.0, 80.0, 40.0, 300)
    return sysm

def fmt(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None

result = {
    "ok": True,
    "class": used,
    "world": [W, H],
    "radius": RADIUS,
    "records": 0,
    "errors": [],
    "nan": 0,
    "out_of_bounds": 0,
    "out_of_solid": 0,
    "min_gap": None,
    "ke_early": None,
    "ke_late": None,
    "mass_start": None,
    "mass_end": None,
    "count_start": None,
    "count_end": None,
}

try:
    system = make()
    particles = system.particles
    result["count_start"] = len(particles)
    result["mass_start"] = fmt(system.total_mass)
    result["ke_early"] = fmt(system.kinetic_energy)

    dt = 1.0 / 60.0
    early_done = False
    for index in range(steps):
        system.step(dt)
        result["records"] += 1
        if index == 49:
            result["ke_early"] = fmt(system.kinetic_energy)
            early_done = True
        if index == 50 and not early_done:
            result["ke_early"] = fmt(system.kinetic_energy)

    particles = system.particles
    result["count_end"] = len(particles)
    result["mass_end"] = fmt(system.total_mass)
    result["ke_late"] = fmt(system.kinetic_energy)

    solids = list(system.solids)

    for (x, y, vx, vy) in particles:
        if not all(map(math.isfinite, (x, y, vx, vy))):
            result["nan"] += 1
            continue
        if not (0.0 <= x <= W and 0.0 <= y <= H):
            result["out_of_bounds"] += 1
        for (sx, sy, sw, sh) in solids:
            if sx < x < sx + sw and sy < y < sy + sh:
                result["out_of_solid"] += 1
                break

    # 邻域最小间距（抽样 80 个粒子，避免 O(n^2) 太慢）
    sample = particles[:80]
    gaps = []
    for i in range(len(sample)):
        ax, ay = sample[i][0], sample[i][1]
        best = None
        for j in range(len(sample)):
            if i == j:
                continue
            bx, by = sample[j][0], sample[j][1]
            d = math.hypot(ax - bx, ay - by)
            if best is None or d < best:
                best = d
        if best is not None:
            gaps.append(best)
    result["min_gap"] = min(gaps) if gaps else None
    result["mean_gap"] = round(sum(gaps) / len(gaps), 4) if gaps else None

except Exception as exc:
    result["ok"] = False
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    raise SystemExit(4) from exc

print(json.dumps(result, ensure_ascii=False, default=str))
'''

#: 结构检查：不运行仿真，只看文件与 API 形状。
_STRUCT = r'''
import ast, json, os, sys

sys.path.insert(0, os.getcwd())
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pkg = sys.argv[1] if len(sys.argv) > 1 else "fluid"
base = os.path.join(os.getcwd(), *pkg.split("."))
report = {"ok": True, "files": [], "problems": [], "notes": []}

if not os.path.isdir(base):
    report["ok"] = False
    report["problems"].append(f"目录不存在: {pkg}")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(3)

py_files = sorted(
    os.path.relpath(os.path.join(root, name), base)
    for root, _dirs, names in os.walk(base)
    for name in names
    if name.endswith(".py") and "__pycache__" not in root
)
report["files"] = py_files

needed = ["main.py", "core.py", "config.py", "README.md"]
for name in needed:
    path = os.path.join(base, name)
    if not os.path.exists(path):
        report["problems"].append(f"缺少 {name}")

core_path = os.path.join(base, "core.py")
if os.path.isfile(core_path):
    source = open(core_path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report["problems"].append(f"core.py:{exc.lineno} 语法错误 {exc.msg}")
        tree = None
    if tree is not None:
        # ponytail: 只看真实 import 语句。之前用子串匹配，把文档字符串里的
        # "不 import pygame" 也判成了违规。
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        if "pygame" in imported:
            report["problems"].append("core.py 不得 import pygame（物理层必须可无头驱动）")

        classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        if "ParticleSystem" not in classes:
            report["problems"].append("core.py 缺少 class ParticleSystem")
        else:
            methods = {
                n.name for n in classes["ParticleSystem"].body if isinstance(n, ast.FunctionDef)
            }
            for name in ("add_water", "add_solid", "step"):
                if name not in methods:
                    report["problems"].append(f"ParticleSystem 缺少方法 {name}")
            properties = {
                n.name
                for n in classes["ParticleSystem"].body
                if isinstance(n, ast.FunctionDef) and any(
                    getattr(d, "id", "") == "property" for d in n.decorator_list
                )
            }
            for name in ("particles", "solids", "total_mass", "kinetic_energy"):
                if name not in properties:
                    report["notes"].append(f"建议提供只读属性 {name}")

readme = os.path.join(base, "README.md")
if os.path.isfile(readme):
    text = open(readme, encoding="utf-8", errors="replace").read()
    for keyword in ("操作", "物理"):
        if keyword not in text:
            report["notes"].append(f"README 建议补充「{keyword}」说明")

for rel in py_files:
    path = os.path.join(base, rel)
    source = open(path, encoding="utf-8", errors="replace").read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        report["problems"].append(f"{rel}:{exc.lineno} 语法错误 {exc.msg}")
    if "TODO" in source or "FIXME" in source:
        report["notes"].append(f"{rel}: 仍有 TODO/FIXME")

print(json.dumps(report, ensure_ascii=False))
'''


def _target(workdir: str, path: str) -> tuple[str, str]:
    """返回 ``(cwd, 传给探测脚本的路径参数)``。

    ponytail: 一律从 ``workdir`` 运行并传相对路径。之前的实现会在非包布局时
    切到子目录里再找 ``<子目录名>/``,必然找不到——探测脚本自己会拼
    ``os.getcwd()/<path>``,从根目录跑才是唯一自洽的口径。
    """
    return str(Path(workdir)), path


def _launch(
    code: str,
    workdir: str,
    path: str,
    args: list[str],
    *,
    timeout: int,
    with_exit: bool = False,
) -> str:
    """在子进程执行探测脚本。"""
    import os

    cwd, name = _target(workdir, path)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPATH": workdir}
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code, name, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"探测超时（>{timeout}s）") from exc
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return f"退出码 {proc.returncode}\n{output}" if with_exit else output


def fluid_check(workdir: str, path: str = "fluid") -> str:
    """结构验收：文件、API 形状、core.py 不依赖 pygame。"""
    return _launch(_STRUCT, workdir, path, [], timeout=30, with_exit=True)


def fluid_probe(workdir: str, path: str = "fluid", steps: int = 500) -> str:
    """物理验收：跑 ``steps`` 步并检查守恒、边界、静置。"""
    steps = max(1, min(int(steps), 3000))
    return _launch(_PROBE, workdir, path, [str(steps)], timeout=180, with_exit=True)
