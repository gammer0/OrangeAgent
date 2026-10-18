"""内置工具实现：agent 实际可执行的原子能力。

设计原则：

* **单一职责**：每个函数只做一件事，返回字符串（工具输出统一为文本）。
* **沙箱边界**：所有文件操作限定在 ``workdir`` 内；越界直接抛异常。
* **超时即失败**：任何执行类工具都在**子进程**里跑，带超时与输出截断，
  避免 agent 写出死循环把主循环拖死（对应 ``execution.py`` 的 3b 超时降级）。
* **不吞异常**：异常向上抛，由执行器统一翻译成 ``ExecResult.error_type``。

工具路径与实现严格一一对应（``_REGISTRY``），未知路径直接拒绝，
因此能力清单里的 ``path`` 无法指向任意代码。
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from harness.permission import CapName, Permission
from harness.tool_tree import ToolDef
from harness.tools_fluid import fluid_check, fluid_probe

__all__ = [
    "TOOL_SPECS",
    "build_tool",
    "known_paths",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_TIMEOUT_S",
]

#: 单个工具输出上限（与 ``truncate.py`` 的 C1 协同，这里先粗截一刀）。
MAX_OUTPUT_CHARS = 8_000

#: 子进程默认超时（秒）。
DEFAULT_TIMEOUT_S = 20


def _resolve(workdir: str, raw: str) -> Path:
    """把用户路径解析到 workdir 内；越界抛 ``ValueError``。"""
    base = Path(workdir).resolve()
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"路径越界（只能在工作目录内操作）: {raw}")
    return target


def _clip(text: str) -> str:
    """截断过长输出，保留头尾用于定位问题。"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = MAX_OUTPUT_CHARS * 3 // 4
    tail = MAX_OUTPUT_CHARS - head
    return f"{text[:head]}\n...（已截断 {len(text) - MAX_OUTPUT_CHARS} 字符）...\n{text[-tail:]}"


def _run(cmd: list[str], cwd: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[int, str]:
    """在子进程运行命令，返回 ``(退出码, 合并输出)``。

    强制 UTF-8 与 ``PYTHONIOENCODING``：Windows 子进程默认 GBK，
    会把项目里的中文输出变成乱码，使 agent 误判失败。
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"命令超时（>{timeout}s）: {' '.join(cmd)}") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, _clip(output.strip())


# ---------------------------------------------------------------------------
# 文件类
# ---------------------------------------------------------------------------


def file_read(workdir: str, path: str, offset: int = 0, limit: int = 120) -> str:
    """读取文本文件；``offset`` / ``limit`` 以行为单位。

    ponytail: 默认 120 行（原 400）——实测一次读整文件会吐出 8k 字符，
    直接吃掉上下文。超长文件请用 offset 续读。
    """
    target = _resolve(workdir, path)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[offset : offset + limit]
    body = "\n".join(f"{offset + i + 1:>4}| {line}" for i, line in enumerate(chunk))
    return _clip(f"[{path}] 共 {len(lines)} 行，显示 {offset + 1}-{offset + len(chunk)}\n{body}")


def file_list(workdir: str, path: str = ".", pattern: str = "*") -> str:
    """列出目录条目（目录在前，按名称排序）。"""
    target = _resolve(workdir, path)
    if not target.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    items = sorted(target.glob(pattern), key=lambda p: (not p.is_dir(), p.name))[:300]
    rows = [f"{'D' if p.is_dir() else 'F'} {p.name}" + (f"  ({p.stat().st_size}B)" if p.is_file() else "")
            for p in items]
    return _clip("\n".join(rows) or "（空目录）")


def file_write(workdir: str, path: str, content: str) -> str:
    """整体写入文件（覆盖），父目录自动创建。"""
    target = _resolve(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}：{len(content)} 字符，{content.count(chr(10)) + 1} 行"


def file_append(workdir: str, path: str, content: str) -> str:
    """追加写入文件（不存在则创建）。"""
    target = _resolve(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return f"已追加 {len(content)} 字符到 {path}"


def os_echo(text: str) -> str:
    """原样返回文本。"""
    return text


# ---------------------------------------------------------------------------
# 网络类
# ---------------------------------------------------------------------------

#: 网络工具默认超时（秒）。比执行类短：HTTP 卡住比命令卡住更常见。
HTTP_TIMEOUT_S = 15

#: 允许的 URL scheme。只支持 http/https，避免 file:// 等意外协议。
_ALLOWED_SCHEMES = ("http://", "https://")


def _check_url(url: str) -> str:
    """校验 URL 协议，越界抛 ``ValueError``。"""
    text = (url or "").strip()
    if not text:
        raise ValueError("URL 不能为空")
    if not text.lower().startswith(_ALLOWED_SCHEMES):
        raise ValueError(f"只支持 http/https: {url}")
    return text


def net_get(url: str, timeout: int = HTTP_TIMEOUT_S) -> str:
    """GET 一个 URL，返回状态码、内容类型与响应体（截断）。"""
    import httpx

    target = _check_url(url)
    limit = max(1, min(int(timeout), 60))
    try:
        response = httpx.get(target, timeout=limit, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"请求超时（{limit}s）: {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"请求失败: {type(exc).__name__}: {exc}") from exc

    body = response.text
    head = (
        f"状态 {response.status_code} {response.reason_phrase}\n"
        f"类型 {response.headers.get('content-type', '未知')}\n"
        f"长度 {len(body)} 字符\n"
        f"---\n"
    )
    return _clip(head + body)


def net_post(
    url: str,
    body: str = "",
    content_type: str = "application/json",
    timeout: int = HTTP_TIMEOUT_S,
) -> str:
    """POST 一个 URL，``body`` 原样发送，返回状态码与响应体（截断）。"""
    import httpx

    target = _check_url(url)
    limit = max(1, min(int(timeout), 60))
    headers = {"Content-Type": content_type or "application/json"}
    try:
        response = httpx.post(
            target, content=body.encode("utf-8"), headers=headers, timeout=limit,
            follow_redirects=True,
        )
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"请求超时（{limit}s）: {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"请求失败: {type(exc).__name__}: {exc}") from exc

    text = response.text
    head = (
        f"状态 {response.status_code} {response.reason_phrase}\n"
        f"类型 {response.headers.get('content-type', '未知')}\n"
        f"长度 {len(text)} 字符\n"
        f"---\n"
    )
    return _clip(head + text)


def net_head(url: str, timeout: int = HTTP_TIMEOUT_S) -> str:
    """HEAD 一个 URL，只返回状态与响应头（用于探测可用性，不下载正文）。"""
    import httpx

    target = _check_url(url)
    limit = max(1, min(int(timeout), 60))
    try:
        response = httpx.head(target, timeout=limit, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"请求超时（{limit}s）: {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"请求失败: {type(exc).__name__}: {exc}") from exc

    lines = [
        f"状态 {response.status_code} {response.reason_phrase}",
        *[f"{key}: {value}" for key, value in sorted(response.headers.items())],
    ]
    return _clip("\n".join(lines))


# ---------------------------------------------------------------------------
# 执行类
# ---------------------------------------------------------------------------


def shell_run(workdir: str, command: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
    """在 workdir 内执行一次命令（不经过 shell 解释器拼接）。"""
    if timeout > 120:
        timeout = 120
    code, output = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        workdir,
        timeout,
    )
    return f"退出码 {code}\n{output}"


def py_check(workdir: str, path: str) -> str:
    """语法检查：编译单个 .py 文件，返回是否通过。"""
    target = _resolve(workdir, path)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    source = target.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(source, filename=str(target))
    except SyntaxError as exc:
        return f"语法错误 {path}:{exc.lineno}:{exc.offset} — {exc.msg}"
    return f"语法 OK：{path}（{len(source.splitlines())} 行）"


def py_run_brief(workdir: str, path: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
    """短时运行脚本并回收输出（用于无交互的自检脚本）。"""
    target = _resolve(workdir, path)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    code, output = _run(
        [sys.executable, "-B", str(target)],
        str(target.parent),
        timeout,
    )
    return f"退出码 {code}\n{output}"


# ---------------------------------------------------------------------------
# 游戏验收类
# ---------------------------------------------------------------------------


def snake_check(workdir: str, path: str = "snake_game") -> str:
    """贪吃蛇验收：检查目录结构、关键文件与常见缺陷。"""
    base = _resolve(workdir, path)
    if not base.is_dir():
        raise NotADirectoryError(f"目录不存在: {path}")
    problems: list[str] = []
    notes: list[str] = []

    pys = sorted(base.rglob("*.py"))
    if not pys:
        problems.append("没有任何 .py 文件")
    entry = base / "main.py"
    if not entry.is_file():
        problems.append("缺少入口 main.py")
    readme = base / "README.md"
    if not readme.is_file():
        notes.append("建议补 README.md（运行方式 + 操作说明）")

    for file in pys:
        source = file.read_text(encoding="utf-8", errors="replace")
        rel = file.relative_to(base).as_posix()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            problems.append(f"{rel}:{exc.lineno} 语法错误 {exc.msg}")
            continue
        if "TODO" in source or "pass  #" in source:
            notes.append(f"{rel}: 仍有 TODO/占位代码")
        if any(isinstance(n, ast.Global) for n in ast.walk(tree)):
            problems.append(f"{rel}: 使用了 global（建议改为类或状态对象）")
        if len(source.splitlines()) > 600:
            notes.append(f"{rel}: {len(source.splitlines())} 行，建议拆分模块")
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        optional = imports & {"pygame", "tkinter", "curses"}
        if optional and rel == "main.py":
            notes.append(f"{rel}: 依赖 {sorted(optional)}，需在 README 说明启动方式")

    report = {
        "目录": str(base),
        "py 文件": [p.relative_to(base).as_posix() for p in pys],
        "问题": problems,
        "建议": notes,
        "结论": "通过" if not problems else "未通过",
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


#: 无头自检驱动：探针脚本，不依赖被测项目的具体 API，只看"能否稳定推进状态"。
#: 以 ``<包名>.<模块>`` 形式导入，兼容项目内部的相对导入。
_DEMO_DRIVER = r'''
import importlib, json, os, sys
sys.path.insert(0, os.getcwd())

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

steps = int(sys.argv[1])
seed = int(sys.argv[2])
package = sys.argv[3] if len(sys.argv) > 3 else ""

import random
random.seed(seed)

BASES = [f"{package}.game", f"{package}.snake", f"{package}.engine",
         "game", "snake", "engine"] if package else ["game", "snake", "engine"]

CANDIDATES = []
for base in BASES:
    for name in ("SnakeGame", "Game"):
        CANDIDATES.append((base, name))

# 如果包里提供了 game 模块，它的父包也应被尝试（snake_game.game）
if package and f"{package}.game" not in BASES:
    CANDIDATES.append((f"{package}.game", "SnakeGame"))

game = None
used = None
for module_name, cls_name in CANDIDATES:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        continue
    cls = getattr(module, cls_name, None)
    if cls is None:
        continue
    for kwargs in ({}, {"width": 20, "height": 20}, {"cols": 20, "rows": 20}):
        try:
            game = cls(**kwargs)
            used = f"{module_name}.{cls_name}"
            break
        except TypeError:
            continue
    if game is not None:
        break

if game is None:
    print(json.dumps({"ok": False, "reason": "未找到可实例化的游戏类",
                      "tried": [f"{m}.{c}" for m, c in CANDIDATES]}, ensure_ascii=False))
    raise SystemExit(3)

MOVES = ["up", "down", "left", "right"]
methods = [name for name in ("step", "tick", "update") if hasattr(game, name)]
if not methods:
    print(json.dumps({"ok": False, "reason": "游戏类缺少 step/tick/update 推进方法",
                      "class": used}, ensure_ascii=False))
    raise SystemExit(4)

move_fn = getattr(game, "move", None) or getattr(game, "set_direction", None) or getattr(game, "change_direction", None)
frames = 0
errors = []
for index in range(steps):
    over = bool(getattr(game, "over", False))
    if move_fn is not None:
        try:
            move_fn(random.choice(MOVES))
        except TypeError:
            move_fn(random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)]))
    try:
        getattr(game, methods[0])()
    except Exception as exc:
        errors.append(f"step {index}: {type(exc).__name__}: {exc}")
        break
    frames += 1
    if over and hasattr(game, "reset"):
        game.reset()

snake = getattr(game, "snake", None)
length = len(snake) if hasattr(snake, "__len__") else None
print(json.dumps({
    "ok": not errors,
    "class": used,
    "frames": frames,
    "moves": methods[0],
    "direction_api": getattr(move_fn, "__name__", None),
    "score": getattr(game, "score", None),
    "length": length,
    "errors": errors,
}, ensure_ascii=False, default=str))
raise SystemExit(0 if not errors else 5)
'''


def snake_demo(workdir: str, path: str = "snake_game", steps: int = 200, seed: int = 7) -> str:
    """无头自检：以固定随机种子推进若干帧，验证核心逻辑不崩溃。

    探针会在 ``<path>/game.py``、``<包名>.game`` 等位置寻找 ``SnakeGame`` / ``Game``，
    因此项目既可以扁平布局也可以包布局。
    """
    base = _resolve(workdir, path)
    if not base.is_dir():
        raise NotADirectoryError(f"目录不存在: {path}")
    # 包布局（含或不含 __init__.py 的命名空间包）从父目录导入 ``<包名>.game``；
    # 扁平布局则从目录内部导入 ``game``。
    package = base.name if base.name.isidentifier() else ""
    code, output = _run(
        [sys.executable, "-B", "-c", _DEMO_DRIVER, str(steps), str(seed), package],
        str(base.parent if package else base),
        DEFAULT_TIMEOUT_S,
    )
    return f"退出码 {code}\n{output}"


# ---------------------------------------------------------------------------
# MCP（filesystem 语义）命名空间
# ---------------------------------------------------------------------------


def mcp_fs_read(workdir: str, path: str) -> str:
    """MCP filesystem read_text_file 的等价实现。"""
    return file_read(workdir, path, limit=2_000)


def mcp_fs_write(workdir: str, path: str, content: str) -> str:
    """MCP filesystem write_file 的等价实现。"""
    return file_write(workdir, path, content)


def mcp_fs_list(workdir: str, path: str = ".") -> str:
    """MCP filesystem list_directory 的等价实现。"""
    return file_list(workdir, path)


# ---------------------------------------------------------------------------
# 规格表
# ---------------------------------------------------------------------------


def _spec(
    path: str,
    fn: Callable[..., Any],
    description: str,
    properties: dict[str, Any],
    required: list[str],
    caps: list[CapName],
    idempotent: bool = False,
    scope: str = "project",
    hidden: tuple[str, ...] = (),
) -> dict[str, Any]:
    """构造一条工具规格。

    Args:
        hidden: 由运行时注入、不暴露给模型的参数名（如 ``workdir``）。
    """
    return {
        "path": path,
        "fn": fn,
        "description": description,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "caps": caps,
        "idempotent": idempotent,
        "scope": scope,
        "hidden": hidden,
    }


_STR = {"type": "string"}
#: 规格表：路径 → 定义。顺序即 CLI 展示顺序。
TOOL_SPECS: dict[str, dict[str, Any]] = {
    s["path"]: s
    for s in [
        _spec("/tool/os/file/read", file_read, "按行读取工作目录内文本文件",
              {"path": _STR, "offset": {"type": "integer", "default": 0},
               "limit": {"type": "integer", "default": 120}}, ["path"], ["read"], True),
        _spec("/tool/os/file/list", file_list, "列出目录条目（文件/目录+大小）",
              {"path": {"type": "string", "default": "."}, "pattern": {"type": "string", "default": "*"}},
              [], ["read"], True),
        _spec("/tool/os/file/write", file_write, "整体写入文件（覆盖），自动建目录",
              {"path": _STR, "content": _STR}, ["path", "content"], ["write"]),
        _spec("/tool/os/file/append", file_append, "追加写入文件",
              {"path": _STR, "content": _STR}, ["path", "content"], ["write"]),
        _spec("/tool/os/echo", os_echo, "原样回显文本，用于连通性测试",
              {"text": _STR}, ["text"], ["read"], True, "session"),
        _spec("/tool/os/shell/run", shell_run, "在子进程中执行一条命令（带超时）",
              {"command": _STR, "timeout": {"type": "integer", "default": DEFAULT_TIMEOUT_S}},
              ["command"], ["exec"]),
        _spec("/tool/code/py/check", py_check, "Python 语法检查，返回错误行号",
              {"path": _STR}, ["path"], ["read"], True),
        _spec("/tool/code/py/run_brief", py_run_brief, "短时运行脚本并返回输出（带超时）",
              {"path": _STR, "timeout": {"type": "integer", "default": DEFAULT_TIMEOUT_S}},
              ["path"], ["exec"]),
        _spec("/tool/game/snake/check", snake_check, "贪吃蛇验收：结构/语法/常见缺陷",
              {"path": {"type": "string", "default": "snake_game"}}, [], ["read"], True),
        _spec("/tool/game/snake/demo", snake_demo, "贪吃蛇无头自检：固定种子跑若干步",
              {"path": {"type": "string", "default": "snake_game"},
               "steps": {"type": "integer", "default": 200},
               "seed": {"type": "integer", "default": 7}},
              [], ["exec"]),
        _spec("/tool/fluid/check", fluid_check, "流体仿真结构验收：文件/API/不依赖 pygame",
              {"path": {"type": "string", "default": "fluid"}}, [], ["read"], True),
        _spec("/tool/fluid/probe", fluid_probe, "流体物理验收：跑 N 步查守恒/边界/静置",
              {"path": {"type": "string", "default": "fluid"},
               "steps": {"type": "integer", "default": 500}},
              [], ["exec"]),
        _spec("/mcp/fs/list", mcp_fs_list, "MCP filesystem：列出目录（URI 语义）",
              {"path": {"type": "string", "default": "."}}, [], ["read"], True),
        _spec("/mcp/fs/read", mcp_fs_read, "MCP filesystem：读取文本文件",
              {"path": _STR}, ["path"], ["read"], True),
        _spec("/mcp/fs/write", mcp_fs_write, "MCP filesystem：写入文本文件",
              {"path": _STR, "content": _STR}, ["path", "content"], ["write"]),
        # ---- 网络访问（caps=net，与文件/执行能力互不重叠）----
        _spec("/tool/net/http/get", net_get, "HTTP GET，返回状态/类型/正文（截断）",
              {"url": _STR, "timeout": {"type": "integer", "default": HTTP_TIMEOUT_S}},
              ["url"], ["net"], True, "global"),
        _spec("/tool/net/http/post", net_post, "HTTP POST，原样发送 body",
              {"url": _STR, "body": {"type": "string", "default": ""},
               "content_type": {"type": "string", "default": "application/json"},
               "timeout": {"type": "integer", "default": HTTP_TIMEOUT_S}},
              ["url"], ["net"], False, "global"),
        _spec("/tool/net/http/head", net_head, "HTTP HEAD，只取状态与响应头",
              {"url": _STR, "timeout": {"type": "integer", "default": HTTP_TIMEOUT_S}},
              ["url"], ["net"], True, "global"),
    ]
}


def known_paths() -> list[str]:
    """规格表中全部内置路径（排序）。"""
    return sorted(TOOL_SPECS)


def build_tool(path: str, workdir: str) -> ToolDef:
    """按路径构造绑定到 ``workdir`` 的可执行 ``ToolDef``。

    绑定只注入**:attr:`_spec` 声明为 ``hidden`` 的参数**，其余参数完全交给模型；
    模型即使传入 ``workdir`` 也会被丢弃，因此工作目录边界不可被参数绕过。

    Raises:
        KeyError: 路径不在规格表中。
    """
    spec = TOOL_SPECS[path]
    fn = spec["fn"]
    hidden = tuple(spec["hidden"])
    params = inspect.signature(fn).parameters
    default_scope = "project"

    def bound(**kwargs: Any) -> str:
        """注入隐藏参数后调用实现。"""
        call_kwargs = {k: v for k, v in kwargs.items() if k not in hidden}
        if "workdir" in hidden or "workdir" in params:
            call_kwargs["workdir"] = workdir
        return str(fn(**call_kwargs))

    caps = list(spec["caps"])
    return ToolDef(
        tool_id="t" + path.replace("/", "_"),
        path=path,
        fn=bound,
        schema=spec["schema"],
        caps=caps,
        idempotent=bool(spec["idempotent"]),
        permission=Permission(src=path.strip("/").split("/")[0], caps=caps),
        side_effect_scope=spec["scope"] or default_scope,
        description=spec["description"],
    )
