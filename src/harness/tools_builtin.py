"""Builtin tool implementations: the atomic capabilities an agent actually runs.

Design principles:

* **Single responsibility**: each function does one thing and returns a string
  (tool output is always text).
* **Sandbox boundary**: every file operation is confined to ``workdir``; an
  escape raises immediately.
* **Timeout means failure**: every execution-style tool runs in a **child
  process** with a timeout and output clipping, so an infinite loop written by
  the agent cannot stall the main loop (matching the 3b timeout degradation in
  ``execution.py``).
* **Never swallow exceptions**: exceptions propagate and the executor translates
  them uniformly into ``ExecResult.error_type``.

Tool paths map one-to-one onto implementations (``_REGISTRY``) and unknown paths
are rejected, so a ``path`` in the capability manifest can never point at
arbitrary code.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

#: Per-tool output cap (works with C1 in ``truncate.py``; this is a coarse first cut).
MAX_OUTPUT_CHARS = 8_000

#: Default child-process timeout (seconds).
DEFAULT_TIMEOUT_S = 20


def _resolve(workdir: str, raw: str) -> Path:
    """Resolve a user path inside ``workdir``; raises ``ValueError`` on escape."""
    base = Path(workdir).resolve()
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes the working directory: {raw}")
    return target


def _clip(text: str) -> str:
    """Clip oversized output, keeping head and tail so problems remain locatable."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = MAX_OUTPUT_CHARS * 3 // 4
    tail = MAX_OUTPUT_CHARS - head
    return f"{text[:head]}\n...（已截断 {len(text) - MAX_OUTPUT_CHARS} 字符）...\n{text[-tail:]}"


def _run(cmd: list[str], cwd: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[int, str]:
    """Run a command in a child process; returns ``(exit code, combined output)``.

    UTF-8 and ``PYTHONIOENCODING`` are forced: Windows child processes default to
    GBK, which turns the project's Chinese output into mojibake and makes the
    agent misjudge a success as a failure.
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
        raise TimeoutError(f"command timed out (>{timeout}s): {' '.join(cmd)}") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, _clip(output.strip())


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


def file_read(workdir: str, path: str, offset: int = 0, limit: int = 120) -> str:
    """Read a text file; ``offset`` / ``limit`` are in lines.

    Note: the default is 120 lines (was 400) -- we measured that reading a whole
    file at once emitted 8k characters and ate the context outright. Use offset
    to continue reading long files.
    """
    target = _resolve(workdir, path)
    if not target.is_file():
        raise FileNotFoundError(f"file does not exist: {path}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[offset : offset + limit]
    body = "\n".join(f"{offset + i + 1:>4}| {line}" for i, line in enumerate(chunk))
    return _clip(f"[{path}] 共 {len(lines)} 行，显示 {offset + 1}-{offset + len(chunk)}\n{body}")


def file_list(workdir: str, path: str = ".", pattern: str = "*") -> str:
    """List directory entries (directories first, sorted by name)."""
    target = _resolve(workdir, path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    items = sorted(target.glob(pattern), key=lambda p: (not p.is_dir(), p.name))[:300]
    rows = [
        f"{'D' if p.is_dir() else 'F'} {p.name}"
        + (f"  ({p.stat().st_size}B)" if p.is_file() else "")
        for p in items
    ]
    return _clip("\n".join(rows) or "（空目录）")


def file_write(workdir: str, path: str, content: str) -> str:
    """Write a whole file (overwriting); parent directories are created."""
    target = _resolve(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}：{len(content)} 字符，{content.count(chr(10)) + 1} 行"


def file_append(workdir: str, path: str, content: str) -> str:
    """Append to a file (creating it when absent)."""
    target = _resolve(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return f"已追加 {len(content)} 字符到 {path}"


# ---------------------------------------------------------------------------
# Working memory (dedicated read/write; no arbitrary paths)
# ---------------------------------------------------------------------------

#: Working-memory relative path (kept in sync with ``memory.py`` to avoid drift).
_WORK_MEMORY_REL = ".agent/WORK.md"


def _work_memory_file(workdir: str) -> Path:
    """Return the absolute path of the working-memory file (fixed; never model-supplied)."""
    return _resolve(workdir, _WORK_MEMORY_REL)


def memory_work_read(workdir: str, offset: int = 0, limit: int = 400) -> str:
    """Read working memory. The path is fixed, so the model cannot target another file."""
    target = _work_memory_file(workdir)
    if not target.is_file():
        return f"（工作记忆尚不存在：{_WORK_MEMORY_REL}）"
    return file_read(workdir, _WORK_MEMORY_REL, offset=offset, limit=limit)


def memory_work_append(workdir: str, content: str) -> str:
    """Append a block to working memory.

    Append rather than overwrite: working memory is shared across sessions, so
    overwriting would erase what another session wrote.
    """
    if not content.strip():
        raise ValueError("content must not be empty")
    target = _work_memory_file(workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if content.endswith("\n") else "\n"
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(content + suffix)
    return f"已追加 {len(content)} 字符到 {_WORK_MEMORY_REL}（累计 {target.stat().st_size} 字节）"


def memory_work_replace(workdir: str, content: str) -> str:
    """Rewrite working memory entirely (for consolidation).

    Differs from ``file_write``: the path is fixed to working memory and cannot
    write anywhere else.
    """
    target = _work_memory_file(workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = content if content.endswith("\n") else content + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return f"已重写 {_WORK_MEMORY_REL}：{len(text)} 字符"


def os_echo(text: str) -> str:
    """Return the text unchanged."""
    return text


# ---------------------------------------------------------------------------
# Network tools
# ---------------------------------------------------------------------------

#: Default timeout for network tools (seconds). Shorter than execution tools:
#: HTTP hanging is more common than a command hanging.
HTTP_TIMEOUT_S = 15

#: Allowed URL schemes. Only http/https, ruling out accidental protocols like file://.
_ALLOWED_SCHEMES = ("http://", "https://")


def _check_url(url: str) -> str:
    """Validate the URL scheme; raises ``ValueError`` when it is not allowed."""
    text = (url or "").strip()
    if not text:
        raise ValueError("URL must not be empty")
    if not text.lower().startswith(_ALLOWED_SCHEMES):
        raise ValueError(f"only http/https is supported: {url}")
    return text


def net_get(url: str, timeout: int = HTTP_TIMEOUT_S) -> str:
    """GET a URL; returns status code, content type and body (clipped)."""
    import httpx

    target = _check_url(url)
    limit = max(1, min(int(timeout), 60))
    try:
        response = httpx.get(target, timeout=limit, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"request timed out ({limit}s): {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"request failed: {type(exc).__name__}: {exc}") from exc

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
    """POST to a URL sending ``body`` as-is; returns status code and body (clipped)."""
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
        raise TimeoutError(f"request timed out ({limit}s): {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"request failed: {type(exc).__name__}: {exc}") from exc

    text = response.text
    head = (
        f"状态 {response.status_code} {response.reason_phrase}\n"
        f"类型 {response.headers.get('content-type', '未知')}\n"
        f"长度 {len(text)} 字符\n"
        f"---\n"
    )
    return _clip(head + text)


def net_head(url: str, timeout: int = HTTP_TIMEOUT_S) -> str:
    """HEAD a URL; returns status and headers only (probes availability without downloading)."""
    import httpx

    target = _check_url(url)
    limit = max(1, min(int(timeout), 60))
    try:
        response = httpx.head(target, timeout=limit, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"request timed out ({limit}s): {target}") from exc
    except httpx.HTTPError as exc:
        raise ConnectionError(f"request failed: {type(exc).__name__}: {exc}") from exc

    lines = [
        f"状态 {response.status_code} {response.reason_phrase}",
        *[f"{key}: {value}" for key, value in sorted(response.headers.items())],
    ]
    return _clip("\n".join(lines))


# ---------------------------------------------------------------------------
# Execution tools
# ---------------------------------------------------------------------------


def shell_run(workdir: str, command: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
    """Run one command inside ``workdir`` (no shell-interpreter string splicing)."""
    if timeout > 120:
        timeout = 120
    code, output = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        workdir,
        timeout,
    )
    return f"退出码 {code}\n{output}"


def py_check(workdir: str, path: str) -> str:
    """Syntax check: compile a single .py file and report whether it passes."""
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
    """Run a script briefly and collect its output (for non-interactive self-check scripts)."""
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
# Game acceptance
# ---------------------------------------------------------------------------


def snake_check(workdir: str, path: str = "snake_game") -> str:
    """Snake acceptance: checks directory structure, key files and common defects."""
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


#: Headless self-check driver: a probe script that does not depend on the project's
#: concrete API, only on whether state advances reliably.
#: Imported as ``<package>.<module>``, which also supports intra-project relative imports.
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

# If the package provides a game module, its parent package should be tried too
# (snake_game.game)
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


def snake_demo(
    workdir: str, path: str = "snake_game", steps: int = 200, seed: int = 7
) -> str:
    """Headless self-check: advance frames with a fixed seed.

    Verifies the core logic does not crash.

    The probe looks for ``SnakeGame`` / ``Game`` in places such as
    ``<path>/game.py`` and ``<package>.game``, so a project can use either a flat
    layout or a package layout.
    """
    base = _resolve(workdir, path)
    if not base.is_dir():
        raise NotADirectoryError(f"目录不存在: {path}")
    # A package layout (including namespace packages with or without __init__.py)
    # imports ``<package>.game`` from the parent directory; a flat layout imports
    # ``game`` from inside the directory.
    package = base.name if base.name.isidentifier() else ""
    code, output = _run(
        [sys.executable, "-B", "-c", _DEMO_DRIVER, str(steps), str(seed), package],
        str(base.parent if package else base),
        DEFAULT_TIMEOUT_S,
    )
    return f"退出码 {code}\n{output}"


# ---------------------------------------------------------------------------
# MCP (filesystem semantics) namespace
# ---------------------------------------------------------------------------


def mcp_fs_read(workdir: str, path: str) -> str:
    """Equivalent of MCP filesystem read_text_file."""
    return file_read(workdir, path, limit=2_000)


def mcp_fs_write(workdir: str, path: str, content: str) -> str:
    """Equivalent of MCP filesystem write_file."""
    return file_write(workdir, path, content)


def mcp_fs_list(workdir: str, path: str = ".") -> str:
    """Equivalent of MCP filesystem list_directory."""
    return file_list(workdir, path)


# ---------------------------------------------------------------------------
# Specification table
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
    """Build one tool specification.

    Args:
        hidden: Names of parameters injected by the runtime and not exposed to
            the model (e.g. ``workdir``).
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
#: Specification table: path -> definition. Order is the CLI display order.
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
        # ---- network access (caps=net; disjoint from file/exec capabilities) ----
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
        # ---- working memory (fixed path; no arbitrary writes) ----
        _spec("/tool/memory/work/read", memory_work_read, "读取工作记忆 .agent/WORK.md",
              {"offset": {"type": "integer", "default": 0},
               "limit": {"type": "integer", "default": 400}},
              [], ["read"], True),
        _spec("/tool/memory/work/append", memory_work_append, "追加一段到工作记忆（不覆盖）",
              {"content": _STR}, ["content"], ["write"]),
        _spec("/tool/memory/work/replace", memory_work_replace, "整体重写工作记忆（整理用）",
              {"content": _STR}, ["content"], ["write"]),
    ]
}


def known_paths() -> list[str]:
    """All builtin paths in the specification table (sorted)."""
    return sorted(TOOL_SPECS)


def build_tool(path: str, workdir: str) -> ToolDef:
    """Build an executable ``ToolDef`` by path, bound to ``workdir``.

    Binding injects **only the parameters declared ``hidden`` in :attr:`_spec`**;
    every other parameter is left entirely to the model. Even if the model passes
    ``workdir`` it is discarded, so the working-directory boundary cannot be
    bypassed through arguments.

    Raises:
        KeyError: The path is not in the specification table.
    """
    spec = TOOL_SPECS[path]
    fn = spec["fn"]
    hidden = tuple(spec["hidden"])
    params = inspect.signature(fn).parameters
    default_scope = "project"

    def bound(**kwargs: Any) -> str:
        """Inject the hidden parameters and call the implementation."""
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
