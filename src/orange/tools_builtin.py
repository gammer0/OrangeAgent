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

import inspect
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from orange.permission import CapName, Permission
from orange.tool_tree import ToolDef

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
#:
#: **Release profile**: this distribution ships only workspace file access.
#: Execution (shell / Python), network and MCP tools are deliberately absent --
#: a published agent that can run arbitrary commands on the user's machine is
#: not something we want to hand out. Tree query is not listed here: every
#: ``abstract`` node gets one implicitly (see ``orange.tool_tree``).
TOOL_SPECS: dict[str, dict[str, Any]] = {
    s["path"]: s
    for s in [
        _spec("/tool/os/file/read", file_read, "按行读取工作目录内文本文件",
              {"path": _STR, "offset": {"type": "integer", "default": 0},
               "limit": {"type": "integer", "default": 120}}, ["path"], ["read"], True),
        _spec("/tool/os/file/list", file_list, "列出目录条目（文件/目录+大小）",
              {
                  "path": {"type": "string", "default": "."},
                  "pattern": {"type": "string", "default": "*"},
              },
              [], ["read"], True),
        _spec("/tool/os/file/write", file_write, "整体写入文件（覆盖），自动建目录",
              {"path": _STR, "content": _STR}, ["path", "content"], ["write"]),
        _spec("/tool/os/file/append", file_append, "追加写入文件",
              {"path": _STR, "content": _STR}, ["path", "content"], ["write"]),
    ]
}


def known_paths() -> list[str]:
    """All builtin paths in the specification table (sorted)."""
    return sorted(TOOL_SPECS)


#: Tools a fresh install mounts automatically on first start.
#:
#: This is the whole shipped capability surface: read, list, write, append
#: inside the working directory. Tree query needs no entry -- every abstract
#: node gets one from ``orange.tool_tree``.
DEFAULT_PATHS: tuple[str, ...] = (
    "/tool/os/file/read",
    "/tool/os/file/list",
    "/tool/os/file/write",
    "/tool/os/file/append",
)


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
