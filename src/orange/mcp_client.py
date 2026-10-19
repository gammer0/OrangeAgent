"""MCP client: JSON-RPC 2.0 over stdio.

See section 4 of the tool-groups-and-external-capabilities doc.

Design notes:

* **stdio transport only** (covers the most common launch style, e.g.
  ``npx @modelcontextprotocol/server-*``); HTTP/SSE is reserved for a later
  extension and not implemented yet.
* **Short-lived connections**: each call spawns a child process, performs
  ``initialize`` -> ``tools/call`` -> closes. Simple, with no state leakage; the
  cost is process startup overhead per call. Persistent connections are left as
  a later optimization (they require handling crashes and restarts).
* **Manifest snapshot**: the ``tools/list`` result is fetched once at ``mcp add``
  time and written into the capability manifest. The runtime therefore never has
  to start a server just to "discover tools"; it spawns only when a tool is
  actually called.
* **No process runs at import time**: this module only offers functions and has
  no side effects.

The protocol version is ``2024-11-05`` (MCP's first stable release).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PROTOCOL_VERSION",
    "McpError",
    "McpServerConfig",
    "load_mcp_servers",
    "list_tools",
    "call_tool",
]

#: The MCP protocol version in use.
PROTOCOL_VERSION = "2024-11-05"

#: Timeout for a single JSON-RPC round trip (seconds).
DEFAULT_TIMEOUT = 120.0


class McpError(RuntimeError):
    """MCP communication failed (process start failure, protocol error, timeout)."""


@dataclass(frozen=True)
class McpServerConfig:
    """Launch configuration for one MCP server.

    Attributes:
        name: Server name, also used as the tool-tree group name (``/mcp/<name>``).
        command: Executable, e.g. ``npx`` / ``python``.
        args: Arguments passed to the command.
        env: Extra environment variables (overriding same-named ones).
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Export as the JSON structure used in the manifest."""
        payload: dict[str, Any] = {"name": self.name, "command": self.command}
        if self.args:
            payload["args"] = list(self.args)
        if self.env:
            payload["env"] = {k: v for k, v in self.env}
        if self.cwd:
            payload["cwd"] = self.cwd
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> McpServerConfig:
        """Restore from the manifest structure."""
        env = data.get("env") or {}
        return cls(
            name=data["name"],
            command=data["command"],
            args=tuple(data.get("args") or ()),
            env=tuple(sorted(env.items())),
            cwd=data.get("cwd", ""),
        )


def load_mcp_servers(workdir: str | Path) -> dict[str, McpServerConfig]:
    """Read MCP server declarations from the capability manifest.

    Raises:
        McpError: The manifest is unreadable or structurally invalid.
    """
    from orange.capabilities import load_manifest

    try:
        manifest = load_manifest(workdir)
    except Exception as exc:  # noqa: BLE001 - unify into an MCP-side error
        raise McpError(f"failed to read the capability manifest: {exc}") from exc

    servers: dict[str, McpServerConfig] = {}
    for entry in manifest.mcp_servers:
        try:
            config = McpServerConfig.from_dict(entry)
        except KeyError as exc:
            raise McpError(f"MCP server declaration is missing field {exc}: {entry!r}") from exc
        servers[config.name] = config
    return servers


def _resolve_command(command: str) -> list[str]:
    """Resolve a launch command, handling Windows ``.cmd`` / ``.bat`` wrappers.

    On Windows the npm toolchain (``npx``, ``uvx``, ...) ships ``npx.cmd`` rather
    than an executable. Passing ``"npx"`` straight to ``subprocess`` raises
    ``FileNotFoundError``, so we must locate the ``.cmd`` explicitly and launch it
    through ``cmd.exe /c`` (CreateProcess refuses ``.cmd`` files directly).

    On POSIX the command is returned unchanged for the system ``exec``.

    Returns:
        An argument list usable directly with ``subprocess.Popen``.
    """
    if os.name != "nt":
        return [command]

    # Already has an extension (e.g. python.exe) or is itself a path -> use as-is
    suffix = Path(command).suffix.lower()
    if suffix in {".exe", ".cmd", ".bat", ".com"}:
        return [command]

    found = shutil.which(command) or shutil.which(f"{command}.cmd") or shutil.which(
        f"{command}.bat"
    )
    if not found:
        # Let Popen raise FileNotFoundError; the caller translates it to McpError
        return [command]

    if found.lower().endswith((".cmd", ".bat")):
        # .cmd/.bat must be interpreted by cmd.exe; /c means run then exit
        return ["cmd.exe", "/c", found]
    return [found]


def _spawn(config: McpServerConfig, workdir: str) -> subprocess.Popen[str]:
    """Spawn the MCP server child process with piped stdin/stdout.

    stderr is captured for diagnostics.
    """
    env = dict(os.environ)
    env.update(dict(config.env))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    cwd = config.cwd or workdir
    argv = [*_resolve_command(config.command), *config.args]
    try:
        return subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise McpError(
            f"MCP command not found: {config.command!r}; "
            "make sure it is installed and on PATH"
            + (" (on Windows npm tools need npx.cmd)" if os.name == "nt" else "")
        ) from exc


def _send(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    """Write one JSON-RPC message as a line."""
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen[str], timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Read one JSON-RPC response line (skipping notifications and log lines)."""
    assert proc.stdout is not None
    import threading

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def read_line() -> None:
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    error.append(McpError("the MCP server closed its output stream early"))
                    return
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    continue  # servers may log to stdout; ignore those lines
                if isinstance(message, dict) and ("result" in message or "error" in message):
                    result.update(message)
                    return
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=read_line, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise McpError(f"MCP call timed out ({timeout:.0f}s)")
    if error:
        raise McpError(str(error[0]))
    return result


def _request(
    proc: subprocess.Popen[str],
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a request and await its response.

    Raises ``McpError`` when the server returns an error.
    """
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    _send(proc, payload)
    response = _recv(proc)
    if "error" in response:
        err = response["error"] or {}
        raise McpError(f"MCP {method} failed: {err.get('message', err)}")
    return response.get("result") or {}


def _bootstrap(proc: subprocess.Popen[str]) -> dict[str, Any]:
    """Complete the initialize handshake and send the initialized notification."""
    result = _request(
        proc,
        1,
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "orange", "version": "1.0"},
        },
    )
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    return result


def list_tools(config: McpServerConfig, workdir: str = ".") -> list[dict[str, Any]]:
    """Fetch the tool list the server exposes (``tools/list``).

    Returns:
        Items shaped like ``{"name": str, "description": str, "inputSchema": dict}``.
    """
    proc = _spawn(config, workdir)
    try:
        _bootstrap(proc)
        result = _request(proc, 2, "tools/list", {})
        tools = result.get("tools") or []
        return [t for t in tools if isinstance(t, dict) and t.get("name")]
    finally:
        _shutdown(proc)


def call_tool(
    config: McpServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
    workdir: str = ".",
) -> str:
    """Call one tool on the server (``tools/call``) and return the concatenated text result.

    Items of the ``content`` array are concatenated: ``text`` is taken directly and
    other kinds are JSON-serialized so no information is lost.
    """
    proc = _spawn(config, workdir)
    try:
        _bootstrap(proc)
        result = _request(
            proc,
            2,
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
    finally:
        _shutdown(proc)

    content = result.get("content") or []
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        if "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    if result.get("isError"):
        raise McpError("".join(parts) or f"MCP tool {tool_name} returned an error")
    return "\n".join(parts)


def _shutdown(proc: subprocess.Popen[str]) -> None:
    """Shut down the child process: close stdin, wait briefly, then kill."""
    if proc.stdin:
        with contextlib.suppress(Exception):
            proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 - terminate on timeout
        proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream:
                with contextlib.suppress(Exception):
                    stream.close()


def stderr_tail(proc: subprocess.Popen[str], limit: int = 2000) -> str:
    """Read the tail of the child's stderr (for diagnostics)."""
    if proc.stderr is None:
        return ""
    try:
        return (proc.stderr.read() or "")[-limit:]
    except Exception:  # noqa: BLE001
        return ""
