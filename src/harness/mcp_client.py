"""MCP 客户端：JSON-RPC 2.0 over stdio。

对应文档：`docs/06-工具分组与外来能力.md` 第四节。

设计要点：

* **只实现 stdio 传输**（覆盖 ``npx @modelcontextprotocol/server-*`` 这类
  最常见的启动方式）；HTTP/SSE 预留扩展位，本期不实现。
* **短连接**：每次调用 spawn 子进程，完成 ``initialize`` → ``tools/call``
  → 关闭。简单、无状态泄漏；代价是每次有进程启动开销。常驻连接留作后续
  优化（需要处理进程崩溃与重启）。
* **清单快照**：``tools/list`` 的结果在 ``mcp add`` 时一次性拉取并写入
  能力清单。这样运行时不需要为「发现工具」而拉起服务器，只有真正调用
  某个工具时才 spawn。
* **不在导入时执行任何进程**：本模块只提供函数，不产生副作用。

协议版本采用 ``2024-11-05``（MCP 首个稳定版）。
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

#: 使用的 MCP 协议版本。
PROTOCOL_VERSION = "2024-11-05"

#: 单次 JSON-RPC 往返的超时（秒）。
DEFAULT_TIMEOUT = 120.0


class McpError(RuntimeError):
    """MCP 通讯失败（进程启动失败、协议错误、超时）。"""


@dataclass(frozen=True)
class McpServerConfig:
    """一个 MCP 服务器的启动配置。

    Attributes:
        name: 服务器名，同时作为工具树上的分组名（``/mcp/<name>``）。
        command: 可执行文件，如 ``npx`` / ``python``。
        args: 传给命令的参数。
        env: 追加的环境变量（覆盖同名项）。
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        """导出为清单中的 JSON 结构。"""
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
        """从清单结构还原。"""
        env = data.get("env") or {}
        return cls(
            name=data["name"],
            command=data["command"],
            args=tuple(data.get("args") or ()),
            env=tuple(sorted(env.items())),
            cwd=data.get("cwd", ""),
        )


def load_mcp_servers(workdir: str | Path) -> dict[str, McpServerConfig]:
    """从能力清单读取 MCP 服务器声明。

    Raises:
        McpError: 清单不可读或结构非法。
    """
    from harness.capabilities import load_manifest

    try:
        manifest = load_manifest(workdir)
    except Exception as exc:  # noqa: BLE001 - 统一转成 MCP 侧错误
        raise McpError(f"读取能力清单失败: {exc}") from exc

    servers: dict[str, McpServerConfig] = {}
    for entry in manifest.mcp_servers:
        try:
            config = McpServerConfig.from_dict(entry)
        except KeyError as exc:
            raise McpError(f"MCP 服务器声明缺少字段 {exc}: {entry!r}") from exc
        servers[config.name] = config
    return servers


def _resolve_command(command: str) -> list[str]:
    """解析启动命令，兼容 Windows 的 ``.cmd`` / ``.bat`` 包装脚本。

    Windows 上 npm 系工具（``npx``、``uvx`` 等）不是可执行文件，而是
    ``npx.cmd``。``subprocess`` 直接传 ``"npx"`` 会抛 ``FileNotFoundError``，
    必须显式找到 ``.cmd`` 并透过 ``cmd.exe /c`` 启动（否则 CreateProcess
    也不接受 .cmd）。

    POSIX 上原样返回，交给系统 ``exec``。

    Returns:
        可直接传给 ``subprocess.Popen`` 的参数列表。
    """
    if os.name != "nt":
        return [command]

    # 已经带扩展名（如 python.exe）或本身就是路径 → 直接用
    suffix = Path(command).suffix.lower()
    if suffix in {".exe", ".cmd", ".bat", ".com"}:
        return [command]

    found = shutil.which(command) or shutil.which(f"{command}.cmd") or shutil.which(
        f"{command}.bat"
    )
    if not found:
        return [command]  # 交给 Popen 抛 FileNotFoundError，由上层翻译成 McpError

    if found.lower().endswith((".cmd", ".bat")):
        # .cmd/.bat 必须经 cmd.exe 解释；/c 表示执行后退出
        return ["cmd.exe", "/c", found]
    return [found]


def _spawn(config: McpServerConfig, workdir: str) -> subprocess.Popen[str]:
    """拉起 MCP 服务器子进程，stdin/stdout 走管道，stderr 收集用于诊断。"""
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
            f"找不到 MCP 服务命令 {config.command!r}；"
            "请确认它已安装且在 PATH 中"
            + ("（Windows 上 npm 工具请确保 npx.cmd 存在）" if os.name == "nt" else "")
        ) from exc


def _send(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    """写一行 JSON-RPC 消息。"""
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen[str], timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """读一行 JSON-RPC 响应（跳过通知与日志行）。"""
    assert proc.stdout is not None
    import threading

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def read_line() -> None:
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    error.append(McpError("MCP 服务器提前关闭了输出流"))
                    return
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    continue  # 服务器可能把日志写到 stdout，忽略
                if isinstance(message, dict) and ("result" in message or "error" in message):
                    result.update(message)
                    return
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=read_line, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise McpError(f"MCP 调用超时（{timeout:.0f}s）")
    if error:
        raise McpError(str(error[0]))
    return result


def _request(
    proc: subprocess.Popen[str],
    request_id: int,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """发一条请求并等对应响应；服务端返回 error 时抛 ``McpError``。"""
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    _send(proc, payload)
    response = _recv(proc)
    if "error" in response:
        err = response["error"] or {}
        raise McpError(f"MCP {method} 失败: {err.get('message', err)}")
    return response.get("result") or {}


def _bootstrap(proc: subprocess.Popen[str]) -> dict[str, Any]:
    """完成 initialize 握手并发送 initialized 通知。"""
    result = _request(
        proc,
        1,
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "harness", "version": "1.0"},
        },
    )
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    return result


def list_tools(config: McpServerConfig, workdir: str = ".") -> list[dict[str, Any]]:
    """拉取服务器暴露的工具清单（``tools/list``）。

    Returns:
        每项形如 ``{"name": str, "description": str, "inputSchema": dict}``。
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
    """调用服务器上的一个工具（``tools/call``），返回拼接后的文本结果。

    结果里的 ``content`` 数组按项拼接：``text`` 直接取，其他类型做 JSON
    序列化，避免丢信息。
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
        raise McpError("".join(parts) or f"MCP 工具 {tool_name} 返回错误")
    return "\n".join(parts)


def _shutdown(proc: subprocess.Popen[str]) -> None:
    """关闭子进程：先关 stdin，再等短暂超时，最后强杀。"""
    if proc.stdin:
        with contextlib.suppress(Exception):
            proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 - 超时则终止
        proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream:
                with contextlib.suppress(Exception):
                    stream.close()


def stderr_tail(proc: subprocess.Popen[str], limit: int = 2000) -> str:
    """读取子进程 stderr 尾部（诊断用）。"""
    if proc.stderr is None:
        return ""
    try:
        return (proc.stderr.read() or "")[-limit:]
    except Exception:  # noqa: BLE001
        return ""
