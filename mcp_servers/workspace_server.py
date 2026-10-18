"""一个完整的 MCP 服务器实现（JSON-RPC 2.0 over stdio）。

与 `tests/fixtures/mcp_echo_server.py` 的区别：那个是最小探针，只实现
三条必需方法；本文件是一个**可实际使用的服务器**，覆盖 MCP 2024-11-05
规范里除网络传输之外的主要能力：

    initialize / notifications/initialized
    tools/list       tools/call
    resources/list   resources/read   resources/templates/list
    prompts/list     prompts/get
    ping             logging/setLevel

实现刻意只用标准库，便于直接 `python mcp_servers/workspace_server.py` 启动。

用法：
    python mcp_servers/workspace_server.py [--root <目录>]

挂载：
    harness mcp add workspace --command .venv\\Scripts\\python.exe \
        --arg mcp_servers/workspace_server.py --arg --root --arg .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "harness-workspace"
SERVER_VERSION = "1.0.0"

#: 单个资源/文件读取上限（字符），避免把大文件整个塞进响应。
MAX_RESOURCE_CHARS = 40_000

#: 允许的 JSON-RPC 错误码。
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """携带 JSON-RPC 错误码的业务异常。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidArgument(RpcError):
    """参数非法（路径越界、缺参数等）。

    与「工具执行失败」区分：前者是调用方式错，后者是环境问题。前者走
    JSON-RPC error，后者走 ``isError`` 的 content 结果。
    """

    def __init__(self, message: str) -> None:
        super().__init__(INVALID_PARAMS, message)


# ---------------------------------------------------------------------------
# 工作区后端：所有工具与资源操作都受 root 约束
# ---------------------------------------------------------------------------


class Workspace:
    """受限于单一根目录的文件工作区。

    所有路径都解析到 ``root`` 内；越界抛 ``RpcError(INVALID_PARAMS)``，
    这样客户端能拿到结构化错误而不是进程崩溃。
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    # -- 路径 ----------------------------------------------------------

    def resolve(self, raw: str) -> Path:
        """把相对路径解析到 root 内，拒绝越界与绝对路径。"""
        if not raw or raw in {".", "/"}:
            return self.root
        candidate = Path(raw)
        target = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise InvalidArgument(f"路径越界（只能在 {self.root} 内操作）: {raw}")
        return target

    def rel(self, path: Path) -> str:
        """转成相对 root 的展示路径。"""
        try:
            return path.relative_to(self.root).as_posix() or "."
        except ValueError:
            return path.as_posix()

    # -- 文件操作 ------------------------------------------------------

    def list_dir(self, raw: str = ".", pattern: str = "*") -> list[dict[str, Any]]:
        """列目录：名称、类型、大小。"""
        target = self.resolve(raw)
        if not target.is_dir():
            raise RpcError(INVALID_PARAMS, f"不是目录: {raw}")
        entries = []
        for item in sorted(target.glob(pattern), key=lambda p: (not p.is_dir(), p.name))[:500]:
            entries.append(
                {
                    "name": item.name,
                    "path": self.rel(item),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        return entries

    def read_file(self, raw: str, offset: int = 0, limit: int = 400) -> str:
        """按行读取文本文件。"""
        target = self.resolve(raw)
        if not target.is_file():
            raise RpcError(INVALID_PARAMS, f"文件不存在: {raw}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, int(offset))
        count = max(1, min(int(limit), 2000))
        chunk = lines[start : start + count]
        body = "\n".join(f"{start + i + 1:>5}| {line}" for i, line in enumerate(chunk))
        header = (
            f"[{self.rel(target)}] 共 {len(lines)} 行，"
            f"显示 {start + 1}-{start + len(chunk)}"
        )
        return f"{header}\n{body}"

    def write_file(self, raw: str, content: str) -> str:
        """整体写入文件（覆盖），自动建父目录。"""
        target = self.resolve(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {self.rel(target)}：{len(content)} 字符"

    def stat_path(self, raw: str) -> dict[str, Any]:
        """返回路径元信息，含 sha256（文件）。"""
        target = self.resolve(raw)
        if not target.exists():
            raise RpcError(INVALID_PARAMS, f"路径不存在: {raw}")
        info: dict[str, Any] = {
            "path": self.rel(target),
            "type": "dir" if target.is_dir() else "file",
            "size": target.stat().st_size,
        }
        if target.is_file():
            info["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        return info

    def search_text(self, needle: str, pattern: str = "*.py", max_hits: int = 50) -> list[dict]:
        """在 root 下按 glob 搜索文本，返回命中行。"""
        if not needle:
            raise RpcError(INVALID_PARAMS, "needle 不能为空")
        hits: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob(pattern)):
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    hits.append(
                        {"path": self.rel(path), "line": number, "text": line.strip()[:200]}
                    )
                    if len(hits) >= max_hits:
                        return hits
        return hits

    # -- 资源 ----------------------------------------------------------

    def resources(self) -> list[dict[str, Any]]:
        """把 root 下的顶层文件与目录暴露为 MCP 资源。"""
        items: list[dict[str, Any]] = []
        for entry in sorted(self.root.iterdir())[:200]:
            uri = f"workspace:///{self.rel(entry)}"
            items.append(
                {
                    "uri": uri,
                    "name": entry.name,
                    "description": "目录" if entry.is_dir() else f"文件 {entry.stat().st_size}B",
                    "mimeType": "text/plain" if entry.is_file() else "inode/directory",
                }
            )
        return items

    def read_resource(self, uri: str) -> str:
        """按 ``workspace:///<相对路径>`` 读取资源内容。"""
        prefix = "workspace:///"
        if not uri.startswith(prefix):
            raise RpcError(INVALID_PARAMS, f"不支持资源 URI: {uri}")
        raw = uri[len(prefix) :] or "."
        target = self.resolve(raw)
        if target.is_dir():
            names = sorted(p.name for p in target.iterdir())[:200]
            return "\n".join(names) or "（空目录）"
        if not target.is_file():
            raise RpcError(INVALID_PARAMS, f"资源不存在: {uri}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_RESOURCE_CHARS:
            return text[:MAX_RESOURCE_CHARS] + f"\n...（已截断，原 {len(text)} 字符）"
        return text


# ---------------------------------------------------------------------------
# MCP 服务器
# ---------------------------------------------------------------------------


class McpServer:
    """按 MCP 规范分发请求。"""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.initialized = False
        self.log_level = "info"
        self._tools = self._build_tools()

    # -- 工具定义 ------------------------------------------------------

    def _build_tools(self) -> dict[str, dict[str, Any]]:
        """工具表：name → {schema, handler}。"""
        ws = self.ws

        def tool(
            name: str,
            description: str,
            properties: dict[str, Any],
            required: list[str],
            handler: Callable[..., Any],
        ) -> tuple[str, dict[str, Any]]:
            return name, {
                "definition": {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
                "handler": handler,
            }

        str_t = {"type": "string"}
        return dict(
            [
                tool(
                    "list_dir",
                    "列出目录内容",
                    {"path": {"type": "string", "default": ".", "description": "相对路径"}},
                    [],
                    lambda path=".": ws.list_dir(path),
                ),
                tool(
                    "read_file",
                    "按行读取文本文件",
                    {
                        "path": str_t,
                        "offset": {"type": "integer", "default": 0},
                        "limit": {"type": "integer", "default": 400},
                    },
                    ["path"],
                    lambda path, offset=0, limit=400: ws.read_file(path, offset, limit),
                ),
                tool(
                    "write_file",
                    "写入文本文件（覆盖）",
                    {"path": str_t, "content": str_t},
                    ["path", "content"],
                    lambda path, content: ws.write_file(path, content),
                ),
                tool(
                    "stat_path",
                    "查询路径元信息（含文件 sha256）",
                    {"path": str_t},
                    ["path"],
                    lambda path: json.dumps(ws.stat_path(path), ensure_ascii=False),
                ),
                tool(
                    "search_text",
                    "在文件中搜索文本",
                    {
                        "needle": str_t,
                        "pattern": {"type": "string", "default": "*.py"},
                        "max_hits": {"type": "integer", "default": 50},
                    },
                    ["needle"],
                    lambda needle, pattern="*.py", max_hits=50: json.dumps(
                        ws.search_text(needle, pattern, max_hits), ensure_ascii=False
                    ),
                ),
            ]
        )

    # -- 分发 ----------------------------------------------------------

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        """按方法名处理请求；未知方法抛 ``RpcError``。"""
        handler = getattr(self, f"_m_{method.replace('/', '_').replace('-', '_')}", None)
        if handler is None:
            raise RpcError(METHOD_NOT_FOUND, f"未实现的方法: {method}")
        return handler(params)

    # -- 生命周期 ------------------------------------------------------

    def _m_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """握手：返回协议版本与能力声明。"""
        self.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "这是受限工作区服务器：所有路径都相对根目录，不能越界。"
                "写文件前建议先 stat_path 确认现状。"
            ),
        }

    def _m_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """探活。"""
        return {}

    def _m_logging_setLevel(self, params: dict[str, Any]) -> dict[str, Any]:
        """调整日志级别（本实现只记录，不输出到 stdout 以免破坏协议）。"""
        level = str(params.get("level", "info"))
        if level not in {"debug", "info", "warning", "error"}:
            raise RpcError(INVALID_PARAMS, f"非法日志级别: {level}")
        self.log_level = level
        return {}

    # -- tools ---------------------------------------------------------

    def _m_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """返回工具清单。"""
        return {"tools": [entry["definition"] for entry in self._tools.values()]}

    def _m_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行工具，返回 MCP 标准的 content 数组。"""
        if not self.initialized:
            raise RpcError(INVALID_REQUEST, "必须先 initialize")
        name = params.get("name")
        args = params.get("arguments") or {}
        entry = self._tools.get(str(name))
        if entry is None:
            raise RpcError(INVALID_PARAMS, f"未知工具: {name}")
        if not isinstance(args, dict):
            raise RpcError(INVALID_PARAMS, "arguments 必须是对象")

        try:
            result = entry["handler"](**args)
        except TypeError as exc:
            # 参数名/数量不匹配 → 客户端参数错误，走 JSON-RPC 错误
            raise InvalidArgument(f"参数不匹配: {exc}") from exc
        except InvalidArgument:
            raise
        except RpcError as exc:
            # 业务失败（文件不存在等）应作为工具结果错误返回：
            # 模型看到 isError 后可以自行调整，而 JSON-RPC 错误会被当成协议故障。
            return {
                "content": [{"type": "text", "text": exc.message}],
                "isError": True,
            }
        except Exception as exc:  # noqa: BLE001 - 兜底业务异常转成 MCP isError
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }

        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    # -- resources -----------------------------------------------------

    def _m_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """列出资源。"""
        return {"resources": self.ws.resources()}

    def _m_resources_templates_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """资源模板：把整个工作区暴露成一个可参数化 URI。"""
        return {
            "resourceTemplates": [
                {
                    "uriTemplate": "workspace:///{path}",
                    "name": "工作区文件",
                    "description": "以相对路径访问工作区内任意文件或目录",
                    "mimeType": "text/plain",
                }
            ]
        }

    def _m_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """读取资源，返回 contents 数组（规范要求带 uri 与 mimeType）。"""
        uri = str(params.get("uri", ""))
        if not uri:
            raise RpcError(INVALID_PARAMS, "缺少 uri")
        text = self.ws.read_resource(uri)
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    # -- prompts -------------------------------------------------------

    def _m_prompts_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """列出提示模板。"""
        return {
            "prompts": [
                {
                    "name": "review_file",
                    "description": "审查一个文件的实现质量",
                    "arguments": [
                        {"name": "path", "description": "相对路径", "required": True},
                        {"name": "focus", "description": "关注点", "required": False},
                    ],
                },
                {
                    "name": "summarize_dir",
                    "description": "总结目录结构与职责划分",
                    "arguments": [{"name": "path", "description": "相对路径", "required": True}],
                },
            ]
        }

    def _m_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """渲染提示模板。"""
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        if name == "review_file":
            path = args.get("path")
            if not path:
                raise RpcError(INVALID_PARAMS, "review_file 需要 path 参数")
            focus = args.get("focus") or "正确性与可维护性"
            return {
                "description": f"审查 {path}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                f"请用 read_file 读取 {path}，围绕「{focus}」审查，"
                                "指出最多三个具体问题并给出修改建议。"
                            ),
                        },
                    }
                ],
            }
        if name == "summarize_dir":
            path = args.get("path")
            if not path:
                raise RpcError(INVALID_PARAMS, "summarize_dir 需要 path 参数")
            return {
                "description": f"总结目录 {path}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": (
                                f"请用 list_dir 与 read_file 了解 {path} 的结构，"
                                "用不超过 10 行总结每个文件/子目录的职责。"
                            ),
                        },
                    }
                ],
            }
        raise RpcError(INVALID_PARAMS, f"未知提示: {name}")


# ---------------------------------------------------------------------------
# 传输层
# ---------------------------------------------------------------------------


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """构造 JSON-RPC 错误响应。"""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve() -> int:
    """从 stdin 逐行读请求，逐行写响应。"""
    parser = argparse.ArgumentParser(description="Harness workspace MCP server")
    parser.add_argument("--root", default=".", help="工作区根目录")
    options = parser.parse_args()

    root = Path(options.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = McpServer(Workspace(root))

    # 注意：stdout 是协议通道，任何调试输出都必须走 stderr
    print(f"[{SERVER_NAME}] root={root}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue

        try:
            message = json.loads(text)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps(_error(None, PARSE_ERROR, str(exc))) + "\n")
            sys.stdout.flush()
            continue

        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            sys.stdout.write(
                json.dumps(_error(None, INVALID_REQUEST, "需要 jsonrpc=2.0")) + "\n"
            )
            sys.stdout.flush()
            continue

        method = str(message.get("method", ""))
        request_id = message.get("id")

        # 通知没有 id，不需要响应
        if request_id is None:
            continue

        try:
            result = server.handle(method, message.get("params") or {})
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except RpcError as exc:
            payload = _error(request_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - 兜底，保持进程存活
            payload = _error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
