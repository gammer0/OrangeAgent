"""一个最小的 MCP 服务器（stdio），用于验证 harness 的 MCP 客户端。

实现 JSON-RPC 2.0 的三条必需方法：
    initialize / tools/list / tools/call

不带任何第三方依赖，纯标准库，直接 ``python tests/fixtures/mcp_echo_server.py``。
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "echo",
        "description": "回显文本",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "两数相加",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
]


def _handle(method: str, params: dict) -> dict:
    """按方法名返回结果。"""
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-server", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            return {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
        if name == "add":
            try:
                total = float(args.get("a", 0)) + float(args.get("b", 0))
            except (TypeError, ValueError):
                return {"content": [{"type": "text", "text": "参数不是数字"}], "isError": True}
            return {"content": [{"type": "text", "text": f"{total:g}"}]}
        return {"content": [{"type": "text", "text": f"未知工具 {name}"}], "isError": True}
    raise ValueError(f"未知方法 {method}")


def main() -> int:
    """逐行读 stdin 的 JSON-RPC 请求，逐行写 stdout 响应。"""
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue

        method = message.get("method", "")
        request_id = message.get("id")

        # 通知（无 id）不需要响应
        if request_id is None:
            continue

        try:
            result = _handle(method, message.get("params") or {})
            payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
