"""`mcp_servers/workspace_server.py` 的协议一致性测试。

直接驱动 JSON-RPC 行协议，不经过 harness 的客户端，这样失败时能判断
问题出在服务器还是客户端。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SERVER = Path(__file__).resolve().parents[1] / "mcp_servers" / "workspace_server.py"


class Session:
    """一个 MCP 会话：管理子进程与请求/响应往返。"""

    def __init__(self, root: Path) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER), "--root", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            payload["params"] = params
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        assert self.proc.stdout is not None
        return json.loads(self.proc.stdout.readline())

    def result(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request(method, params).get("result") or {}

    def handshake(self) -> dict[str, Any]:
        out = self.result(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self.proc.stdin.flush()
        return out

    def close(self) -> int:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)
        return self.proc.returncode or 0


@pytest.fixture()
def session(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("line one\nline two\n", encoding="utf-8")
    (tmp_path / "sub" / "b.py").write_text("needle = 1\n", encoding="utf-8")
    s = Session(tmp_path)
    s.handshake()
    try:
        yield s
    finally:
        s.close()


class TestHandshake:
    def test_initialize_shape(self, tmp_path: Path) -> None:
        s = Session(tmp_path)
        try:
            out = s.result(
                "initialize",
                {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {}},
            )
        finally:
            s.close()
        assert out["protocolVersion"] == "2024-11-05"
        assert out["serverInfo"]["name"] == "harness-workspace"
        caps = out["capabilities"]
        for key in ("tools", "resources", "prompts", "logging"):
            assert key in caps

    def test_ping(self, session: Session) -> None:
        assert session.result("ping") == {}

    def test_unknown_method_is_method_not_found(self, session: Session) -> None:
        response = session.request("does/not/exist")
        assert response["error"]["code"] == -32601


class TestTools:
    def test_list(self, session: Session) -> None:
        tools = session.result("tools/list")["tools"]
        names = {t["name"] for t in tools}
        assert names == {"list_dir", "read_file", "write_file", "stat_path", "search_text"}
        for tool in tools:
            assert tool["inputSchema"]["type"] == "object"

    def test_list_dir(self, session: Session) -> None:
        out = session.result("tools/call", {"name": "list_dir", "arguments": {"path": "."}})
        assert out["isError"] is False
        text = out["content"][0]["text"]
        assert "a.txt" in text and "sub" in text

    def test_read_and_write(self, session: Session, tmp_path: Path) -> None:
        session.result(
            "tools/call",
            {"name": "write_file", "arguments": {"path": "new.txt", "content": "abc"}},
        )
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "abc"

        out = session.result("tools/call", {"name": "read_file", "arguments": {"path": "new.txt"}})
        assert "abc" in out["content"][0]["text"]

    def test_read_line_numbers(self, session: Session) -> None:
        out = session.result("tools/call", {"name": "read_file", "arguments": {"path": "a.txt"}})
        text = out["content"][0]["text"]
        assert "1| line one" in text
        assert "2| line two" in text

    def test_stat_path_reports_sha256(self, session: Session) -> None:
        out = session.result("tools/call", {"name": "stat_path", "arguments": {"path": "a.txt"}})
        data = json.loads(out["content"][0]["text"])
        assert data["type"] == "file"
        assert len(data["sha256"]) == 64

    def test_search_text(self, session: Session) -> None:
        out = session.result(
            "tools/call",
            {"name": "search_text", "arguments": {"needle": "needle", "pattern": "*.py"}},
        )
        hits = json.loads(out["content"][0]["text"])
        assert hits and hits[0]["path"].endswith("b.py")

    def test_unknown_tool(self, session: Session) -> None:
        response = session.request("tools/call", {"name": "nope", "arguments": {}})
        assert response["error"]["code"] == -32602

    def test_bad_arguments(self, session: Session) -> None:
        response = session.request("tools/call", {"name": "read_file", "arguments": {}})
        assert response["error"]["code"] == -32602

    @pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd", "/etc/passwd"])
    def test_path_escape_rejected(self, session: Session, path: str) -> None:
        response = session.request("tools/call", {"name": "read_file", "arguments": {"path": path}})
        assert response["error"]["code"] == -32602
        assert "越界" in response["error"]["message"]

    def test_business_error_sets_is_error(self, session: Session) -> None:
        """业务异常应作为 isError 返回，而不是 JSON-RPC 错误。"""
        out = session.result(
            "tools/call", {"name": "read_file", "arguments": {"path": "missing.txt"}}
        )
        assert out["isError"] is True
        assert "不存在" in out["content"][0]["text"]


class TestResources:
    def test_list(self, session: Session) -> None:
        resources = session.result("resources/list")["resources"]
        uris = {r["uri"] for r in resources}
        assert "workspace:///a.txt" in uris

    def test_read(self, session: Session) -> None:
        contents = session.result("resources/read", {"uri": "workspace:///a.txt"})["contents"]
        assert contents[0]["uri"] == "workspace:///a.txt"
        assert "line one" in contents[0]["text"]
        assert contents[0]["mimeType"] == "text/plain"

    def test_templates(self, session: Session) -> None:
        templates = session.result("resources/templates/list")["resourceTemplates"]
        assert templates[0]["uriTemplate"] == "workspace:///{path}"

    def test_unsupported_uri(self, session: Session) -> None:
        response = session.request("resources/read", {"uri": "http://x/y"})
        assert response["error"]["code"] == -32602


class TestPrompts:
    def test_list(self, session: Session) -> None:
        prompts = session.result("prompts/list")["prompts"]
        assert {p["name"] for p in prompts} == {"review_file", "summarize_dir"}

    def test_get_review_file(self, session: Session) -> None:
        result = session.result(
            "prompts/get", {"name": "review_file", "arguments": {"path": "a.txt"}}
        )
        message = result["messages"][0]
        assert message["role"] == "user"
        assert "a.txt" in message["content"]["text"]

    def test_get_requires_path(self, session: Session) -> None:
        response = session.request("prompts/get", {"name": "review_file", "arguments": {}})
        assert response["error"]["code"] == -32602


class TestProtocolEdge:
    def test_missing_initialize_blocks_calls(self, tmp_path: Path) -> None:
        """未握手就调用工具应被拒。"""
        s = Session(tmp_path)
        try:
            response = s.request("tools/call", {"name": "list_dir", "arguments": {}})
        finally:
            s.close()
        assert response["error"]["code"] == -32600

    def test_malformed_json(self, tmp_path: Path) -> None:
        s = Session(tmp_path)
        try:
            assert s.proc.stdin is not None and s.proc.stdout is not None
            s.proc.stdin.write("{not json\n")
            s.proc.stdin.flush()
            response = json.loads(s.proc.stdout.readline())
        finally:
            s.close()
        assert response["error"]["code"] == -32700

    def test_notification_gets_no_response(self, tmp_path: Path) -> None:
        """通知（无 id）不该产生响应行。"""
        s = Session(tmp_path)
        try:
            assert s.proc.stdin is not None and s.proc.stdout is not None
            s.proc.stdin.write('{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
            s.proc.stdin.flush()
            # 紧接着发一个带 id 的请求，读到的第一行必须是对它的响应
            out = s.request("ping")
        finally:
            s.close()
        assert "result" in out

    def test_logging_set_level(self, session: Session) -> None:
        assert session.result("logging/setLevel", {"level": "debug"}) == {}
        response = session.request("logging/setLevel", {"level": "invalid"})
        assert response["error"]["code"] == -32602

    def test_clean_exit(self, tmp_path: Path) -> None:
        s = Session(tmp_path)
        s.handshake()
        assert s.close() == 0
