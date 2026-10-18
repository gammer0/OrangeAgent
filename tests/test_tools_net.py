"""网络访问内置工具的测试。

不打外部网络：用本地临时 HTTP 服务做真实往返，保证测试可离线跑。
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from harness.tools_builtin import TOOL_SPECS, build_tool, net_get, net_head, net_post


class _Handler(http.server.BaseHTTPRequestHandler):
    """极简 HTTP 服务：/echo 回显、/status/404 返回 404、其余 200。"""

    def log_message(self, *args: Any) -> None:  # noqa: ANN002 - 静默日志
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path.startswith("/status/"):
            code = int(self.path.rsplit("/", 1)[-1])
            self.send_response(code)
            self.end_headers()
            self.wfile.write(f"code={code}".encode())
            return
        body = f"GET {self.path}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length).decode("utf-8")
        body = f"POST {self.path} <- {payload}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("X-Probe", "yes")
        self.end_headers()


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """启动本地 HTTP 服务，返回 base URL。"""
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


class TestRegistration:
    def test_three_net_tools_registered(self) -> None:
        paths = [p for p in TOOL_SPECS if p.startswith("/tool/net/")]
        assert paths == ["/tool/net/http/get", "/tool/net/http/post", "/tool/net/http/head"]

    @pytest.mark.parametrize(
        "path", ["/tool/net/http/get", "/tool/net/http/post", "/tool/net/http/head"]
    )
    def test_caps_is_net(self, path: str, tmp_path) -> None:
        entry = build_tool(path, str(tmp_path))
        assert entry.caps == ["net"]
        # B11：``["net"]`` 单独声明推导为 exclusive（见 derive_concurrency）
        assert entry.concurrency == "exclusive"

    def test_read_net_would_be_safe(self) -> None:
        """留档：若要给 GET/HEAD 并发执行权，需同时声明 read。

        当前 GET/HEAD 只声明 net，因此串行执行——这是保守选择，
        避免并发请求同一主机时被误判为幂等。
        """
        from harness.tool_tree import derive_concurrency

        assert derive_concurrency(["net"]) == "exclusive"
        assert derive_concurrency(["read", "net"]) == "safe"

    def test_post_is_not_idempotent(self) -> None:
        assert TOOL_SPECS["/tool/net/http/post"]["idempotent"] is False

    @pytest.mark.parametrize("path", ["/tool/net/http/get", "/tool/net/http/head"])
    def test_get_and_head_are_idempotent(self, path: str) -> None:
        assert TOOL_SPECS[path]["idempotent"] is True


class TestProtocolGuard:
    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", ""],
    )
    def test_rejects_non_http(self, url: str) -> None:
        with pytest.raises(ValueError):
            net_get(url)

    def test_rejects_non_http_on_post_and_head(self) -> None:
        with pytest.raises(ValueError):
            net_post("file:///x")
        with pytest.raises(ValueError):
            net_head("file:///x")


class TestRealRequests:
    def test_get_returns_status_and_body(self, server: str) -> None:
        out = net_get(f"{server}/hello")
        assert "状态 200" in out
        assert "GET /hello" in out

    def test_get_reports_error_status(self, server: str) -> None:
        out = net_get(f"{server}/status/404")
        assert "状态 404" in out
        assert "code=404" in out

    def test_post_sends_body(self, server: str) -> None:
        out = net_post(f"{server}/submit", body='{"a":1}')
        assert "状态 200" in out
        assert "POST /submit" in out
        assert '{"a":1}' in out

    def test_head_returns_headers(self, server: str) -> None:
        out = net_head(f"{server}/probe")
        assert "状态 204" in out
        assert "x-probe: yes" in out.lower()

    def test_connection_error_wrapped(self) -> None:
        # 未监听的端口 → 必须转成 ConnectionError，而不是裸 httpx 异常
        with pytest.raises((ConnectionError, TimeoutError)):
            net_get("http://127.0.0.1:1/nope", timeout=2)

    def test_output_is_clipped(self, server: str) -> None:
        from harness.tools_builtin import MAX_OUTPUT_CHARS

        out = net_get(f"{server}/{'x' * 200}")
        assert len(out) <= MAX_OUTPUT_CHARS + 200  # 截断提示本身占几十字符
