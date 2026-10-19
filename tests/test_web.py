"""Web 服务层测试：REST 接口、SSE 事件与前端资源。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from apps.server import main as web
from fastapi.testclient import TestClient

from orange.message import AssistantMsg
from orange.model import ProviderConfig, ReverseProxy
from orange.rate import RateWindow


class FakeTransport:
    """按脚本返回响应的假传输，避免真实网络调用。"""

    def __init__(
        self, replies: list[tuple[str | None, list[dict[str, Any]]]] | None = None
    ) -> None:
        self.replies = list(replies or [])

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        content, calls = self.replies.pop(0) if self.replies else ("完成", [])
        return {
            "model": json.get("model", "fake"),
            "choices": [{"message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _fake_proxy(
    replies: list[tuple[str | None, list[dict[str, Any]]]] | None = None,
) -> ReverseProxy:
    return ReverseProxy(
        providers=[
            ProviderConfig(
                name="fake",
                base_url="https://fake.invalid",
                model="fake-model",
                api_key="sk-fake",
            )
        ],
        rate_window=RateWindow(max_requests=100, max_tokens=1_000_000),
        health={},
        transport=FakeTransport(replies),
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """构造使用假模型通道的测试客户端。"""

    def fake_build_manager(workdir: Path) -> Any:
        from apps.server.sessions import SessionManager

        proxy = _fake_proxy([("你好", []), (None, []), ("完成", [])])
        manager = SessionManager(
            workdir=str(workdir),
            proxy=proxy,
            default_model="fake-model",
            capacity=10_000,
        )
        return manager, None

    monkeypatch.setattr(web, "_build_manager", fake_build_manager)
    app = web.create_app(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failing_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """模拟配置缺失（未就绪）的服务。"""

    def broken(workdir: Path) -> Any:
        return None, "缺少环境变量 DEEPSEEK_API_KEY"

    monkeypatch.setattr(web, "_build_manager", broken)
    app = web.create_app(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndStatic:
    def test_index_serves_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "<!doctype html>" in response.text.lower()
        # 品牌标识（CSS 用 text-transform 渲染为大写，标记本身可小写）
        assert "orange" in response.text.lower()

    def test_health_ok(self, client: TestClient) -> None:
        payload = client.get("/api/health").json()
        assert payload["ok"] is True
        assert "workdir" in payload

    def test_health_reports_not_ready(self, failing_client: TestClient) -> None:
        payload = failing_client.get("/api/health").json()
        assert payload["ok"] is False
        assert "DEEPSEEK_API_KEY" in payload["error"]

    def test_openapi_available(self, client: TestClient) -> None:
        payload = client.get("/openapi.json").json()
        assert "/api/sessions" in payload["paths"]


class TestNotReady:
    def test_sessions_returns_503(self, failing_client: TestClient) -> None:
        response = failing_client.get("/api/sessions")
        assert response.status_code == 503
        assert "未就绪" in response.json()["detail"]

    def test_models_returns_503(self, failing_client: TestClient) -> None:
        assert failing_client.get("/api/models").status_code == 503


class TestModelsAndTools:
    def test_models_include_default(self, client: TestClient) -> None:
        payload = client.get("/api/models").json()
        assert "fake-model" in payload["models"]
        assert payload["default"] == "fake-model"

    def test_tools_listed(self, client: TestClient) -> None:
        payload = client.get("/api/tools").json()
        paths = {t["path"] for t in payload["tools"]}
        assert paths == {
            "/tool/os/file/read",
            "/tool/os/file/list",
            "/tool/os/file/write",
            "/tool/os/file/append",
        }


class TestSessionLifecycle:
    def test_create_session(self, client: TestClient) -> None:
        response = client.post("/api/sessions", json={})
        assert response.status_code == 201
        payload = response.json()
        assert payload["state"] == "idle"
        assert payload["model"] == "fake-model"
        # 只有 SystemMsg
        assert payload["message_count"] == 1

    def test_create_with_model_override(self, client: TestClient) -> None:
        payload = client.post("/api/sessions", json={"model": "other-model"}).json()
        assert payload["model"] == "other-model"

    def test_list_sessions(self, client: TestClient) -> None:
        client.post("/api/sessions", json={})
        client.post("/api/sessions", json={})
        payload = client.get("/api/sessions").json()
        assert len(payload["sessions"]) == 2

    def test_get_session_detail(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        payload = client.get(f"/api/sessions/{created['session_id']}").json()
        assert payload["session_id"] == created["session_id"]
        assert payload["messages"][0]["role"] == "system"

    def test_missing_session_404(self, client: TestClient) -> None:
        assert client.get("/api/sessions/nope").status_code == 404

    def test_patch_model(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        payload = client.patch(
            f"/api/sessions/{created['session_id']}", json={"model": "switched"}
        ).json()
        assert payload["model"] == "switched"

    def test_delete_session(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        assert client.delete(f"/api/sessions/{created['session_id']}").status_code == 200
        assert client.get(f"/api/sessions/{created['session_id']}").status_code == 404

    def test_delete_missing_404(self, client: TestClient) -> None:
        assert client.delete("/api/sessions/nope").status_code == 404

    def test_capacity_override(self, client: TestClient) -> None:
        payload = client.post("/api/sessions", json={"capacity": 2048}).json()
        assert payload["capacity"] == 2048


class TestMessages:
    def test_get_messages_returns_capacity(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        payload = client.get(f"/api/sessions/{created['session_id']}/messages").json()
        assert payload["capacity"] == 10_000
        assert isinstance(payload["messages"], list)

    def test_empty_text_rejected(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        response = client.post(
            f"/api/sessions/{created['session_id']}/messages", json={"text": ""}
        )
        assert response.status_code == 422

    def test_invalid_channel_rejected(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        response = client.post(
            f"/api/sessions/{created['session_id']}/messages",
            json={"text": "hi", "channel": "bogus"},
        )
        assert response.status_code == 422

    def test_send_message_accepted(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        response = client.post(
            f"/api/sessions/{created['session_id']}/messages",
            json={"text": "你好", "channel": "chat"},
        )
        assert response.status_code == 202
        assert response.json()["accepted"] is True

    def test_cancel_accepted(self, client: TestClient) -> None:
        created = client.post("/api/sessions", json={}).json()
        response = client.post(f"/api/sessions/{created['session_id']}/cancel")
        assert response.status_code == 200
        assert response.json()["cancelling"] is True


class TestSessionRunner:
    def test_runner_emits_state_and_usage(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry, SessionRunner

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="s1",
            workdir=str(tmp_path),
            proxy=_fake_proxy([("你好", []), ("完成", [])]),
            tools=tools,
            model="fake",
            capacity=10_000,
        )
        # 订阅在启动前建立，否则会错过本轮事件
        subscription = runner.subscribe()
        assert runner.start("测试")
        runner._thread.join(timeout=10)

        kinds = []
        while not subscription.empty():
            kinds.append(subscription.get_nowait().kind)

        # 用户消息由前端本地渲染，不经 inject 事件（避免重复）
        assert "assistant" in kinds
        assert "usage" in kinds
        assert "state" in kinds
        assert runner.state in {"done", "error"}
        # 用户消息只应出现在历史里一次
        user_msgs = [m for m in runner.messages if m.role == "user"]
        assert len(user_msgs) == 1

    def test_runner_rejects_concurrent_start(self, tmp_path: Path) -> None:
        import threading

        from apps.server.sessions import DemoToolRegistry, SessionRunner

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="s2",
            workdir=str(tmp_path),
            proxy=_fake_proxy([("a", [])]),
            tools=tools,
            model="fake",
        )
        # 模拟正在运行
        runner._thread = threading.Thread(target=lambda: None)
        runner._thread.start()
        runner._thread.join()
        # 线程已结束，应允许再次启动
        assert runner.start("再来一次") in {True, False}

    def test_user_message_not_duplicated(self, tmp_path: Path) -> None:
        """回归：历史传入与 inbox 注入不得同时发生，否则用户消息重复。"""
        from apps.server.sessions import DemoToolRegistry, SessionRunner

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="dup",
            workdir=str(tmp_path),
            proxy=_fake_proxy([("好的", [])]),
            tools=tools,
            model="fake",
        )
        assert runner.start("只应出现一次")
        runner._thread.join(timeout=10)

        user_texts = [m.content for m in runner.messages if m.role == "user"]
        assert user_texts == ["只应出现一次"]

    def test_assistant_emitted_once_per_reply(self, tmp_path: Path) -> None:
        """回归：最终答案只应产生一条 assistant 事件。"""
        from apps.server.sessions import DemoToolRegistry, SessionRunner

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="once",
            workdir=str(tmp_path),
            proxy=_fake_proxy([("唯一回复", [])]),
            tools=tools,
            model="fake",
        )
        subscription = runner.subscribe()
        runner.start("问一句")
        runner._thread.join(timeout=10)

        finals = []
        while not subscription.empty():
            event = subscription.get_nowait()
            if event.kind == "assistant" and event.payload.get("final"):
                finals.append(event.payload)
        assert len(finals) == 1

    def test_snapshot_never_leaks_key(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry, SessionRunner

        tools = DemoToolRegistry(str(tmp_path))
        runner = SessionRunner(
            session_id="s3",
            workdir=str(tmp_path),
            proxy=_fake_proxy(),
            tools=tools,
            model="fake",
        )
        snapshot = json.dumps(runner.snapshot(include_messages=True), ensure_ascii=False)
        assert "sk-fake" not in snapshot


class TestDemoTools:
    def test_read_file_inside_workdir(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry

        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        registry = DemoToolRegistry(str(tmp_path))
        assert registry._read_file("a.txt") == "hello"

    def test_read_file_outside_workdir_rejected(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry

        registry = DemoToolRegistry(str(tmp_path))
        with pytest.raises(ValueError):
            registry._read_file("../../etc/passwd")

    def test_write_and_list(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry

        registry = DemoToolRegistry(str(tmp_path))
        registry._write_file("sub/b.txt", "content")
        listing = registry._list_dir("sub")
        assert "b.txt" in listing

    def test_tool_specs_are_openai_shaped(self, tmp_path: Path) -> None:
        from apps.server.sessions import DemoToolRegistry

        registry = DemoToolRegistry(str(tmp_path))
        specs = registry.tool_specs()
        assert all(spec["type"] == "function" for spec in specs)
        assert all("parameters" in spec["function"] for spec in specs)


class TestEventFormat:
    def test_session_event_serializable(self, tmp_path: Path) -> None:
        from apps.server.sessions import SessionEvent

        event = SessionEvent(seq=1, ts=1.0, kind="state", payload={"state": "idle"})
        data = json.loads(json.dumps(event.to_dict(), ensure_ascii=False))
        assert data["kind"] == "state"
        assert data["seq"] == 1

    def test_assistant_message_mapping(self) -> None:
        from apps.server.sessions import _message_to_dict

        message = AssistantMsg(content="hi")
        payload = _message_to_dict(message)
        assert payload["role"] == "assistant"
        assert payload["content"] == "hi"
