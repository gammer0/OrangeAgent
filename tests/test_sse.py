"""SSE 广播与前端渲染约定测试（S5）。

三件事在此前是断的：

* ``SessionRunner`` 只有一个共享队列，任何第二条 SSE 连接（新标签页、重连、
  调试连接）都会把事件抢走，导致界面永久停在「执行中」；
* 前端仍绑定已删除的 ``refresh-context``，抛 ``TypeError`` 后整段脚本中断，
  连健康轮询都没启动；
* 对话区还在渲染工具调用与结果卡片。

本文件锁定修复后的行为。
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from apps.server.sessions import DemoToolRegistry, SessionRunner

from orange.model import ProviderConfig, ReverseProxy
from orange.rate import RateWindow

_STATIC = Path(__file__).resolve().parents[1] / "src" / "orange" / "web"

Reply = tuple[str | None, list[dict[str, Any]]]


class FakeTransport:
    """按脚本返回响应的假传输。"""

    def __init__(self, replies: list[Reply] | None = None) -> None:
        self.replies = list(replies or [])

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> dict[str, Any]:
        content, calls = self.replies.pop(0) if self.replies else ("完成", [])
        return {
            "model": json.get("model", "fake"),
            "choices": [{"message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _proxy(replies: list[tuple[str | None, list[dict[str, Any]]]] | None = None) -> ReverseProxy:
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


def _runner(tmp_path: Path, session_id: str = "sse") -> SessionRunner:
    return SessionRunner(
        session_id=session_id,
        workdir=str(tmp_path),
        proxy=_proxy(),
        tools=DemoToolRegistry(str(tmp_path)),
        model="fake",
        capacity=10_000,
    )


def _drain(q: queue.Queue[Any]) -> list[Any]:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class TestBroadcast:
    def test_two_subscribers_both_receive_every_event(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        first = runner.subscribe()
        second = runner.subscribe()

        runner.emit("state", {"state": "running"})
        runner.emit("assistant", {"content": "hi", "final": True})

        # 关键回归：第二个订阅者不得把事件从第一个手里抢走
        assert [e.kind for e in _drain(first)] == ["state", "assistant"]
        assert [e.kind for e in _drain(second)] == ["state", "assistant"]

    def test_late_subscriber_gets_replay_backlog(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        runner.emit("state", {"state": "running"})
        runner.emit("assistant", {"content": "早期事件", "final": True})

        late = runner.subscribe()
        # 连接建立前发生的事件必须补发，否则界面状态会永久错位
        assert [e.kind for e in _drain(late)] == ["state", "assistant"]

    def test_last_event_id_skips_already_seen(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        runner.emit("state", {"state": "running"})
        seen = runner.subscribe().get_nowait()

        runner.emit("assistant", {"content": "新事件", "final": True})
        resumed = runner.subscribe(after_seq=seen.seq)

        assert [e.kind for e in _drain(resumed)] == ["assistant"]

    def test_unsubscribe_stops_delivery(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        q = runner.subscribe()
        runner.unsubscribe(q)

        runner.emit("state", {"state": "done"})
        assert _drain(q) == []

    def test_unsubscribe_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner(tmp_path)
        q = runner.subscribe()
        runner.unsubscribe(q)
        runner.unsubscribe(q)  # 不得抛异常（连接结束时会被调用多次）

    def test_replay_backlog_is_bounded(self, tmp_path: Path) -> None:
        from apps.server.sessions import EVENT_REPLAY_MAX

        runner = _runner(tmp_path)
        for i in range(EVENT_REPLAY_MAX + 25):
            runner.emit("state", {"state": "running", "detail": str(i)})

        backlog = _drain(runner.subscribe())
        assert len(backlog) == EVENT_REPLAY_MAX
        # 保留的是最近的，不是最早的
        assert backlog[-1].payload["detail"] == str(EVENT_REPLAY_MAX + 24)

    def test_full_subscriber_does_not_block_producer(self, tmp_path: Path) -> None:
        from apps.server.sessions import EVENT_QUEUE_MAX

        runner = _runner(tmp_path)
        stuck = runner.subscribe()  # 故意不消费
        for i in range(EVENT_QUEUE_MAX + 5):
            runner.emit("state", {"state": "running", "detail": str(i)})

        # 生产者没有被阻塞，队列保持在上限，且丢的是最旧的
        assert stuck.qsize() == EVENT_QUEUE_MAX
        assert stuck.get_nowait().payload["detail"] != "0"


class TestToolTraceEvent:
    """工具调用的参数必须能被轨迹面板取到（对话区不再显示工具卡片）。"""

    def test_tool_event_carries_call_arguments(self, tmp_path: Path) -> None:
        from orange.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

        runner = _runner(tmp_path, "trace")
        q = runner.subscribe()

        call = ToolCall(id="c1", name="/fs/read", arguments='{"path": "a.txt"}')
        runner.messages.append(UserMsg(content="读文件"))
        runner._on_assistant(AssistantMsg(content=None, tool_calls=[call]))
        runner.messages.append(AssistantMsg(content=None, tool_calls=[call]))
        runner.messages.append(ToolMsg(tool_call_id="c1", name="/fs/read", content="内容"))
        runner._on_batch_done(runner.messages)

        tool_events = [e for e in _drain(q) if e.kind == "tool"]
        assert len(tool_events) == 1
        payload = tool_events[0].payload
        assert payload["path"] == "/fs/read"
        assert payload["arguments"] == '{"path": "a.txt"}'
        assert payload["content"] == "内容"

    def test_arguments_survive_far_back_assistant(self, tmp_path: Path) -> None:
        """产生调用的 AssistantMsg 远在窗口之外时，参数仍要能配对上。"""
        from orange.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

        runner = _runner(tmp_path, "far")
        q = runner.subscribe()

        call = ToolCall(id="deep", name="/fs/read", arguments='{"path": "z.txt"}')
        runner._on_assistant(AssistantMsg(content=None, tool_calls=[call]))
        # 中间塞入大量历史，把 AssistantMsg 挤出任何固定窗口
        for i in range(80):
            runner.messages.append(UserMsg(content=f"填充 {i}"))
        runner.messages.append(ToolMsg(tool_call_id="deep", name="/fs/read", content="ok"))
        runner._on_batch_done(runner.messages)

        tool_events = [e for e in _drain(q) if e.kind == "tool"]
        assert tool_events and tool_events[0].payload["arguments"] == '{"path": "z.txt"}'

    def test_previous_groups_are_not_reannounced(self, tmp_path: Path) -> None:
        """回归：第二轮不得把第一轮的工具结果再发一遍（那会显示成空参数）。"""
        from orange.message import AssistantMsg, ToolCall, ToolMsg, UserMsg

        runner = _runner(tmp_path, "twice")
        q = runner.subscribe()

        # 第一组
        c1 = ToolCall(id="a1", name="/fs/read", arguments='{"path": "one.txt"}')
        runner._traced_count = len(runner.messages)
        runner.messages.append(UserMsg(content="第一问"))
        runner._on_assistant(AssistantMsg(content=None, tool_calls=[c1]))
        runner.messages.append(AssistantMsg(content=None, tool_calls=[c1]))
        runner.messages.append(ToolMsg(tool_call_id="a1", name="/fs/read", content="一"))
        runner._on_batch_done(runner.messages)

        # 第二组
        c2 = ToolCall(id="b2", name="/fs/write", arguments='{"path": "two.txt"}')
        runner.messages.append(UserMsg(content="第二问"))
        runner._on_assistant(AssistantMsg(content=None, tool_calls=[c2]))
        runner.messages.append(AssistantMsg(content=None, tool_calls=[c2]))
        runner.messages.append(ToolMsg(tool_call_id="b2", name="/fs/write", content="二"))
        runner._on_batch_done(runner.messages)

        tool_events = [e for e in _drain(q) if e.kind == "tool"]
        # 恰好两条：每组一条，且参数都不是空的
        assert len(tool_events) == 2
        assert [e.payload["arguments"] for e in tool_events] == [
            '{"path": "one.txt"}',
            '{"path": "two.txt"}',
        ]


class TestFrontendContract:
    """静态检查前端与后端的事件契约，避免再次出现「静默中断」。

    前端已从单文件拆成 ``index.html`` + ``app.js`` + ``app.css``，所以契约检查
    必须覆盖两个源码文件：只查 HTML 会让 JS 里的绑定和事件分支逃过检查。
    """

    def setup_method(self) -> None:
        root = _STATIC if _STATIC.is_dir() else _STATIC.parent
        self.html = (root / "index.html").read_text(encoding="utf-8")
        self.js = (root / "app.js").read_text(encoding="utf-8")
    def test_no_orphan_element_binding(self) -> None:
        """每个被绑定的元素 id 必须真的存在于 HTML 里。

        之前 ``$("refresh-context").onclick`` 在元素删除后仍保留，抛出
        ``TypeError`` 让整段脚本停下，健康轮询随之失效。
        """
        import re

        assert self.js.strip(), "app.js 不应为空"
        # ``$("x")`` lookups anywhere in the script must resolve against the DOM.
        used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', self.js))
        # Elements created at runtime are referenced by the same helper, so only
        # ids that are never created dynamically need to exist in the markup.
        created = set(re.findall(r'\.id\s*=\s*"([A-Za-z0-9_-]+)"', self.js))
        assert used, "未找到任何元素引用，检查选择器格式"

        for element_id in used - created:
            assert f'id="{element_id}"' in self.html, f"引用到不存在的元素：{element_id}"

    def test_conversation_does_not_render_tool_cards(self) -> None:
        # 工具调用/结果只进轨迹面板，不再污染对话区
        assert "appendToolCall" not in self.js
        assert "appendToolResult" not in self.js

    def test_trace_renders_tool_arguments_and_result(self) -> None:
        assert "trace-row" in self.js
        assert 'item.kind === "tool"' in self.js
        # 参数与结果各有独立的 <pre>，样式类名用于区分来源
        assert '"pre"' in self.js or "el(\"pre\"" in self.js

    def test_sse_uses_event_source(self) -> None:
        assert "new EventSource(" in self.js
        # 断点续传：已加载的轨迹不应被重复回放
        assert "after_seq=" in self.js

    def test_state_event_updates_controls(self) -> None:
        # 「执行中」必须能通过 state 事件解除，否则界面永久卡住
        assert "updateSendState" in self.js
        assert 'case "state":' in self.js

    def test_delta_events_render_incrementally(self) -> None:
        """流式预览是独立通道：delta 追加、assistant 校准。"""
        assert 'case "delta":' in self.js
        assert "appendDelta" in self.js
        assert "finishDelta" in self.js

    def test_delta_is_excluded_from_trace(self) -> None:
        """流式碎片不进轨迹面板，否则一次回复数百条会淹没工具审计。"""
        assert 'if (kind === "delta") return;' in self.js

    def test_settings_ui_is_wired(self) -> None:
        assert 'id="settings-overlay"' in self.html
        assert "/api/settings" in self.js
        assert "restart_required" in self.js

    def test_offline_failure_is_explained(self) -> None:
        """服务挂了不能只显示 ``Failed to fetch``。

        用户看到原始 fetch 报错完全无法判断该做什么；把它换成可执行的提示，
        并在设置弹窗里给出重试按钮，是唯一能自救的路径。
        """
        assert "无法连接服务" in self.js
        assert "err.offline = true" in self.js
        # the settings dialog must offer a retry rather than a dead end
        assert "重试" in self.js
        assert "loadSettings" in self.js

    def test_brand_uses_the_logo_asset(self) -> None:
        """顶栏图标必须是真实的 SVG 资源，而不是内联占位符。"""
        root = _STATIC if _STATIC.is_dir() else _STATIC.parent
        logo = root / "logo.svg"
        assert logo.is_file(), "缺少 logo.svg，安装后图标会 404"
        assert "/static/logo.svg" in self.html
        assert 'type="image/svg+xml"' in self.html
