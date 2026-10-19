"""分块压缩回归测试：超长输入必须分块处理，不得硬截断丢数据。

历史问题：``_summarize`` 用 ``text[:24000]`` 直接截断，超出部分静默丢失。
现改为按 SUMMARY_INPUT_CHARS 切块逐块压缩，再合并。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / ".."))

from apps.server.sessions import SessionRunner  # noqa: E402


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeProxy:
    """记录每次调用的输入，返回可识别的摘要。"""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def chat(self, messages, **kwargs) -> Any:  # noqa: ANN001, ANN003
        payload = messages[-1].content
        self.inputs.append(payload)
        return _FakeResponse(f"[摘要{len(self.inputs)}] 共 {len(payload)} 字符")


def _runner(proxy: _FakeProxy) -> SessionRunner:
    """构造最小 SessionRunner，只测 _summarize。"""

    class _R(SessionRunner):
        pass

    runner = _R.__new__(_R)
    runner.session_id = "test"
    runner.model = "fake"
    runner.proxy = proxy  # type: ignore[assignment]
    return runner


def test_short_input_single_call() -> None:
    """输入未超限时只调用一次。"""
    proxy = _FakeProxy()
    runner = _runner(proxy)
    text = "x" * 1000
    runner._summarize(text)
    assert len(proxy.inputs) == 1


def test_long_input_is_chunked_not_truncated() -> None:
    """超长输入分块处理，所有内容都会进入某次调用。"""
    proxy = _FakeProxy()
    runner = _runner(proxy)
    limit = runner.SUMMARY_INPUT_CHARS
    text = "".join(f"[chunk-{i}]" for i in range(4000))  # 远超 limit
    runner._summarize(text)

    assert len(proxy.inputs) > 1, "超长输入必须分块，不能一次吞下"
    # 每块都不超过上限
    assert all(len(chunk) <= limit for chunk in proxy.inputs)
    # 关键：内容没有被静默丢弃——首尾标记都应出现在某次调用里
    assert any("[chunk-0]" in chunk for chunk in proxy.inputs)
    assert any("[chunk-3999]" in chunk for chunk in proxy.inputs)


def test_chunks_are_merged_after_partial_summaries() -> None:
    """多块压缩后还有一次合并调用。"""
    proxy = _FakeProxy()
    runner = _runner(proxy)
    text = "y" * (runner.SUMMARY_INPUT_CHARS * 2 + 100)
    result = runner._summarize(text)

    assert len(proxy.inputs) >= 3, "应包含分段调用 + 至少一次合并"
    assert "摘要" in result
