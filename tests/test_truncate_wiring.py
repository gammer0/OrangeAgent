"""C1 截断已接入执行器的回归测试。

历史问题：``truncate_result`` 只定义未调用，8021 字符的整文件读取
原样进入上下文，导致会话 token 冲破 capacity。
"""

from __future__ import annotations

from harness.execution import plan_batch
from harness.executor import execute_batch
from harness.message import ToolCall


class _BigTool:
    concurrency = "safe"
    idempotent = True

    @staticmethod
    def fn() -> str:
        return "HEAD-" + "x" * 20_000 + "-TAIL"


class _SmallTool:
    concurrency = "safe"
    idempotent = True

    @staticmethod
    def fn() -> str:
        return "short"


def _run(tool: object, call_id: str = "c1") -> str:
    plan = plan_batch([ToolCall(id=call_id, name="/tool/fake/x", arguments="{}")], lambda _n: "safe")
    return execute_batch(plan, lambda _c: tool)[call_id].content


def test_long_result_is_truncated_to_default_limit() -> None:
    """超长结果被截到 4000 字符以内。"""
    content = _run(_BigTool())
    assert len(content) <= 4_000
    assert "字符已省略" in content


def test_truncation_keeps_head_and_tail() -> None:
    """截断保留头尾，便于定位问题。"""
    content = _run(_BigTool())
    assert content.startswith("HEAD-")
    assert content.rstrip().endswith("-TAIL")


def test_short_result_is_unchanged() -> None:
    """短结果逐字节原样返回。"""
    assert _run(_SmallTool()) == "short"


def test_custom_max_chars_is_honored() -> None:
    """调用方可覆盖上限。"""
    plan = plan_batch([ToolCall(id="c9", name="/tool/fake/x", arguments="{}")], lambda _n: "safe")
    content = execute_batch(plan, lambda _c: _BigTool(), max_chars=500)["c9"].content
    assert len(content) <= 500


def test_failed_result_is_also_truncated() -> None:
    """失败路径（3c）同样受上限约束，避免 traceback 撑爆上下文。"""

    class _Boom:
        concurrency = "safe"
        idempotent = False

        @staticmethod
        def fn() -> str:
            raise RuntimeError("y" * 20_000)

    content = _run(_Boom(), "cb")
    assert len(content) <= 4_000
