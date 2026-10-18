"""端到端接线示例：装配器 → 适配器 → 反向代理 → 主循环。

运行方式::

    uv run python examples/wiring.py

本示例使用**假传输**，不发起真实网络请求；把 ``FakeTransport`` 换成
``OpenAITransport`` 并配置 ``DEEPSEEK_API_KEY`` 即可对接真实模型。

它演示三件事：

1. ``ContextAssembler`` 装配 ``SystemMsg`` 五段，并在水位超标时压缩；
2. ``OpenAIAdapter`` 把内部消息投影成请求体，``ReverseProxy`` 负责限流与重试；
3. ``JevRouter`` 由主模型代为选择工具分支，失败即回退全量。
"""

from __future__ import annotations

from typing import Any

from harness.context import ContextAssembler, SystemPromptParts, clear_after_compress
from harness.discovery import TreeSessionState
from harness.execution import ExecResult
from harness.executor import GroupReport
from harness.inbox import Inbox
from harness.jev import JevRouter
from harness.loop import AgentLoop, StepController
from harness.message import ToolCall, UserMsg
from harness.model import OpenAIAdapter, ProviderConfig, ReverseProxy
from harness.rate import RateWindow

# ---------------------------------------------------------------------------
# 1) 工具目录（真实场景来自 FlatStore + build_tree）
# ---------------------------------------------------------------------------

TOOL_INDEX = """/tool      — 操作系统能力
/tool/os   — 文件与进程
/skill     — 可加载的技能包
/mcp       — 外部 MCP 服务"""

BRANCHES = ["/tool/os", "/skill", "/mcp"]


def fake_tool(path: str) -> dict[str, Any]:
    """最小的工具执行体：按路径返回一句结果。"""
    return {"path": path, "result": f"已执行 {path}"}


def resolver(call: ToolCall) -> Any:
    """把 ToolCall 解析为带 fn / concurrency / idempotent 的可执行定义。"""

    class _Tool:
        concurrency = "safe"
        idempotent = True

        @staticmethod
        def fn(**kwargs: Any) -> str:
            return str(fake_tool(call.name))

    return _Tool()


def executor(
    plan: Any,
    resolver_fn: Any,
    on_group_done: Any = None,
    *,
    max_workers: int = 4,
) -> dict[str, ExecResult]:
    """同步执行批次；每组结束后触发安全点回调。"""
    results: dict[str, ExecResult] = {}
    for group in plan.groups:
        for call in group.calls:
            results[call.id] = ExecResult(ok=True, content=f"ok:{call.name}", duration_ms=0)
        if on_group_done is not None:
            on_group_done(
                GroupReport(
                    index=group.index,
                    group=group,
                    results=dict(results),
                    is_last=group.index == len(plan.groups) - 1,
                    plan=plan,
                )
            )
    return results


# ---------------------------------------------------------------------------
# 2) 模型通道：适配器 + 代理 + 假传输
# ---------------------------------------------------------------------------


class FakeTransport:
    """把同一份固定响应按脚本返回，避免真实网络请求。"""

    def __init__(self, replies: list[tuple[str | None, list[dict[str, Any]]]]) -> None:
        self.replies = list(replies)

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        content, calls = self.replies.pop(0) if self.replies else ("完成", [])
        return {
            "model": json.get("model", ""),
            "choices": [{"message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }


def build_proxy(replies: list[tuple[str | None, list[dict[str, Any]]]]) -> ReverseProxy:
    """装配一个使用假传输的反向代理。"""
    return ReverseProxy(
        providers=[
            ProviderConfig(
                name="deepseek",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-placeholder",
                adapter=OpenAIAdapter(),
            )
        ],
        rate_window=RateWindow(max_requests=60, max_tokens=200_000),
        health={},
        transport=FakeTransport(replies),
    )


# ---------------------------------------------------------------------------
# 3) 端到端
# ---------------------------------------------------------------------------


def main() -> None:
    """跑一轮完整会话并打印结果。"""
    trace_state = TreeSessionState()

    assembler = ContextAssembler(
        capacity=32_768,
        summarizer=lambda text: f"[摘要] 已折叠 {len(text)} 字符的历史",
    )

    system_msg = assembler.build_system(
        SystemPromptParts(
            tool_index=TOOL_INDEX,
            working_dir="C:/Users/USER/Desktop/agent",
            goal="演示装配与路由",
        )
    )
    print("=== SystemMsg 五段（前 3 行）===")
    print("\n".join(system_msg.content.splitlines()[:3]))

    proxy = build_proxy(
        [
            # 第一次请求：模型要求调用工具
            (
                None,
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "/tool/os/file/read",
                            "arguments": '{"path": "pyproject.toml"}',
                        },
                    }
                ],
            ),
            # 第二次请求：给出最终答案
            ("已读取配置并完成检查。", []),
        ]
    )

    class ProxyClient:
        """把 ReverseProxy 适配成编排层要求的 ``chat(messages)`` 协议。"""

        def __init__(self, proxy: ReverseProxy) -> None:
            self.proxy = proxy

        def chat(self, messages: Any) -> Any:
            return self.proxy.chat(messages, session_id="demo-session").message

    model = ProxyClient(proxy)

    loop = AgentLoop(
        model=model,
        inbox=Inbox(),
        executor=executor,
        resolver=resolver,
        system_msg=system_msg,
        controller=StepController(max_steps=5),
        assembler=assembler,
        router=JevRouter(chat=model.chat),
        branch_provider=lambda: BRANCHES,
        tool_count_provider=lambda: 40,
        history_tokens_provider=lambda n: 5000,
    )

    history = [system_msg, UserMsg(content="读一下 pyproject.toml 并检查依赖是否合规")]
    result = loop.run(history)

    print("\n=== 运行结果 ===")
    print(f"步数: {result.steps}  终止原因: {result.stop_reason}")
    print(f"最终答案: {result.final_text}")
    print(
        f"路由决策: branch={result.route.branch if result.route else None} "
        f"fallback={result.route.fallback if result.route else None}"
    )
    print(f"消息条数: {len(result.messages)}")
    print(f"水位占用: {assembler.ratio(result.messages):.1%}")
    print(f"已下发 schema 缓存: {sorted(trace_state.delivered)}")

    # 演示压缩：把容量压到极小，下一次装配会触发压缩
    sealed = ContextAssembler(capacity=200, summarizer=lambda t: "[摘要] 已折叠")
    assembler_result = sealed.assemble(
        result.messages, clear_trace_state=clear_after_compress(trace_state)
    )
    print(
        f"\n压缩演示: action={assembler_result.action} "
        f"compressed={assembler_result.compressed} "
        f"条数 {len(result.messages)} → {len(assembler_result.messages)}"
    )


if __name__ == "__main__":
    main()
