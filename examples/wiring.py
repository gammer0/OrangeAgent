"""端到端接线示例：装配器 → 适配器 → 反向代理 → 主循环。

默认**真实调用**模型（API Key 从环境变量读取）。

运行方式::

    # 真实调用（需要 .env 或进程环境里有 DEEPSEEK_API_KEY）
    uv run python examples/wiring.py

    # 离线演练：使用假传输，不发网络请求
    uv run python examples/wiring.py --dry-run

    # 覆盖模型
    uv run python examples/wiring.py --model deepseek-chat

它演示三件事：

1. ``ContextAssembler`` 装配 ``SystemMsg`` 五段，并在水位超标时压缩；
2. ``OpenAIAdapter`` 把内部消息投影成请求体，``ReverseProxy`` 负责限流与重试；
3. ``JevRouter`` 由主模型代为选择工具分支，失败即回退全量。

密钥只从环境变量读取（见 ``harness.config``），不写入代码、不打印明文。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Any

from harness.config import ConfigError, build_proxy, load_llm_config
from harness.context import ContextAssembler, SystemPromptParts, clear_after_compress
from harness.discovery import TreeSessionState
from harness.execution import ExecResult
from harness.executor import GroupReport
from harness.inbox import Inbox
from harness.jev import JevRouter
from harness.loop import AgentLoop, StepController
from harness.message import Message, ToolCall, UserMsg
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

#: 工具声明（OpenAI ``tools`` 格式）。真实场景由 FlatStore 的 ToolDef.schema 生成。
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "/tool/os/file/read",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "/tool/os/file/list",
            "description": "列出指定目录下的文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def resolver(call: ToolCall) -> Any:
    """把 ToolCall 解析为带 fn / concurrency / idempotent 的可执行定义。"""

    class _Tool:
        concurrency = "safe"
        idempotent = True

        @staticmethod
        def fn(**kwargs: Any) -> str:
            return f"已执行 {call.name}，参数 {call.arguments}"

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
            tool = resolver_fn(call)
            try:
                content = str(tool.fn())
                results[call.id] = ExecResult(ok=True, content=content, duration_ms=0)
            except Exception as exc:  # noqa: BLE001 - 演示用，统一收敛为 3c
                results[call.id] = ExecResult(
                    ok=False,
                    content=f"执行失败: {exc}",
                    error_type="3c",
                    retryable=False,
                    duration_ms=0,
                )
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
# 2) 模型通道：真实 HTTP 或离线假传输
# ---------------------------------------------------------------------------


class FakeTransport:
    """把固定响应按脚本返回，用于 ``--dry-run``。"""

    def __init__(self, replies: list[tuple[str | None, list[dict[str, Any]]]]) -> None:
        self.replies = list(replies)

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> dict[str, Any]:
        content, calls = self.replies.pop(0) if self.replies else ("完成", [])
        return {
            "model": json.get("model", ""),
            "choices": [{"message": {"content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }


DRY_RUN_REPLIES: list[tuple[str | None, list[dict[str, Any]]]] = [
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
    ("已读取配置并完成检查。", []),
]


def build_fake_proxy() -> ReverseProxy:
    """构建离线演练代理：不读取密钥、不发网络请求。"""
    return ReverseProxy(
        providers=[
            ProviderConfig(
                name="dry-run",
                base_url="https://dry-run.invalid",
                model="dry-run",
                api_key="",
                adapter=OpenAIAdapter(),
            )
        ],
        rate_window=RateWindow(max_requests=60, max_tokens=200_000),
        health={},
        transport=FakeTransport(list(DRY_RUN_REPLIES)),
    )


def build_real_proxy(model_override: str | None) -> ReverseProxy:
    """按环境变量构建真实代理。

    Raises:
        ConfigError: 缺少或非法配置。
    """
    config = load_llm_config()
    if model_override:
        config = replace(config, model=model_override)
    print(f"接入: {config.base_url}  模型: {config.model}  密钥: {config.masked_key()}")
    return build_proxy(config)


# ---------------------------------------------------------------------------
# 3) 端到端
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Harness 端到端接线示例")
    parser.add_argument(
        "--dry-run", action="store_true", help="使用假传输离线演练，不发真实请求"
    )
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument(
        "--goal",
        default="读一下 pyproject.toml 并检查依赖是否合规",
        help="本次会话目标",
    )
    parser.add_argument("--capacity", type=int, default=32_768, help="上下文容量")
    parser.add_argument("--seen-tokens", type=int, default=5000, help="路由判定用历史 token")
    parser.add_argument("--tool-count", type=int, default=40, help="路由判定用工具总数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """跑一轮完整会话并打印结果。"""
    args = parse_args(argv)
    trace_state = TreeSessionState()

    try:
        proxy = build_fake_proxy() if args.dry_run else build_real_proxy(args.model)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        print("提示：复制 .env.example 为 .env 并填入真实 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    assembler = ContextAssembler(
        capacity=args.capacity,
        summarizer=lambda text: f"[摘要] 已折叠 {len(text)} 字符的历史",
    )

    system_msg = assembler.build_system(
        SystemPromptParts(
            tool_index=TOOL_INDEX,
            working_dir="C:/Users/USER/Desktop/agent",
            goal=args.goal,
        )
    )

    class ProxyClient:
        """把 ReverseProxy 适配成编排层要求的 ``chat(messages)`` 协议。

        关键点：必须随请求声明 ``tools``，否则模型会把工具调用当作普通文本输出
        （表现为响应里出现 ``tool_calls`` 标记文本而不是结构化 ``tool_calls``）。
        """

        def __init__(self, proxy: ReverseProxy) -> None:
            self.proxy = proxy

        def chat(self, messages: list[Message]) -> Any:
            return self.proxy.chat(
                messages,
                session_id="demo-session",
                tools=TOOL_SPECS,
            ).message

    model = ProxyClient(proxy)

    loop = AgentLoop(
        model=model,
        inbox=Inbox(),
        executor=executor,
        resolver=resolver,
        system_msg=system_msg,
        controller=StepController(max_steps=6),
        assembler=assembler,
        router=JevRouter(chat=model.chat),
        branch_provider=lambda: BRANCHES,
        tool_count_provider=lambda: args.tool_count,
        history_tokens_provider=lambda n: args.seen_tokens,
    )

    history: list[Message] = [system_msg, UserMsg(content=args.goal)]
    result = loop.run(history)

    print("\n=== 运行结果 ===")
    print(f"模式: {'离线演练' if args.dry_run else '真实调用'}")
    print(f"步数: {result.steps}  终止原因: {result.stop_reason}")
    print(f"最终答案: {result.final_text}")
    if result.route is not None:
        print(f"路由决策: branch={result.route.branch} fallback={result.route.fallback}")
    print(f"消息条数: {len(result.messages)}")
    print(f"水位占用: {assembler.ratio(result.messages):.1%}")

    sealed = ContextAssembler(capacity=300, summarizer=lambda t: "[摘要] 已折叠")
    compressed = sealed.assemble(
        result.messages, clear_trace_state=clear_after_compress(trace_state)
    )
    print(
        f"压缩演示: action={compressed.action} compressed={compressed.compressed} "
        f"条数 {len(result.messages)} → {len(compressed.messages)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
