"""验证 C1 截断已接入 executor：超长工具结果会被截到 max_chars 以内。

修复前：8021 字符的整文件读取原样进入 ToolMsg，占满上下文。
修复后：应被截到 4000 以内，且保留头尾。
"""

from __future__ import annotations

import sys

from harness.execution import ExecResult, plan_batch
from harness.executor import execute_batch
from harness.message import ToolCall


class BigTool:
    """返回超长文本的假工具。"""

    concurrency = "safe"
    idempotent = True

    @staticmethod
    def fn() -> str:
        return "HEAD-" + ("x" * 20_000) + "-TAIL"


def main() -> int:
    call = ToolCall(id="c1", name="/tool/fake/big", arguments="{}")
    plan = plan_batch([call], lambda _name: "safe")

    results: dict[str, ExecResult] = execute_batch(plan, lambda _call: BigTool())

    content = results["c1"].content
    ok_truncated = len(content) <= 4000
    ok_has_head = content.startswith("HEAD-")
    ok_has_tail = content.rstrip().endswith("-TAIL")
    ok_has_notice = "字符已省略" in content

    print(f"原始长度: {len('HEAD-' + 'x' * 20_000 + '-TAIL')}")
    print(f"截断后长度: {len(content)}")
    print(f"[{'PASS' if ok_truncated else 'FAIL'}] 长度 <= 4000")
    print(f"[{'PASS' if ok_has_head else 'FAIL'}] 保留头部")
    print(f"[{'PASS' if ok_has_tail else 'FAIL'}] 保留尾部")
    print(f"[{'PASS' if ok_has_notice else 'FAIL'}] 含省略提示")

    # 短结果必须逐字节原样返回
    class SmallTool:
        concurrency = "safe"
        idempotent = True

        @staticmethod
        def fn() -> str:
            return "short"

    small = plan_batch([ToolCall(id="c2", name="/tool/fake/small", arguments="{}")], lambda _n: "safe")
    small_result = execute_batch(small, lambda _c: SmallTool())["c2"]
    ok_small = small_result.content == "short"
    print(f"[{'PASS' if ok_small else 'FAIL'}] 短结果原样返回")

    passed = sum([ok_truncated, ok_has_head, ok_has_tail, ok_has_notice, ok_small])
    print(f"\n{passed}/5 通过")
    return 0 if passed == 5 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
