"""统计每个会话的消息构成，按「工具结果」与其他角色的字符数占比分析。

数据来源：Web 服务 /api/sessions/{id}。会话已在服务重启后丢失时自动跳过。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8012"
SESSIONS = ["7115dab9aa9a", "a9a24e3b7eea", "6d89e31532cc", "e4bd9a5670e5"]


def fetch(session_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE}/api/sessions/{session_id}", timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"{session_id}: HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"{session_id}: {type(exc).__name__}")
        return None


def main() -> int:
    for session_id in SESSIONS:
        data = fetch(session_id)
        if data is None:
            continue
        messages = data["messages"]
        buckets: dict[str, int] = {}
        counts: dict[str, int] = {}
        for message in messages:
            role = message["role"]
            if role == "assistant":
                size = len(message.get("content") or "") + sum(
                    len(str(call.get("arguments")) or "") for call in message.get("tool_calls") or []
                )
            else:
                size = len(message.get("content") or "")
            buckets[role] = buckets.get(role, 0) + size
            counts[role] = counts.get(role, 0) + 1

        total = sum(buckets.values()) or 1
        print("=" * 62)
        print(f"会话 {session_id}  状态={data['state']}  token={data['tokens']}/{data['capacity']}")
        print(f"消息数: {len(messages)}")
        for role in sorted(buckets, key=lambda r: -buckets[r]):
            share = buckets[role] / total * 100
            bar = "#" * int(share / 2)
            print(f"  {role:<10} {counts[role]:>3} 条  {buckets[role]:>7} 字符  {share:>5.1f}% {bar}")

        tools = [m for m in messages if m["role"] == "tool"]
        if tools:
            sizes = sorted((len(m.get("content") or "") for m in tools), reverse=True)
            print(f"  工具结果: 最大 {sizes[0]} 字符, 中位 {sizes[len(sizes)//2]}, 总计 {sum(sizes)}")
            big = [s for s in sizes if s > 2000]
            print(f"  其中 >2000 字符的有 {len(big)} 条，占工具结果字符的 "
                  f"{sum(big)/max(sum(sizes),1)*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
