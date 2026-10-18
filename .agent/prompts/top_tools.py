"""列出各会话中最"吃上下文"的单条工具结果，定位来源。"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8012"
SESSIONS = ["a9a24e3b7eea", "e4bd9a5670e5", "6d89e31532cc"]


def main() -> int:
    for session_id in SESSIONS:
        try:
            with urllib.request.urlopen(f"{BASE}/api/sessions/{session_id}", timeout=8) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue

        messages = data["messages"]
        rows = []
        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            content = message.get("content") or ""
            rows.append((len(content), index, message.get("name", "?"), content))

        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        print("=" * 78)
        print(f"会话 {session_id}  token={data['tokens']}  工具结果 {len(rows)} 条 / {total} 字符")
        print(f"{'字符':>6}  {'msg#':>4}  {'工具':<24} 首行摘要")
        print("-" * 78)
        for size, index, name, content in rows[:8]:
            head = content.strip().splitlines()[0][:44] if content.strip() else "(空)"
            print(f"{size:>6}  {index:>4}  {name:<24} {head}")
        top5 = sum(r[0] for r in rows[:5])
        print(f"→ 前 5 条占工具结果总量的 {top5 / max(total, 1) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
