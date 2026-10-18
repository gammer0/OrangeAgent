"""读取 Harness Web 会话的最新消息（控制台按 UTF-8 安全输出）。"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8012"


def main() -> int:
    session_id = sys.argv[1]
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    with urllib.request.urlopen(f"{BASE}/api/sessions/{session_id}", timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    print(f"state={data['state']} msgs={data['message_count']} tokens={data['tokens']}")
    for message in data["messages"][-tail:]:
        role = message["role"]
        calls = message.get("tool_calls") or []
        if calls:
            print(f"[{role}] tool_calls: " + ", ".join(c["name"] for c in calls))
            for call in calls:
                print("    " + str(call.get("arguments"))[:300])
        else:
            body = (message.get("content") or "").strip()
            print(f"[{role}] {body[:1500]}")
        print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
