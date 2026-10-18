"""消费 Harness 会话的 SSE 事件流，打印最近事件（UTF-8 安全）。

用法::

    python .agent/prompts/events.py <session_id> [seconds]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8012"


def main() -> int:
    session_id = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    url = f"{BASE}/api/sessions/{session_id}/events"
    deadline = time.time() + seconds
    with urllib.request.urlopen(url, timeout=10) as response:
        while time.time() < deadline:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data: "):
                continue
            event = json.loads(text[6:])
            kind = event["kind"]
            payload = event["payload"]
            if kind == "error":
                print(f"ERROR: {payload.get('message')}")
            elif kind == "state":
                print(f"state: {payload}")
            elif kind == "tool":
                print(f"tool: {payload.get('path')} -> {str(payload.get('preview'))[:150]}")
            else:
                print(f"{kind}: {str(payload)[:200]}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
