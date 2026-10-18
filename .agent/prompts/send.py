"""向运行中的 Harness Web 会话发送一条中文消息（UTF-8 安全）。

用法::

    python .agent/prompts/send.py <session_id> <text_file>

为什么单独写脚本：PowerShell 的 here-string 与管道默认编码会破坏中文，
直接以 UTF-8 字节 POST 才能保证服务端收到原文。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8012"


def main() -> int:
    session_id, text_file = sys.argv[1], sys.argv[2]
    text = open(text_file, encoding="utf-8").read().strip()
    payload = json.dumps({"text": text, "channel": "chat"}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE}/api/sessions/{session_id}/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
