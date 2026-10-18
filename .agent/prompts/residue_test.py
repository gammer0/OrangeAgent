"""验证帧变短时的渲染残留。

场景：上一帧某行是 "方向 right"（更长），下一帧是 "方向 up"（更短）。
若 draw() 只做 ESC[H + 整屏 + ESC[J，短行右侧会残留上一帧的字符。
"""

from __future__ import annotations

import sys

import pyte


def show(screen: pyte.Screen, rows: int = 4) -> list[str]:
    return [
        "".join(screen.buffer[y][x].data for x in range(screen.columns)).rstrip()
        for y in range(rows)
    ]


def feed(stream: pyte.Stream, lines: list[str]) -> None:
    payload = "\x1b[H" + "\n".join(lines) + "\n\x1b[J"
    stream.feed(payload.replace("\n", "\r\n"))


def main() -> int:
    screen = pyte.Screen(80, 12)
    stream = pyte.Stream(screen)

    feed(stream, ["aaaa", "方向 right"])
    before = show(screen)
    print("第一帧:", before)

    feed(stream, ["aaaa", "方向 up"])
    after = show(screen)
    print("第二帧:", after)

    residue = any("ght" in row for row in after)
    print(f"\n出现残留: {residue}")
    print("结论:", "存在渲染残留缺陷" if residue else "无残留")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
