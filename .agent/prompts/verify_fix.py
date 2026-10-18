"""调用真实 draw() 验证残留是否消除（而不是模拟旧实现）。

核心不变量：在同一个终端连续画 N 帧后的画面，必须等于「只画最后一帧」的新鲜画面。
任何残留都会让两者不同。
"""

from __future__ import annotations

import sys

import pyte

sys.path.insert(0, ".")

from snake_game.game import SnakeGame  # noqa: E402
from snake_game.main import build_frame, draw  # noqa: E402


class FakeTerm:
    """把 draw() 的写入喂给 pyte；\\n -> \\r\\n 模拟 Windows 控制台 ONLCR。"""

    def __init__(self, screen: pyte.Screen) -> None:
        self.screen = screen
        self.stream = pyte.Stream(screen)

    def write(self, text: str) -> None:
        self.stream.feed(text.replace("\n", "\r\n"))

    def flush(self) -> None:
        pass


class NoColor:
    def __call__(self, code: str) -> str:
        return ""


def render(screen: pyte.Screen) -> list[str]:
    return [
        "".join(screen.buffer[y][x].data for x in range(screen.columns)).rstrip()
        for y in range(screen.lines)
    ]


def paint(term: FakeTerm, game: SnakeGame) -> None:
    """调用产品代码的 draw()。"""
    old = sys.stdout
    sys.stdout = term  # type: ignore[assignment]
    try:
        draw(build_frame(game, NoColor(), False, game.steps_per_sec))
    finally:
        sys.stdout = old  # type: ignore[assignment]


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # --- 场景 1：过去的问题帧，状态行从 right 变 up ---
    game = SnakeGame(width=10, height=6, seed=7)
    term = FakeTerm(pyte.Screen(80, 16))
    paint(term, game)
    first = render(term.screen)[8]
    # 强制转向并推进，使方向从 right 变为 up
    game.move("up")
    game.step()
    paint(term, game)
    second = render(term.screen)[8]
    checks.append(("状态行 right->up 无残留", "ght" not in second, f"{second!r}"))

    # --- 场景 2：多帧叠加 == 新鲜屏（通用不变量）---
    game = SnakeGame(width=10, height=6, seed=7)
    stacked = FakeTerm(pyte.Screen(80, 16))
    for _ in range(60):
        if not game.over:
            head = game.snake[0]
            food = game.food
            if food[0] != head[0]:
                game.move("right" if food[0] > head[0] else "left")
            elif food[1] != head[1]:
                game.move("down" if food[1] > head[1] else "up")
            game.step()
        paint(stacked, game)

    fresh = FakeTerm(pyte.Screen(80, 16))
    paint(fresh, game)
    same = [r for r in render(stacked.screen)] == [r for r in render(fresh.screen)]
    checks.append(("多帧叠加与新鲜屏一致", same, ""))

    # --- 场景 3：蛇变长后重开，棋盘区域无残留 ---
    game2 = SnakeGame(width=10, height=6, seed=7)
    term2 = FakeTerm(pyte.Screen(80, 16))
    for _ in range(40):
        if game2.over:
            break
        head = game2.snake[0]
        food = game2.food
        if food[0] != head[0]:
            game2.move("right" if food[0] > head[0] else "left")
        elif food[1] != head[1]:
            game2.move("down" if food[1] > head[1] else "up")
        game2.step()
        paint(term2, game2)
    game2.reset()
    paint(term2, game2)
    fresh2 = FakeTerm(pyte.Screen(80, 16))
    paint(fresh2, game2)
    same2 = render(term2.screen) == render(fresh2.screen)
    checks.append(("重开后与新鲜屏一致", same2, ""))

    print(f"场景1 状态行: {first!r} -> {second!r}")
    print()
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n{passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
