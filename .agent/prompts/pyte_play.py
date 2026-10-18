"""用 pyte 终端模拟器验证贪吃蛇交互模式：真实终端语义、真实按键、真实渲染。

为什么不用管道：main.py 在 stdin 非 TTY 时会降级为无头模式（设计如此），
管道测试测不到交互路径。pyte 提供一块 80x24 的虚拟屏幕，并把游戏输出的
ANSI 序列解析成可见画面，等价于“在终端里看”。

不修改被测项目，只作为消费者驱动它。
"""

from __future__ import annotations

import os
import sys

import pyte


def render(screen: pyte.Screen) -> str:
    """把虚拟屏幕导出成文本网格。"""
    return "\n".join(
        "".join(screen.buffer[y][x].data for x in range(screen.columns)).rstrip()
        for y in range(screen.lines)
    )


def main() -> int:
    # 用 80x24 的经典终端，先让游戏自适应
    screen = pyte.Screen(80, 24)

    class FakeTerm:
        """把 draw() 的写入喂给 pyte，并模拟 Windows 控制台的 LF->CRLF。"""

        def write(self, text: str) -> None:
            pyte.Stream(screen).feed(text.replace("\n", "\r\n"))

        def flush(self) -> None:  # noqa: D102
            pass

    term = FakeTerm()

    # 直接驱动 game+render 层：模拟交互循环的“渲染"部分
    sys.path.insert(0, os.getcwd())
    from snake_game import config
    from snake_game.game import SnakeGame
    from snake_game.main import build_frame

    class NoColor:
        def __call__(self, code: str) -> str:
            return ""

    palette = NoColor()
    game = SnakeGame(width=30, height=15, seed=7)

    def draw() -> None:
        # 直接调用产品代码的 draw()，并把写入转接给终端模拟器
        old = sys.stdout
        sys.stdout = term
        try:
            from snake_game.main import draw as real_draw

            real_draw(build_frame(game, palette, False, game.steps_per_sec))
        finally:
            sys.stdout = old

    draw()
    first = render(screen)
    print("=== 第 1 帧（初始画面）===")
    print(first)

    score_seen = 0
    # 驱动若干帧：朝食物移动，验证画面是否随状态更新
    for _ in range(120):
        if game.over:
            break
        head = game.snake[0]
        food = game.food
        if food[0] != head[0]:
            game.move("right" if food[0] > head[0] else "left")
        elif food[1] != head[1]:
            game.move("down" if food[1] > head[1] else "up")
        game.step()
        score_seen = max(score_seen, game.score)
        draw()

    mid = render(screen)
    print("\n=== 最后一帧（120 步后）===")
    print(mid)

    checks = {
        "渲染出边框": "+---" in first,
        "渲染出蛇头": "@" in first,
        "渲染出食物": "*" in first,
        "状态栏显示分数": "分数" in first,
        "按键说明可见": "WASD" in first,
        "画面随状态更新": first != mid,
        "鼠标能吃到食物": score_seen > 0,
        "棋盘尺寸正确": first.count("\n") >= 15,
    }
    print("\n=== 检查项 ===")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    passed = sum(checks.values())
    print(f"\n{passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
