"""贪吃蛇终端入口：参数解析 + 固定时间步主循环 + 整屏重绘。

运行：
    python -m snake_game.main                  # 交互式游玩
    python snake_game/main.py                  # 同上（脚本方式）
    python -m snake_game.main --headless 200   # 无头自检：跑 200 步就退出

逻辑全部在 game.py，本文件只负责输入、时间与渲染。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

try:  # 以包方式运行：python -m snake_game.main
    from . import config
    from .game import SnakeGame
except ImportError:  # 以脚本方式运行：python snake_game/main.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from snake_game import config
    from snake_game.game import SnakeGame

try:
    import msvcrt  # type: ignore[import-not-found]  # Windows 专有
except ImportError:  # pragma: no cover - 非 Windows
    msvcrt = None

if msvcrt is None:  # pragma: no cover - 非 Windows
    import select
    import termios
    import tty

_ARROWS = {"H": "up", "P": "down", "K": "left", "M": "right"}
_ESCAPE_ARROWS = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}
_IS_WINDOWS = os.name == "nt"
_ALL_DIRECTIONS = ("up", "down", "left", "right")


# --------------------------------------------------------------------- 终端
def enable_ansi() -> None:
    """Windows 控制台默认不解析 ANSI，这里打开 VT 处理；其他平台无需处理。"""
    if not _IS_WINDOWS:
        return
    try:  # pragma: no cover - 依赖具体控制台
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001 - 控制台不支持就退化为无颜色
        pass


class KeyReader:
    """非阻塞读键，统一成 'up'/'down'/'left'/'right' 或单个字符。"""

    def __init__(self) -> None:
        self._saved: list[object] | None = None

    def __enter__(self) -> KeyReader:
        if msvcrt is None and sys.stdin.isatty():  # pragma: no cover - POSIX
            fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._saved is not None:  # pragma: no cover - POSIX
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            self._saved = None

    def poll(self) -> list[str]:
        if msvcrt is not None:
            return self._poll_windows()
        return self._poll_posix()  # pragma: no cover - POSIX

    @staticmethod
    def _poll_windows() -> list[str]:
        keys: list[str] = []
        while msvcrt is not None and msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                arrow = _ARROWS.get(msvcrt.getwch())
                if arrow:
                    keys.append(arrow)
            else:
                keys.append(ch)
        return keys

    @staticmethod
    def _poll_posix() -> list[str]:  # pragma: no cover - POSIX
        keys: list[str] = []
        fd = sys.stdin.fileno()
        while select.select([fd], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                if select.select([fd], [], [], 0.005)[0]:
                    arrow = _ESCAPE_ARROWS.get(sys.stdin.read(2))
                    if arrow:
                        keys.append(arrow)
            elif ch:
                keys.append(ch)
        return keys


class Palette:
    """颜色开关：--no-color 时所有颜色码退化为空串。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, code: str) -> str:
        return code if self.enabled else ""


# --------------------------------------------------------------------- 渲染
def build_frame(game: SnakeGame, color: Palette, paused: bool, fps: float) -> list[str]:
    """生成本帧的所有行；每帧整屏重绘，避免逐字符刷新造成闪烁。"""
    snake = game.snake
    head = snake[0]
    body = set(snake[1:])
    food = game.food
    inner = game.width * 2
    border = color(config.COLOR_BORDER)
    reset = color(config.COLOR_RESET)
    edge = config.BORDER_CORNER + config.BORDER_H * inner + config.BORDER_CORNER

    lines = [border + edge + reset]
    for y in range(game.height):
        row = [border + config.BORDER_V + reset]
        for x in range(game.width):
            cell = (x, y)
            if cell == head:
                row.append(color(config.COLOR_HEAD) + config.CH_HEAD + reset + " ")
            elif cell in body:
                row.append(color(config.COLOR_BODY) + config.CH_BODY + reset + " ")
            elif cell == food:
                row.append(color(config.COLOR_FOOD) + config.CH_FOOD + reset + " ")
            else:
                row.append(config.CH_EMPTY + " ")
        row.append(border + config.BORDER_V + reset)
        lines.append("".join(row))
    lines.append(border + edge + reset)

    text = color(config.COLOR_TEXT)
    lines.append(
        f"{text}分数 {game.score}    蛇长 {len(snake)}    "
        f"速度 {fps:.1f} 步/秒    方向 {game.direction}{reset}"
    )
    lines.append(f"{text}操作: WASD/方向键 移动   p 暂停   r 重开   q 退出{reset}")
    if paused and not game.over:
        lines.append(color(config.COLOR_WIN) + "-- 已暂停，按 p 继续 --" + reset)
    if game.over:
        if game.won:
            lines.append(color(config.COLOR_WIN) + "恭喜！你填满了整个棋盘！" + reset)
        else:
            lines.append(color(config.COLOR_OVER) + "游戏结束" + reset)
        lines.append(
            color(config.COLOR_OVER)
            + f"最终得分 {game.score}   ——   按 r 重开，按 q 退出"
            + reset
        )
    return lines


def draw(lines: list[str]) -> None:
    """光标归位后一次性写整屏，再清掉尾部残留。"""
    sys.stdout.write("\x1b[H" + "\n".join(lines) + "\n\x1b[J")
    sys.stdout.flush()


# --------------------------------------------------------------------- 玩法
def run_interactive(game: SnakeGame, color: Palette) -> int:
    """交互主循环：固定时间步推进，按键非阻塞轮询。"""
    enable_ansi()
    sys.stdout.write("\x1b[2J\x1b[?25l")  # 清屏 + 隐藏光标
    paused = False
    acc = 0.0
    last = time.perf_counter()
    needs_draw = True
    exit_code = 0
    try:
        with KeyReader() as reader:
            running = True
            while running:
                for key in reader.poll():
                    if key in config.KEYS_QUIT:
                        running = False
                        break
                    if key in config.KEYS_RESTART:
                        game.reset()
                        paused = False
                        acc = 0.0
                        needs_draw = True
                    elif key in config.KEYS_PAUSE:
                        paused = not game.over and not paused
                        needs_draw = True
                    elif key in config.DIRECTION_KEYS and not paused and not game.over:
                        game.move(config.DIRECTION_KEYS[key])

                now = time.perf_counter()
                delta = now - last
                last = now
                if not paused and not game.over:
                    acc += delta
                    interval = game.frame_interval
                    while acc >= interval and not game.over:
                        game.step()
                        acc -= interval
                        interval = game.frame_interval
                        needs_draw = True
                if needs_draw:
                    draw(build_frame(game, color, paused, game.steps_per_sec))
                    needs_draw = False
                time.sleep(config.MIN_FRAME_SLEEP)
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        sys.stdout.write("\x1b[?25h\n")  # 恢复光标
        sys.stdout.flush()
    return exit_code


def run_headless(game: SnakeGame, steps: int) -> int:
    """无头模式：随机方向推进指定步数，用于自检与冒烟测试。"""
    rng = random.Random(game.seed)
    done = 0
    for _ in range(steps):
        if game.over:
            break
        if rng.random() < 0.5:
            game.move(rng.choice(_ALL_DIRECTIONS))
        game.step()
        done += 1
    print(
        f"headless: steps={done} score={game.score} len={len(game.snake)} "
        f"over={game.over} won={game.won} food={game.food}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="终端贪吃蛇（Python 标准库实现）")
    parser.add_argument("--width", type=int, default=config.WIDTH, help="棋盘宽（格）")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="棋盘高（格）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    parser.add_argument(
        "--headless",
        type=int,
        nargs="?",
        const=200,
        default=None,
        metavar="STEPS",
        help="无头自检模式，跑 STEPS 步（默认 200）后退出",
    )
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        game = SnakeGame(width=args.width, height=args.height, seed=args.seed)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    # 无头模式，或 stdin 不是终端（例如被自检工具重定向）时自动走无头
    steps = args.headless
    if steps is None and not sys.stdin.isatty():
        steps = 200
        print("检测到非交互式标准输入，自动切换为无头模式（--headless 200）。")
    if steps is not None:
        return run_headless(game, max(steps, 0))

    return run_interactive(game, Palette(enabled=not args.no_color))


if __name__ == "__main__":
    raise SystemExit(main())
