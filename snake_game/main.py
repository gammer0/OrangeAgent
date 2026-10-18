"""贪吃蛇终端入口：参数解析 + 固定时间步主循环 + 整屏重绘。

用法：
    python snake_game/main.py                    # 交互模式
    python snake_game/main.py --headless 100      # 无头自检 100 步
    python snake_game/main.py --width 30 --seed 7 --no-color

终端文本一律使用 ASCII 字符，避免不同编码的控制台报 UnicodeEncodeError。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:  # 作为包运行：python -m snake_game.main
    from . import config
    from .game import SnakeGame
except ImportError:  # 直接运行：python snake_game/main.py
    import config  # type: ignore[no-redef]
    from game import SnakeGame  # type: ignore[no-redef]

Cell = Tuple[int, int]

#: 方向 -> (dx, dy)，与 game.py 的坐标约定一致（y 向下增长）
DIRS: Dict[str, Cell] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# ---- ANSI 控制序列 ----
CSI_CLEAR_SCREEN = "\x1b[2J"
CSI_HOME = "\x1b[H"
CSI_CLEAR_EOL = "\x1b[K"
CSI_CLEAR_BELOW = "\x1b[J"
CSI_HIDE_CURSOR = "\x1b[?25l"
CSI_SHOW_CURSOR = "\x1b[?25h"

# ---- 键盘映射 ----
KEY_TO_ACTION: Dict[str, str] = {
    "w": "up", "a": "left", "s": "down", "d": "right",
    "W": "up", "A": "left", "S": "down", "D": "right",
    "p": "pause", "P": "pause",
    "r": "restart", "R": "restart",
    "q": "quit", "Q": "quit",
}

WIN_ARROWS: Dict[str, str] = {"H": "up", "P": "down", "K": "left", "M": "right"}
POSIX_ARROWS: Dict[str, str] = {"A": "up", "B": "down", "C": "right", "D": "left"}

RUNNING, PAUSED = "running", "paused"


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="snake_game",
        description="Terminal snake: fixed timestep, WASD/arrows, p pause, r restart, q quit.",
    )
    parser.add_argument("--width", type=int, default=config.WIDTH, help="board width in cells")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="board height in cells")
    parser.add_argument("--seed", type=int, default=None, help="random seed (reproducible food)")
    parser.add_argument(
        "--headless",
        type=int,
        nargs="?",
        const=config.DEFAULT_HEADLESS_STEPS,
        default=None,
        metavar="N",
        help="run N steps without a terminal, print result, exit",
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return parser.parse_args(argv)


def make_game(args: argparse.Namespace) -> SnakeGame:
    return SnakeGame(width=args.width, height=args.height, seed=args.seed)


# ----------------------------------------------------------------------
# 速度曲线
# ----------------------------------------------------------------------
def tick_interval(score: int) -> float:
    """按分数计算单步时间（秒），随分数线性加速但不低于 config.MIN_TICK。"""
    return max(config.MIN_TICK, config.BASE_TICK - score * config.TICK_DECREASE)


# ----------------------------------------------------------------------
# 键盘：非阻塞读键
# ----------------------------------------------------------------------
class KeyReader:
    """非阻塞读键。Windows 用 msvcrt，POSIX 用 termios + select。"""

    def __init__(self) -> None:
        self._is_windows = sys.platform.startswith("win")
        self._fd: Optional[int] = None
        self._saved_attrs: object = None
        if not self._is_windows:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def poll(self) -> List[str]:
        """返回自上次调用以来按下的动作名列表（可能为空）。"""
        raw: List[str] = []
        if self._is_windows:
            import msvcrt

            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    raw.append(WIN_ARROWS.get(msvcrt.getwch(), ""))
                else:
                    raw.append(char)
        else:
            import select

            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    break
                char = sys.stdin.read(1)
                if char == "\x1b":
                    seq = ""
                    while len(seq) < 2:
                        more, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if not more:
                            break
                        seq += sys.stdin.read(1)
                    raw.append(POSIX_ARROWS.get(seq, ""))
                else:
                    raw.append(char)
        actions = [KEY_TO_ACTION[key] for key in raw if key in KEY_TO_ACTION]
        return actions

    def close(self) -> None:
        if not self._is_windows and self._fd is not None and self._saved_attrs is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            self._fd = None
            self._saved_attrs = None


# ----------------------------------------------------------------------
# 渲染：整屏重绘
# ----------------------------------------------------------------------
class Renderer:
    """把一个 SnakeGame 渲染成一整屏字符串。

    每行末尾附加 ``\\x1b[K`` 清到行尾，整屏末尾附加 ``\\x1b[J`` 清到屏幕尾部，
    因此短行不会残留上一帧的字符。
    """

    def __init__(self, no_color: bool = False) -> None:
        self._on = not no_color

    def paint(self, code: str) -> str:
        return code if self._on else ""

    def frame(self, game: SnakeGame, mode: str) -> str:
        width, height = game.width, game.height
        cells: List[List[Tuple[str, str]]] = [
            [(config.GLYPH_EMPTY, "") for _ in range(width)] for _ in range(height)
        ]

        for index, (x, y) in enumerate(game.snake):
            if 0 <= x < width and 0 <= y < height:
                if index == 0:
                    cells[y][x] = (config.GLYPH_HEAD, self.paint(config.COLOR_HEAD))
                else:
                    cells[y][x] = (config.GLYPH_BODY, self.paint(config.COLOR_BODY))

        fx, fy = game.food
        if 0 <= fx < width and 0 <= fy < height and cells[fy][fx][0] == config.GLYPH_EMPTY:
            cells[fy][fx] = (config.GLYPH_FOOD, self.paint(config.COLOR_FOOD))

        border = "+" + "-" * width + "+"
        lines: List[str] = []
        lines.append(
            self.paint(config.COLOR_TEXT)
            + "Snake %dx%d  score=%d  len=%d  speed=%.1f/s"
            % (width, height, game.score, len(game.snake), 1.0 / tick_interval(game.score))
            + self.paint(config.COLOR_RESET)
        )
        lines.append(border)
        for row in cells:
            body = "".join(color + glyph + self.paint(config.COLOR_RESET) for glyph, color in row)
            lines.append("|" + body + "|")
        lines.append(border)

        if game.over:
            if game.won:
                lines.append(self.paint(config.COLOR_WARN) + "*** YOU FILLED THE BOARD ***"
                             + self.paint(config.COLOR_RESET))
            else:
                lines.append(self.paint(config.COLOR_WARN) + "*** GAME OVER ***"
                             + self.paint(config.COLOR_RESET))
            lines.append("final score: %d" % game.score)
            lines.append("press r to restart, q to quit")
        elif mode == PAUSED:
            lines.append(self.paint(config.COLOR_WARN) + "-- PAUSED -- press p to resume"
                         + self.paint(config.COLOR_RESET))
        else:
            lines.append("WASD/arrows move   p pause   r restart   q quit")

        return "\n".join(line + CSI_CLEAR_EOL for line in lines) + CSI_CLEAR_BELOW


def enable_ansi() -> None:
    """Windows 10+ 控制台需要显式开启 VT 转义支持。"""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError, ValueError):
        return


# ----------------------------------------------------------------------
# 无头模式：确定性策略驱动，不需要终端
# ----------------------------------------------------------------------
def _headless_choice(game: SnakeGame) -> Optional[str]:
    """返回下一步想走的方向；无路可走时返回 None。

    优先级：离食物更近 > 邻格更空旷，撞墙 / 撞身体 / 掉头的方向直接排除。
    """
    snake = game.snake
    if not snake:
        return None
    hx, hy = snake[0]
    if len(snake) > 1:
        px, py = snake[1]
        cur = (hx - px, hy - py)
    else:
        cur = DIRS["right"]
    food_x, food_y = game.food
    body = set(snake)
    tail = snake[-1]

    scored: List[Tuple[int, int, str]] = []
    for name in sorted(DIRS):
        dx, dy = DIRS[name]
        if (dx, dy) == (-cur[0], -cur[1]):
            continue  # 掉头：game.move 也会忽略，提前跳过
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < game.width and 0 <= ny < game.height):
            continue
        blocked = body if (nx, ny) == (food_x, food_y) else body - {tail}
        if (nx, ny) in blocked:
            continue
        free = 0
        for ddx, ddy in DIRS.values():
            cell = (nx + ddx, ny + ddy)
            if (
                0 <= cell[0] < game.width
                and 0 <= cell[1] < game.height
                and cell not in blocked
            ):
                free += 1
        dist = abs(nx - food_x) + abs(ny - food_y)
        scored.append((dist, -free, name))

    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def run_headless(game: SnakeGame, steps: int) -> int:
    """固定步数无头推进，打印结果并返回退出码。"""
    done = 0
    for _ in range(max(0, steps)):
        if game.over:
            break
        direction = _headless_choice(game)
        if direction is not None:
            game.move(direction)
        game.step()
        done += 1

    print("mode=headless steps_requested=%d steps_done=%d" % (steps, done))
    print(
        "board=%dx%d score=%d length=%d over=%s won=%s"
        % (game.width, game.height, game.score, len(game.snake), game.over, game.won)
    )
    if game.over:
        print("final score: %d" % game.score)
    return 0


# ----------------------------------------------------------------------
# 交互模式：固定时间步 + 非阻塞读键 + 整屏重绘
# ----------------------------------------------------------------------
def run_interactive(game: SnakeGame, no_color: bool) -> int:
    renderer = Renderer(no_color=no_color)
    enable_ansi()
    reader: Optional[KeyReader] = None
    out = sys.stdout
    try:
        reader = KeyReader()
        out.write(CSI_CLEAR_SCREEN + CSI_HIDE_CURSOR)
        mode = RUNNING
        out.write(CSI_HOME + renderer.frame(game, mode))
        out.flush()

        last_tick = time.monotonic()
        dirty = False
        running = True
        while running:
            for action in reader.poll():
                if action == "quit":
                    running = False
                    break
                if action == "restart":
                    game.reset()
                    mode = RUNNING
                    last_tick = time.monotonic()
                    dirty = True
                elif action == "pause":
                    if not game.over:
                        mode = PAUSED if mode == RUNNING else RUNNING
                        last_tick = time.monotonic()
                        dirty = True
                elif action in DIRS:
                    if mode == RUNNING and not game.over:
                        game.move(action)
            if not running:
                break

            now = time.monotonic()
            if mode == RUNNING and not game.over and now - last_tick >= tick_interval(game.score):
                game.step()
                last_tick = now
                dirty = True

            if dirty:
                out.write(CSI_HOME + renderer.frame(game, mode))
                out.flush()
                dirty = False

            time.sleep(0.005)
    except KeyboardInterrupt:
        return 0
    finally:
        if reader is not None:
            reader.close()
        out.write(CSI_SHOW_CURSOR + config.COLOR_RESET + "\n")
        out.flush()
    return 0


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _isatty(stream: object) -> bool:
    """安全地判断流是否连到终端（流可能为 None 或已被替换）。"""
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return False


def is_interactive_terminal() -> bool:
    """只有 stdin 和 stdout 都是终端时才进入交互模式。"""
    return _isatty(sys.stdin) and _isatty(sys.stdout)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        game = make_game(args)
    except ValueError as exc:
        print("invalid board size: %s" % exc, file=sys.stderr)
        return 2

    headless_steps = args.headless
    if headless_steps is None and not is_interactive_terminal():
        # stdin 或 stdout 不是终端（管道 / 重定向 / 自检工具）：
        # 自动切无头模式，否则没有按键来源时会一直空转，无法自检。
        headless_steps = config.DEFAULT_HEADLESS_STEPS
    if headless_steps is not None:
        if headless_steps <= 0:
            headless_steps = config.DEFAULT_HEADLESS_STEPS
        return run_headless(game, headless_steps)
    return run_interactive(game, no_color=args.no_color)


if __name__ == "__main__":
    sys.exit(main())
