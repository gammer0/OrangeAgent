"""贪吃蛇纯逻辑层：状态、规则、推进。

本模块不导入任何终端 / 键盘 / 时间库，可被无头自检工具直接驱动。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

try:  # 作为包导入（python -m snake_game.main）
    from . import config
except ImportError:  # 直接运行 snake_game/main.py 时的兜底
    import config  # type: ignore[no-redef]

Cell = Tuple[int, int]

#: 方向 -> (dx, dy)，y 轴向下增长
DIRECTIONS: Dict[str, Cell] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

#: 每个方向的反向
OPPOSITE: Dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


class SnakeGame:
    """一局贪吃蛇的全部状态与规则。

    坐标约定：``(x, y)``，``x`` 向右增长、``y`` 向下增长，
    左上角为 ``(0, 0)``，有效范围 ``0 <= x < width``、``0 <= y < height``。
    """

    def __init__(self, width: int = 20, height: int = 20,
                 seed: Optional[int] = None) -> None:
        if width < 5 or height < 5:
            raise ValueError("棋盘至少需要 5x5 格")
        self._width: int = int(width)
        self._height: int = int(height)
        self._seed: Optional[int] = None if seed is None else int(seed)
        self._rng: random.Random = random.Random(self._seed)
        self._snake: List[Cell] = []
        self._direction: str = "right"
        self._pending: List[str] = []
        self._score: int = 0
        self._over: bool = False
        self._won: bool = False
        self._food: Cell = (-1, -1)
        self._reset_state()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        """把状态恢复成开局：同 seed 则食物序列完全一致。"""
        self._rng = random.Random(self._seed)
        cx, cy = self._width // 2, self._height // 2
        length = min(config.START_LENGTH, self._width - 2)
        body = [(cx - i, cy) for i in range(length)]
        left = min(x for x, _ in body)
        if left < 1:
            shift = 1 - left
            body = [(x + shift, y) for x, y in body]
        self._snake = body          # 头在前
        self._direction = "right"
        self._pending = []
        self._score = 0
        self._over = False
        self._won = False
        self._food = (-1, -1)
        self._spawn_food()

    def reset(self) -> None:
        """重开一局（保留构造时的 seed，结果可复现）。"""
        self._reset_state()

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------
    def move(self, direction: str) -> None:
        """登记一次转向请求。

        - 反向输入（相对队列中最后一次有效方向）静默忽略，不会原地掉头；
        - 与当前有效方向相同的重复输入直接忽略，不占用队列；
        - 一帧内最多缓存 ``config.MAX_PENDING_MOVES`` 次转向，按序生效；
        - 非法方向字符串静默忽略；游戏结束后不再接受输入。
        """
        if self._over or direction not in DIRECTIONS:
            return
        last = self._pending[-1] if self._pending else self._direction
        if direction == last or direction == OPPOSITE[last]:
            return
        if len(self._pending) >= config.MAX_PENDING_MOVES:
            return
        self._pending.append(direction)

    # ------------------------------------------------------------------
    # 推进
    # ------------------------------------------------------------------
    def step(self) -> None:
        """前进一步；游戏结束后调用不改变任何状态。"""
        if self._over:
            return

        if self._pending:
            self._direction = self._pending.pop(0)

        dx, dy = DIRECTIONS[self._direction]
        hx, hy = self._snake[0]
        head: Cell = (hx + dx, hy + dy)

        if not self._inside(head):
            self._over = True
            return

        eating = head == self._food
        # 不吃食物时尾巴会移开，所以尾部格子不算障碍
        obstacle = self._snake if eating else self._snake[:-1]
        if head in obstacle:
            self._over = True
            return

        self._snake.insert(0, head)
        if eating:
            self._score += 1
            self._spawn_food()
        else:
            self._snake.pop()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _inside(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self._width and 0 <= y < self._height

    def _spawn_food(self) -> None:
        occupied = set(self._snake)
        free = [
            (x, y)
            for y in range(self._height)
            for x in range(self._width)
            if (x, y) not in occupied
        ]
        if not free:
            # 满屏：胜利，状态冻结
            self._won = True
            self._over = True
            return
        self._food = self._rng.choice(free)

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def score(self) -> int:
        return self._score

    @property
    def snake(self) -> List[Cell]:
        """蛇身坐标，头在前。返回副本，外部无法改内部状态。"""
        return list(self._snake)

    @property
    def food(self) -> Cell:
        return self._food

    @property
    def over(self) -> bool:
        return self._over

    @property
    def won(self) -> bool:
        return self._won

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

