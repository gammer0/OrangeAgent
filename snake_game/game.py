"""贪吃蛇纯逻辑层：状态、规则、推进。

本模块不导入任何终端/键盘/时间库，也不打印任何东西，
以便被无头自检工具直接调用。坐标约定：x 向右、y 向下，(0, 0) 在左上角。
"""

from __future__ import annotations

import random

from . import config

DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# 待执行方向队列容量。2 足够做「一帧内连按两次」的连续折返，
# 又不至于让手速快的玩家把方向堆到失控。
DIRECTION_QUEUE_SIZE = 2


class SnakeGame:
    """一局贪吃蛇的全部状态与规则。

    中立 API（验收契约，勿改名）：
        __init__(width, height, seed) / move(direction) / step()
        score / snake / food / over
    """

    def __init__(
        self,
        width: int = config.WIDTH,
        height: int = config.HEIGHT,
        seed: int | None = None,
    ) -> None:
        if width < config.MIN_SIZE or height < config.MIN_SIZE:
            raise ValueError(
                f"棋盘至少 {config.MIN_SIZE}x{config.MIN_SIZE}，收到 {width}x{height}"
            )
        self.width = int(width)
        self.height = int(height)
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """重置为新一局（seed 固定时可完全复现）。"""
        self._rng = random.Random(self.seed)
        mid_x = self.width // 2
        mid_y = self.height // 2
        self._snake: list[tuple[int, int]] = [
            (mid_x - i, mid_y) for i in range(config.INITIAL_LENGTH)
        ]
        self._direction = "right"
        self._direction_queue: list[str] = []  # 队首是下一帧要转的方向
        self._score = 0
        self._over = False
        self._won = False
        self._food: tuple[int, int] = (mid_x + 1, mid_y)
        self._spawn_food()

    @property
    def score(self) -> int:
        return self._score

    @property
    def snake(self) -> list[tuple[int, int]]:
        """蛇身坐标，头在前。返回副本，外部改动不影响内部状态。"""
        return list(self._snake)

    @property
    def food(self) -> tuple[int, int]:
        return self._food

    @property
    def over(self) -> bool:
        return self._over

    @property
    def won(self) -> bool:
        """是否填满棋盘获胜。"""
        return self._won

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def steps_per_sec(self) -> float:
        """按当前分数给出的固定时间步速度（含上限）。"""
        return min(config.INITIAL_SPS + config.SPS_PER_FOOD * self._score, config.MAX_SPS)

    @property
    def frame_interval(self) -> float:
        """一帧（一步）应持续多少秒。"""
        return 1.0 / self.steps_per_sec

    # ------------------------------------------------------------------ 规则
    def move(self, direction: str) -> None:
        """把方向放入待执行队列。

        非法值抛 ValueError。以下情况静默忽略（不报错、不入队）：
        与「队列最后一个方向（空则当前方向）」相同、与它相反、
        或队列已满（DIRECTION_QUEUE_SIZE）。
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"未知方向: {direction!r}")
        if self._over:
            return
        # 以队列尾为基准，而不是当前方向：这样「上、左」连按两次都能保留
        last = self._direction_queue[-1] if self._direction_queue else self._direction
        if direction == last:
            return
        # 只有长度 > 1 时「禁止反向」才有意义（单节蛇没有方向可反）
        if len(self._snake) > 1 and direction == OPPOSITE[last]:
            return
        if len(self._direction_queue) >= DIRECTION_QUEUE_SIZE:
            return
        self._direction_queue.append(direction)

    def step(self) -> None:
        """前进一步：位移、判定吃食物 / 撞墙 / 撞自身。结束后不再改变状态。"""
        if self._over:
            return

        # 每步只消费队首一个方向；队列空则维持当前方向
        if self._direction_queue:
            self._direction = self._direction_queue.pop(0)
        dx, dy = DIRECTIONS[self._direction]
        head_x, head_y = self._snake[0]
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self._over = True
            return

        eating = new_head == self._food
        # 不吃食物时尾巴会挪走，因此尾巴那一格不算碰撞
        blockers = self._snake if eating else self._snake[:-1]
        if new_head in blockers:
            self._over = True
            return

        self._snake.insert(0, new_head)
        if eating:
            self._score += 1
            self._spawn_food()
        else:
            self._snake.pop()

    # ------------------------------------------------------------------ 内部
    def _spawn_food(self) -> tuple[int, int]:
        """在空格中随机放一个食物；无空格则获胜并结束。"""
        occupied = set(self._snake)
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free:
            self._won = True
            self._over = True
            return self._food
        self._food = self._rng.choice(free)
        return self._food

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"SnakeGame({self.width}x{self.height}, score={self._score}, "
            f"len={len(self._snake)}, over={self._over})"
        )
