"""game.SnakeGame 的单元测试（标准库 unittest，无第三方依赖）。

断言来源：.agent/repro_direction_queue.py 与 .agent/invariant_check.py 中
不依赖临时路径、不碰私有属性的那部分校验，整理成可长期保留的回归测试。

运行：
    python -m unittest discover -s snake_game/tests -t . -v
"""

from __future__ import annotations

import unittest

from snake_game import config
from snake_game.game import DIRECTIONS, OPPOSITE, SnakeGame

ALL_DIRECTIONS = ("up", "down", "left", "right")


def greedy_direction(game: SnakeGame) -> str | None:
    """朝食物走一步的朴素贪心；取不到合法方向时返回 None。

    只用于让测试能稳定吃到食物，不代表任何游戏策略。
    """
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    candidates: list[str] = []
    if food_x > head_x:
        candidates.append("right")
    elif food_x < head_x:
        candidates.append("left")
    if food_y > head_y:
        candidates.append("down")
    elif food_y < head_y:
        candidates.append("up")

    for direction in candidates:
        if direction == OPPOSITE[game.direction]:
            continue
        dx, dy = DIRECTIONS[direction]
        if 0 <= head_x + dx < game.width and 0 <= head_y + dy < game.height:
            return direction
    return None


class TestDirectionRules(unittest.TestCase):
    def test_reverse_input_ignored(self) -> None:
        """向右时按「左」必须被忽略，不得原地掉头。"""
        game = SnakeGame(seed=7)
        self.assertEqual(game.direction, "right")

        game.move("left")
        self.assertEqual(game.direction, "right")
        game.step()
        self.assertEqual(game.direction, "right")
        self.assertEqual(game.snake[0], (11, 10))  # (10,10) 向右一格
        self.assertFalse(game.over)

    def test_invalid_direction_raises(self) -> None:
        game = SnakeGame(seed=7)
        with self.assertRaises(ValueError):
            game.move("nope")
        with self.assertRaises(ValueError):
            game.move("")
        # 抛错后状态不受污染
        self.assertEqual(game.direction, "right")
        game.step()
        self.assertFalse(game.over)


class TestWallCollision(unittest.TestCase):
    def test_wall_sets_over_and_freezes_state(self) -> None:
        """一路向右撞墙 -> over=True，且此后 step() 完全冻结状态。"""
        game = SnakeGame(seed=7)
        start_x = game.snake[0][0]
        steps_to_wall = game.width - start_x
        for _ in range(steps_to_wall):
            game.step()

        self.assertTrue(game.over)
        frozen = (game.snake, game.score, game.food, game.direction)

        for _ in range(5):
            game.step()
            game.move("up")
        self.assertEqual((game.snake, game.score, game.food, game.direction), frozen)


class TestEating(unittest.TestCase):
    def test_eating_scores_and_grows(self) -> None:
        game = SnakeGame(seed=7)
        start_len = len(game.snake)
        self.assertEqual(start_len, config.INITIAL_LENGTH)

        steps = 0
        while game.score < 1 and not game.over and steps < 500:
            direction = greedy_direction(game)
            if direction:
                game.move(direction)
            game.step()
            steps += 1

        self.assertFalse(game.over, "贪心路径不应死亡")
        self.assertGreaterEqual(game.score, 1)
        self.assertEqual(len(game.snake), start_len + game.score)
        self.assertNotIn(game.food, set(game.snake))


class TestDeterminism(unittest.TestCase):
    def test_same_seed_reproducible(self) -> None:
        first = SnakeGame(seed=7)
        second = SnakeGame(seed=7)
        self.assertEqual(first.food, second.food)

        for _ in range(30):
            first.step()
            second.step()
            self.assertEqual(first.snake, second.snake)
            self.assertEqual(first.food, second.food)
            self.assertEqual(first.over, second.over)

        other = SnakeGame(seed=8)
        self.assertNotEqual(SnakeGame(seed=7).food, other.food)


class TestDirectionQueue(unittest.TestCase):
    def test_two_turns_apply_in_order(self) -> None:
        """向右时连按「上」「左」：先上再左，两次输入都生效。"""
        game = SnakeGame(seed=7)
        self.assertEqual(game.snake[0], (10, 10))

        game.move("up")
        game.move("left")
        game.step()
        self.assertEqual((game.snake[0], game.direction), ((10, 9), "up"))
        game.step()
        self.assertEqual((game.snake[0], game.direction), ((9, 9), "left"))

    def test_queue_capacity_is_two(self) -> None:
        """队列上限 2，第三个转向被丢弃。"""
        game = SnakeGame(seed=7)
        game.move("up")
        game.move("left")
        game.move("down")
        directions = []
        for _ in range(3):
            game.step()
            directions.append(game.direction)
        self.assertEqual(directions, ["up", "left", "left"])

    def test_repeated_direction_ignored(self) -> None:
        """重复同向输入不得占满队列。"""
        game = SnakeGame(seed=7)
        for _ in range(3):
            game.move("up")
        for _ in range(2):
            game.step()
        self.assertEqual((game.snake[0], game.direction), ((10, 8), "up"))

    def test_reset_clears_queue(self) -> None:
        game = SnakeGame(seed=7)
        game.move("up")
        game.move("left")
        game.reset()
        directions = []
        for _ in range(2):
            game.step()
            directions.append(game.direction)
        self.assertEqual(directions, ["right", "right"])


if __name__ == "__main__":
    unittest.main()
