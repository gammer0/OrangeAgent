"""独立验收：不复用项目内部测试，只按 SKILL.md 的契约验证。"""

from __future__ import annotations

import subprocess
import sys


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    results: list[bool] = []
    try:
        from snake_game.game import SnakeGame
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 导入 snake_game.game: {exc}")
        return 1

    # 1. 反向输入
    game = SnakeGame(seed=1)
    game.move("left")
    head = game.snake[0]
    game.step()
    results.append(check("反向输入被忽略", game.snake[0] != (head[0] - 1, head[1])))

    # 2. 非法方向
    try:
        SnakeGame(seed=1).move("north")
        results.append(check("非法方向抛错", False))
    except ValueError:
        results.append(check("非法方向抛错", True))

    # 3. 撞墙
    game = SnakeGame(width=12, height=12, seed=3)
    for _ in range(20):
        game.step()
    after = list(game.snake)
    game.step()
    results.append(check("撞墙后 over", game.over))
    results.append(check("over 后状态冻结", game.snake == after))

    # 4. 吃食物增长
    game = SnakeGame(width=20, height=20, seed=7)
    start_len = len(game.snake)
    game.move("up")
    game.step()
    eaten = 0
    for _ in range(300):
        if game.over:
            break
        head = game.snake[0]
        food = game.food
        if food[0] != head[0]:
            game.move("right" if food[0] > head[0] else "left")
        elif food[1] != head[1]:
            game.move("down" if food[1] > head[1] else "up")
        before_score = game.score
        game.step()
        if game.score > before_score:
            eaten += 1
    results.append(check("吃食物加分", game.score == eaten, f"score={game.score}"))
    results.append(check("吃食物变长", len(game.snake) == start_len + game.score))

    # 5. 种子复现
    seq_a = [SnakeGame(seed=99).food]
    seq_b = [SnakeGame(seed=99).food]
    results.append(check("同种子食物一致", seq_a == seq_b))

    # 6. 方向队列
    game = SnakeGame(seed=7)
    origin = game.snake[0]
    game.move("up")
    game.move("left")
    game.step()
    first = game.snake[0]
    game.step()
    second = game.snake[0]
    results.append(
        check(
            "两次转向按序生效",
            first[1] < origin[1] and second[0] < first[0],
            f"{origin} -> {first} -> {second}",
        )
    )

    # 7. 包导入与入口
    proc = subprocess.run(
        [sys.executable, "-B", "snake_game/main.py", "--headless", "60", "--seed", "5"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    results.append(check("headless 启动成功", proc.returncode == 0, proc.stdout.strip()[:80]))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 项通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
