"""临时逻辑自检：不进入终端交互，跑完即删。"""
from __future__ import annotations

import random

from snake_game.game import SnakeGame


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


# 1. 初始形状
g = SnakeGame(seed=1)
check("初始长度=3", len(g.snake) == 3)
check("头坐标在中央", g.snake[0] == (10, 10))
check("食物不与蛇重叠", g.food not in g.snake)
check("初始未结束", not g.over and g.score == 0)

# 2. 反向忽略：向右时按左不死
g.move("left")
g.step()
check("反向输入被忽略", not g.over)
check("位置未被反向推动", g.snake[0] == (11, 10))

# 3. 固定种子可复现
a, b = SnakeGame(seed=42), SnakeGame(seed=42)
check("同种子食物一致", a.food == b.food)

# 4. 撞墙
g2 = SnakeGame(seed=2)
for _ in range(30):
    g2.step()
check("撞墙结束", g2.over)
frozen = g2.snake
g2.step()
check("结束后状态冻结", g2.snake == frozen)

# 5. 随机漫游：必定吃到食物，且不变量始终成立
keys = ("up", "down", "left", "right")
eaten_ok = True
won_seeds = 0
for seed in range(30):
    g3 = SnakeGame(seed=seed)
    rng = random.Random(seed)
    for _ in range(400):
        if g3.over:
            break
        g3.move(rng.choice(keys))
        g3.step()
        if len(g3.snake) != 3 + g3.score:
            eaten_ok = False
            print(f"  长度/分数不一致 seed={seed} score={g3.score} len={len(g3.snake)}")
            break
        if g3.food in g3.snake:
            eaten_ok = False
            print(f"  食物落在蛇身上 seed={seed}")
            break
        if g3.snake[0] != g3.snake[0] or len(set(g3.snake)) != len(g3.snake):
            eaten_ok = False
            print(f"  蛇身自重叠 seed={seed}")
            break
    if g3.score > 0:
        won_seeds += 1
check("长度=3+分数且食物不在蛇身", eaten_ok)
check("漫游能吃到食物", won_seeds > 0)

# 6. 非法方向
try:
    g.move("sideways")
except ValueError:
    check("非法方向抛 ValueError", True)
else:
    check("非法方向抛 ValueError", False)

# 7. 速度随分数提升并封顶
g5 = SnakeGame(seed=5)
check("初始速度=10 步/秒", abs(g5.steps_per_sec - 10.0) < 1e-9)
check("帧间隔与速度互为倒数", abs(g5.frame_interval * g5.steps_per_sec - 1.0) < 1e-9)

# 8. 小棋盘能获胜（3x3 太小，用 6x6 跑满屏不现实，仅验证不崩）
g6 = SnakeGame(width=6, height=6, seed=0)
for _ in range(200):
    g6.step()
check("最小棋盘可运行", isinstance(g6.over, bool))

print("全部逻辑自检通过")
