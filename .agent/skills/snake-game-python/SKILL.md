---
name: snake-game-python
description: 用 Python 从零实现可在终端运行的贪吃蛇游戏：固定时间步、无头自检、分层结构。用于编写或验收 snake_game 项目。
---

# Python 贪吃蛇游戏工程规范

## 目标

产出一个**能在终端直接运行**的贪吃蛇游戏，并且**逻辑与渲染分离**，
使得不依赖键盘也能被自检工具验证。

## 目录结构（必须）

```
snake_game/
├── main.py          # 唯一入口：参数解析 + 主循环 + 渲染
├── game.py          # 纯逻辑：状态、规则、推进（无 input / 无 print 循环）
├── config.py        # 常量：棋盘尺寸、初始速度、加速曲线、颜色
└── README.md        # 运行方式、按键、规则
```

允许拆分更多模块，但 `main.py` 必须是入口，`game.py` 必须存在且**不导入**
任何终端/键盘库。

## 必须提供的中立 API（供验收工具调用）

```python
# game.py
class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None): ...
    def move(self, direction: str) -> None: ...
        # direction ∈ {"up", "down", "left", "right"}
    def step(self) -> None: ...
        # 前进一步：移动蛇身、判定吃食物/撞墙/撞自己
    @property
    def score(self) -> int: ...
    @property
    def snake(self) -> list[tuple[int, int]]: ...   # 头在前
    @property
    def food(self) -> tuple[int, int]: ...
    @property
    def over(self) -> bool: ...
```

这些名字是**验收契约**，不要改名。内部实现自由。

## 规则要求

1. 反向输入忽略：蛇向右时按左不产生立即死亡。
2. 吃到食物：长度 +1，分数 +1（或按食物类型加权），在空格随机生成新食物。
3. 撞墙 / 撞自身：置 `over=True`，之后 `step()` 不再改变状态。
4. 满屏：所有格子被占据时视为胜利，`over=True`。
5. 随机性必须可通过 `seed` 复现。

## 主循环要求

- `main.py` 使用 `curses`（Windows 用 `windows-curses`）**或**纯 ANSI 转义 + `msvcrt`。
- 固定时间步：默认 8–15 步/秒，随分数提高逐步加速，设下限。
- 必须支持 `q` 退出、`p` 暂停、方向键或 WASD。
- 结束显示分数与「按 r 重开」。
- 单帧渲染必须**整屏重绘**（不逐字符刷屏），避免闪烁。

## 禁止事项

- 不要在 `game.py` 里写 `input()`、`time.sleep()`、`curses`。
- 不要用 `global` 持有游戏状态；状态放在类实例上。
- 不要引入除标准库（及 `windows-curses`）以外的依赖。
- 不要留下 `TODO`、未使用的导入、`pass  # 占位`。

## 自检流程（提交前必须全绿）

1. `harness` 侧工具 `/tool/code/py/check` 逐个文件检查语法。
2. `/tool/game/snake/demo` 固定种子跑 200 步，要求 `ok: true`、无异常。
3. `/tool/game/snake/check` 要求 `结论: 通过`。
4. 手动冒烟：`python -m snake_game.main`（或 `python snake_game/main.py`）能进入界面。

## 交付说明

完成后在 `README.md` 写明：

- 一行启动命令（含 `windows-curses` 安装提示，若使用）
- 按键表
- 计分与加速规则
- 已知限制
