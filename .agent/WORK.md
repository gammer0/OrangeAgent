# 工作记忆

## 最终结构

- `config.py`：常量（尺寸/速度/渲染字符/ANSI 颜色/按键映射），无逻辑无副作用。
- `game.py`：纯逻辑层 `SnakeGame`，状态+规则+推进；不导入终端/键盘/时间库、不打印，可被无头工具直接调用。契约（勿改名）：`__init__(w,h,seed)` / `move(direction)` / `step()` / `score` / `snake` / `food` / `over`
- `main.py`：入口，参数解析 + 非阻塞读键 + 固定时间步主循环（累加器 `acc`）+ 整屏重绘。
- `tests/test_game.py`（+ `tests/__init__.py`）：`unittest` 单测。
- `README.md`：文档；方向处理见「方向缓冲」节。

## 验收命令

- `Set-Location C:\Users\USER\Desktop\agent; python -m snake_game.main --headless 200 --seed 7`
- `Set-Location C:\Users\USER\Desktop\agent; python -m unittest discover -s snake_game/tests -t . -v`

## 设计决定

`move()` 做方向队列（容量 2，判重/禁反向以队尾为基准，`step()` 每帧 `pop(0)`）而非单槽覆盖：一帧内连按的第二方向不被丢弃，「上+左」的连续折返能按序生效。
