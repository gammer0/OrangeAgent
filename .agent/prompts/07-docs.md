临时脚本已删干净，谢谢。剩下文档还没更新，这次只做文档，两个文件各一次写入：

A. 用 `/tool/os/file/write` 整体重写 `.agent/WORK.md`（覆盖），内容严格如下，不要增删小节：

```markdown
# 工作记忆

## 最终结构
- `snake_game/config.py`：全部常量（棋盘、速度、字符、配色、按键），无逻辑
- `snake_game/game.py`：纯逻辑层 `SnakeGame`，不碰终端/键盘/时间
- `snake_game/main.py`：入口，参数解析 + 固定时间步主循环 + 整屏重绘
- `snake_game/tests/test_game.py`：9 个 unittest 回归测试
- `snake_game/README.md`：运行方式、按键、计分规则

## 验收命令
- `python -m snake_game.main --headless 200 --seed 7`
- `python -m unittest discover -s snake_game/tests -t . -v`

## 设计决定
- 单槽 `_pending` 会让一帧内的连续两次转向丢掉一次，手感发黏；改为容量 2 的方向队列后按序生效。
```

B. 在 `snake_game/README.md` 的「按键」小节之后、「计分与加速规则」之前，插入这段（用 `/tool/os/file/read` 读出全文，找到锚点后用 `/tool/os/file/write` 整体写回）：

```markdown
## 方向缓冲

- 一帧内最多缓存 2 次转向，按顺序在下一次 `step()` 起逐帧生效，因此支持快速连按「上」「左」这类连续折返。
- 队列满时的第 3 次转向被丢弃；与队列末尾方向相同或相反的输入也会被丢弃。

```

做完后只回答一句话：两个文件各插入/重写成功，以及 unittest 是否仍为 9 项通过。
