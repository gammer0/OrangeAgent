你上一轮只做了复现，`draw()` 还是原样：

```python
def draw(lines: list[str]) -> None:
    sys.stdout.write("\x1b[H" + "\n".join(lines) + "\n\x1b[J")
```

现在**直接修**，做完就停。

## 根因（你已复现，这里确认一下）

`ESC[J` 是「从光标位置清到屏幕末尾」。`ESC[H` 把光标移到 (1,1) 后，逐行写 `lines` 的内容——
但每行**只是覆盖写**，如果新行比旧行短，旧行尾部字符会留在原地。`ESC[J` 在最后才执行，
它清的是「最后一帧内容之后的区域」，管不到中间某一行的尾部。

## 请实现

在 `draw()` 里，让每一行在写完自身内容后立刻清除本行剩余部分。最小改动，例如：

```python
def draw(lines: list[str]) -> None:
    """光标归位后逐行重绘，每行清除到行尾，避免短行残留上一帧字符。"""
    sys.stdout.write("\x1b[H" + "\n".join(line + "\x1b[K" for line in lines) + "\n\x1b[J")
    sys.stdout.flush()
```

`ESC[K` = 清除光标到本行行尾，正好解决短行残留。

## 修完必须验证（全部贴输出）

1. 你之前写的 `.tmp_render_check.py` 里「场景 1」和「场景 5」必须从 BAD 变 OK。
2. `Set-Location C:\Users\USER\Desktop\agent; python -m unittest discover -s snake_game/tests -t . -v` → 9 项全过。
3. `Set-Location C:\Users\USER\Desktop\agent; python -m snake_game.main --headless 200 --seed 7` → 退出码 0。
4. 单独确认：状态行 `方向 right` → `方向 up` 之后是 `方向 up` 而不是 `方向 upght`。

## 约束

- 只改 `draw()` 这一处，不要动 `game.py`、`build_frame()`、主循环。
- 不要引入依赖（pyte 只能用于临时验证脚本，不能出现在 `snake_game/` 里）。
- 验证完把 `.tmp_render_check.py` 删除，不要留在项目根目录。
