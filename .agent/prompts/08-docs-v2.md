你好。这个项目是 snake_game，你之前参与过其中一部分工作。现在需要你把两处文档补齐——**代码不要动**，只改文档。

## 背景（请先自己核实，不要直接信我）

1. 用 `/tool/os/file/read` 读 `snake_game/game.py`，确认 `move()` 当前是「单槽覆盖」还是「方向队列」。
2. 用 `/tool/os/file/read` 读 `snake_game/README.md` 第 40–55 行，看「按键」小节对同一行为的描述是否与代码一致。
3. 读 `.agent/WORK.md`（当前只有一个占位标题，因为之前的内容被我清掉了）。

## 请完成的三件事

**A. 补 README**
如果 README 对「一帧内多次转向」的描述与 game.py 实际实现不符，就在「按键」小节之后插入一节 `## 方向缓冲` 说明真实行为（容量、按序生效、丢弃规则）。如果本来就一致，就明确告诉我「无需修改」，不要硬加一节。

**B. 重写 .agent/WORK.md**
用 `/tool/os/file/write` 整体重写，20 行以内，三节：
- `## 最终结构`：snake_game/ 下各文件职责
- `## 验收命令`：无头运行命令 + unittest 命令
- `## 设计决定`：game.py 的方向处理为什么要做成现在这样（一句话）

**C. 验证**
改完跑这两条，并把输出贴给我：
- `Set-Location C:\Users\USER\Desktop\agent; python -m snake_game.main --headless 200 --seed 7`
- `Set-Location C:\Users\USER\Desktop\agent; python -m unittest discover -s snake_game/tests -t . -v`

## 约束

- 不改任何 .py 文件。
- 每次只写一个文件。
- 如果发现 README 与代码不一致，**以代码为准**，不要为了迁就 README 去改代码。
- 汇报时说明：你核实到的实际实现是什么、README 是否真的需要改、改了哪一节。
