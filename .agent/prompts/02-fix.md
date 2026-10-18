我看到你之前已经在 snake_game/ 下写了 config.py、game.py、main.py，结构是对的。现在继续，但请严格按我的顺序来。

第一步，先回答我一个问题，并给出证据：
你用 /tool/game/snake/demo 跑一次（path=snake_game, steps=200, seed=7）。
这个工具会以 snake_game 包的方式导入你的 game 模块。如果它报"未找到可实例化的游戏类"，请你自己判断根因是什么，然后只改最小的那一处把它修好。

第二步，修好后重新跑一次 demo，要求输出里 ok 为 true、frames 等于 200、errors 为空。

第三步，跑 /tool/game/snake/check，并补写 README.md（启动命令、按键、计分与加速规则、已知限制）。

注意：
- 不要重写 game.py 的核心逻辑，除非 demo 的报错确实指向它。
- 每次只改一个文件，改完立刻用 /tool/code/py/check 验证语法。
- 汇报时请贴出 demo 的完整 JSON 输出，不要只说"通过了"。
