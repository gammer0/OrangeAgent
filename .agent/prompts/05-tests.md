上一轮你被杀在步数上限，收尾没做完。这次只做**第 1 步**，做完就停，不要继续别的。

目标：把 `.agent/` 下的临时验证脚本整理成正式测试。

具体要求：
1. 先读 `.agent/invariant_check.py` 和 `.agent/repro_direction_queue.py`，把其中**独立于临时环境**的断言抽取出来。
2. 新建 `snake_game/tests/__init__.py`（空文件）与 `snake_game/tests/test_game.py`，用标准库 `unittest`，覆盖：
   - 反向输入被忽略
   - 非法方向抛 ValueError
   - 撞墙后 over=True 且后续 step 状态冻结
   - 吃食物加分且变长
   - 同 seed 食物可复现
   - 连续 move('up') / move('left') 后两次 step 依次生效
3. 用 `/tool/game/snake/check` 之外的方式跑一次：
   `Set-Location C:\Users\USER\Desktop\agent; python -m unittest discover -s snake_game/tests -t . -v`
4. 把测试的完整输出贴给我。

不要删任何 .agent 下的文件（下一步再删），不要改 game.py。
