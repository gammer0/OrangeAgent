收尾。我刚才独立跑了 9 项验收，全部通过：

```
[PASS] 反向输入被忽略        [PASS] 撞墙后 over
[PASS] 非法方向抛错          [PASS] over 后状态冻结
[PASS] 吃食物加分 score=6    [PASS] 吃食物变长
[PASS] 同种子食物一致        [PASS] 两次转向按序生效 (10,10) -> (10,9) -> (9,9)
[PASS] headless 启动成功
```

三件事，做完就结束：

1. README 里补一段「方向缓冲」说明：一帧内最多缓存 2 次转向，按序生效；反向输入仍被丢弃。放在「按键」小节后面。

2. `.agent/` 下有几份临时脚本（patch_game.py、repro_direction_queue.py、invariant_check.py、game_before.py）。它们不属于交付物：
   - 如果其中有值得长期保留的验证逻辑，请把它整理成 snake_game/tests/test_game.py（用 unittest，标准库，不引第三方），并确保 `python -m unittest discover -s snake_game/tests` 能跑通；
   - 整理完删掉 .agent/ 下那 4 个临时脚本。

3. 更新 .agent/WORK.md：写清最终结构、验收命令、以及「方向队列」这个设计决定的原因。不要写成流水账，控制在 20 行内。

注意：
- 不要改动 game.py 里已经通过验收的逻辑。
- 新增测试文件后要跑一次，把完整输出贴给我。
