我在真实终端里试玩了你写的贪吃蛇，发现一个渲染 bug，请你定位并修好。**代码由你自己改**，我只给复现方法和判断依据。

## 现象

状态行会残留上一帧的字符。实测：

```
第一帧: 分数 0    蛇长 3    速度 10.0 步/秒    方向 right
第二帧: 分数 8    蛇长 11    速度 14.8 步/秒    方向 upght
                                                    ^^^^ 残留
```

`right`（5 字符）变成 `up`（2 字符）时，多出来的 `ght` 没有被清掉。

## 复现方法（可以直接跑）

```powershell
Set-Location C:\Users\USER\Desktop\agent
python -c "import sys,pyte;sys.path.insert(0,'.');from snake_game.game import SnakeGame;from snake_game.main import build_frame;s=pyte.Screen(80,12);st=pyte.Stream(s);NC=type('N',(),{'__call__':lambda self,c:''})();st.feed(('\x1b[H'+'\n'.join(['aaaa','方向 right'])+'\n\x1b[J').replace('\n','\r\n'));st.feed(('\x1b[H'+'\n'.join(['aaaa','方向 up'])+'\n\x1b[J').replace('\n','\r\n'));print(''.join(s.buffer[1][x].data for x in range(20)).rstrip())"
```

预期看到 `方向 upght`，确认后即可开始修。

## 请你做的

1. **先读 `main.py` 里的 `draw()`**，说清 `\x1b[J` 到底清的是「从光标到屏幕末尾」还是「整行剩余部分」。这是本 bug 的根因所在。
2. 判断根因：是 `\x1b[J` 语义不覆盖短行尾部，还是每行输出前没有清除行内旧内容。
3. 选一个**最小改动**方案并实现，例如：每行先输出 `\x1b[K`（清除到行尾），或在写屏前整屏 `\x1b[2J`。不要重写渲染架构。
4. 修完后自测：
   - 让状态行从 `right` 变 `up`，确认没有 `ght` 残留；
   - 让分数位数变化（如 `分数 9` -> `分数 10`）时也不错位；
   - 跑 `python -m unittest discover -s snake_game/tests -t . -v` 确认 9 项仍通过；
   - 跑 `python -m snake_game.main --headless 200 --seed 7` 确认无头模式不受影响。
5. 把「修复前 / 修复后」的实际输出贴给我。

## 约束

- 不要动 `game.py`（逻辑层与渲染无关）。
- 不要引入第三方依赖。
- 改完请说明你改的是哪几行、为什么这样改。
