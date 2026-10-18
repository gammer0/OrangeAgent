# 贪吃蛇（终端版）

纯 Python 标准库实现的终端贪吃蛇：固定时间步推进、非阻塞读键、整屏重绘。
不依赖任何第三方包，Windows / macOS / Linux 均可运行。

```
snake_game/
├── config.py   # 常量：棋盘、速度曲线、字符与颜色
├── game.py     # 纯逻辑：SnakeGame（不碰终端）
├── main.py     # 入口：参数、主循环、渲染、无头自检
└── README.md
```

## 运行

在仓库根目录（`snake_game/` 的上一级）执行：

```bash
python snake_game/main.py
```

也可以作为模块运行：

```bash
python -m snake_game.main
```

需要 Python 3.8+。

## 按键

| 按键 | 作用 |
| --- | --- |
| `W` `A` `S` `D` / 方向键 | 上、左、下、右 |
| `P` | 暂停 / 继续 |
| `R` | 重新开始（任意时刻） |
| `Q` | 退出 |

规则细节：

- 不能直接反向掉头（会忽略该次输入，游戏继续）。
- 一帧内最多缓存 2 次转向（`config.MAX_PENDING_MOVES`），快速连按不会丢失，也不会攒成一串乱转。
- 结束时（或填满棋盘时）显示最终分数，按 `R` 重开、按 `Q` 退出。

## 玩法与数值

- 棋盘默认 20×20，蛇初始长度 3。
- 吃到食物 `*`：分数 +1，蛇长 +1，并在空格中随机生成新食物。
- 撞墙或撞到自己身体即结束；把整块棋盘填满则判定胜利。
- 速度随分数线性加快：`每步秒数 = BASE_TICK - 分数 × TICK_DECREASE`，
  默认从 `0.12 s/步`（约 8.3 步/秒）开始，每得 1 分快 `0.004 s`，
  下限 `0.045 s/步`（约 22 步/秒），不会无限加速。

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--width N` | 棋盘宽度（格），默认 20，最小 5 |
| `--height N` | 棋盘高度（格），默认 20，最小 5 |
| `--seed N` | 随机种子；给定后食物序列可复现 |
| `--headless [N]` | 无头模式跑 N 步（省略 N 则用 `DEFAULT_HEADLESS_STEPS=100`），打印结果后退出 |
| `--no-color` | 关闭 ANSI 颜色，适合单色终端或日志重定向 |

示例：

```bash
python snake_game/main.py --seed 7           # 固定食物序列的交互局
python snake_game/main.py --width 30 --height 15 --no-color
python snake_game/main.py --headless 100 --seed 7
```

固定种子 + 固定步数 ⇒ 输出完全可复现，便于回归对比。

## 无头模式与自检

`--headless` 不读键盘、不写光标控制序列，用内置的确定性策略（优先靠近食物，
其次选择邻格更空旷、不会立刻撞死的方向）推进 `N` 步，然后打印一行机器可读的结果：

```
$ python snake_game/main.py --headless 100 --seed 7
mode=headless steps_requested=100 steps_done=100
board=20x20 score=5 length=8 over=False won=False
```

若中途结束，会额外打印 `final score: <分数>`。

此外，当 **stdin 或 stdout 不是终端**（管道、重定向、CI 或自检工具直接调用）时，
即使没写 `--headless` 也会自动切到无头模式，避免在拿不到按键的环境里空转。
因此在终端里直接 `python snake_game/main.py` 是交互游戏，被重定向时则是自检。

## 实现要点

- **逻辑与终端分离**：`game.py` 的 `SnakeGame` 只维护状态（`snake` / `food` / `score` /
  `over` / `won`），`main.py` 负责输入与显示，因此逻辑可以脱离终端测试。
- **固定时间步**：主循环用 `time.monotonic()` 累计时间，只有跨过一个 `tick_interval(score)`
  才调用一次 `step()`；渲染与输入轮询不推进世界状态，速度与帧率、机器性能无关。
- **非阻塞读键**：Windows 用 `msvcrt.kbhit()/getwch()`，POSIX 用 `termios` 的 cbreak 模式 +
  `select`，不阻塞主循环；退出时恢复终端原属性并重新显示光标。
- **整屏重绘**：每帧 `\x1b[H` 回到左上角，逐行输出且每行末尾 `\x1b[K` 清到行尾，
  全帧末尾 `\x1b[J` 清到屏幕底部，短行不会留下上一帧残影，也无需闪屏式的 `\x1b[2J` 刷新。
- **ASCII 字符集**：只使用 `@ o *` 与 `| - +`，避免不同代码页下出现乱码或
  `UnicodeEncodeError`。
- Windows 10+ 启动时通过 `SetConsoleMode` 开启 VT 转义支持；开启失败也不会崩，
  只是颜色不可用（可加 `--no-color`）。

## 已知限制

- 交互模式需要真实终端。若 stdin 或 stdout 被管道 / 重定向，会自动转为无头模式，
  此时无法用键盘操作（这是刻意设计，用于自检；需要玩请直接在终端里启动）。
- 未做“终端窗口过小”检测：棋盘比窗口宽时会出现折行，建议用默认 20×20 或调小 `--width`。
- 颜色依赖 ANSI 转义序列；极老的 Windows 控制台、部分 IDE 运行窗口可能不识别，
  请用 `--no-color`。
- 无头模式的内置策略只是贪心近似，不保证最优、也不保证一定赢，仅用于可复现的冒烟自检。
- 食物在空格中随机生成；当蛇几乎填满棋盘时可能耗时略增，但不会死循环。
- 分数、加速参数与字符均为 `config.py` 中的常量，需改配置才能调整，命令行只暴露尺寸 / 种子 / 颜色。
