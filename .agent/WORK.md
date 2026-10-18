# 工作记忆

## 最终结构
- snake_game/config.py：全部常量（棋盘、速度、字符、配色、按键），无逻辑
- snake_game/game.py：纯逻辑层 SnakeGame，不碰终端/键盘/时间
- snake_game/main.py：入口，参数解析 + 固定时间步主循环 + 整屏重绘
- snake_game/tests/test_game.py：9 个 unittest 回归测试
- snake_game/README.md：运行方式、按键、计分规则

## 验收命令
- python -m snake_game.main --headless 200 --seed 7
- python -m unittest discover -s snake_game/tests -t . -v

## 设计决定
- 单槽 _pending 会让一帧内的连续两次转向丢掉一次，手感发黏；改为容量 2 的方向队列后按序生效。
