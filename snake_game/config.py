"""贪吃蛇游戏的常量配置（纯常量，无副作用）。

只放常量与颜色，不放逻辑，方便被 game.py / main.py 共用。
"""

from __future__ import annotations

# --- 棋盘与初始状态 -------------------------------------------------------
WIDTH = 20          # 棋盘宽（格）
HEIGHT = 20         # 棋盘高（格）
MIN_SIZE = 6        # 允许的最小边长（要放得下 3 节蛇）
INITIAL_LENGTH = 3  # 初始蛇长

# --- 速度（固定时间步，单位：步/秒） --------------------------------------
INITIAL_SPS = 10.0      # 初始速度
MAX_SPS = 22.0          # 速度上限
SPS_PER_FOOD = 0.6      # 每吃 1 个食物提升的步/秒
MIN_FRAME_SLEEP = 0.002  # 主循环最小休眠，避免空转占满 CPU

# --- 渲染字符 -------------------------------------------------------------
CH_HEAD = "@"
CH_BODY = "o"
CH_FOOD = "*"
CH_EMPTY = " "
BORDER_H = "-"
BORDER_V = "|"
BORDER_CORNER = "+"
CELL_WIDTH = 1

# --- 颜色（ANSI 转义） ----------------------------------------------------
COLOR_RESET = "\x1b[0m"
COLOR_BORDER = "\x1b[90m"
COLOR_HEAD = "\x1b[92m"
COLOR_BODY = "\x1b[32m"
COLOR_FOOD = "\x1b[91m"
COLOR_TEXT = "\x1b[97m"
COLOR_OVER = "\x1b[91m"
COLOR_WIN = "\x1b[93m"

# --- 按键 -----------------------------------------------------------------
KEYS_QUIT = ("q", "Q")
KEYS_PAUSE = ("p", "P")
KEYS_RESTART = ("r", "R")
DIRECTION_KEYS = {
    "w": "up", "W": "up", "up": "up",
    "s": "down", "S": "down", "down": "down",
    "a": "left", "A": "left", "left": "left",
    "d": "right", "D": "right", "right": "right",
}
