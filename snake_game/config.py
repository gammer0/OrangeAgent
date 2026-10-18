"""贪吃蛇常量配置：棋盘尺寸、速度曲线、终端字符与颜色。

本模块只放常量，不参与任何逻辑判断，便于统一调参。
"""

# ---- 棋盘 ----
WIDTH = 20          # 棋盘宽（格）
HEIGHT = 20         # 棋盘高（格）

# ---- 蛇 ----
START_LENGTH = 3    # 初始蛇长

# ---- 固定时间步（秒/步）----
BASE_TICK = 0.12    # 初始 0.12 秒/步，约 8.3 步/秒
TICK_DECREASE = 0.004   # 每吃一个食物减少的时间
MIN_TICK = 0.045    # 加速下限，约 22 步/秒

# ---- 输入 ----
MAX_PENDING_MOVES = 2   # 一帧内最多缓存 2 次转向

# ---- 终端字符 ----
GLYPH_HEAD = "@"
GLYPH_BODY = "o"
GLYPH_FOOD = "*"
GLYPH_EMPTY = " "

# ---- ANSI 颜色 ----
COLOR_RESET = "\x1b[0m"
COLOR_HEAD = "\x1b[92m"
COLOR_BODY = "\x1b[32m"
COLOR_FOOD = "\x1b[91m"
COLOR_TEXT = "\x1b[97m"
COLOR_WARN = "\x1b[93m"

# ---- 无头自检 ----
DEFAULT_HEADLESS_STEPS = 100
