"""pygame 主循环：事件 -> 固定步长物理 -> 渲染。

分层约定
--------
* 本文件是本项目**唯一** import pygame 的地方；物理层 ``core.py`` 与常量表
  ``config.py`` 都不依赖 pygame，因此可以无头驱动。
* 物理步长固定为 ``config.PHYSICS_DT``，帧率与物理推进用**累加器**解耦：
  帧率高不加速、帧率低不减速；卡顿时每帧最多补 ``config.MAX_SUBSTEPS`` 个子步，
  并把积压的时间丢弃，避免"追帧爆炸"。
* 窗口只在 ``main()`` 里通过 ``pygame.display.set_mode`` 创建，**import 本模块不会
  打开任何窗口**，所以无头测试可以安全地 ``import main`` 或带 ``--headless`` 跑循环。

命令行
------
    python fluid/main.py
    python fluid/main.py --headless --frames 120
"""

from __future__ import annotations

import argparse
import math
import os
import sys

try:  # 作为包导入：fluid.main
    from . import config
    from .core import ParticleSystem
except ImportError:  # 作为脚本导入：sys.path 位于 fluid/ 内
    import config  # type: ignore[no-redef]
    from core import ParticleSystem  # type: ignore[no-redef]

try:  # 顶层 import pygame 只加载模块，不会创建窗口
    import pygame
except ImportError:  # pragma: no cover - 环境缺依赖时给出可读提示
    pygame = None  # type: ignore[assignment]

__all__ = ["main", "build_default_scene", "UiState"]

# ---------------------------------------------------------------- 界面常量
SPAWN_INTERVAL = 0.035      # 左键按住时，每隔多少秒落一批粒子
MAX_FRAME_DT = 0.25         # 单帧时间上限：断点/拖窗口后不要让物理一口气补太多
PANEL_PAD = 14              # 面板内边距
MIN_DRAG_SIZE = 4.0         # 右键拖出的矩形小于该尺寸就忽略（避免误点出零面积固体）


# ---------------------------------------------------------------- 场景
def build_default_scene(system: ParticleSystem) -> None:
    """默认场景：两级台阶 + 底部托盘 + 上方两团水。R 键与启动都走这里。"""
    system.clear_water()
    system.clear_solids()
    w = float(config.WORLD_WIDTH)
    h = float(config.WORLD_HEIGHT)
    system.add_solid((0.10 * w, 0.66 * h, 0.26 * w, 14.0))
    system.add_solid((0.52 * w, 0.48 * h, 0.24 * w, 14.0))
    system.add_solid((0.06 * w, 0.92 * h, 0.88 * w, 16.0))
    system.add_water(0.42 * w, 0.16 * h, 46.0, 420)
    system.add_water(0.78 * w, 0.10 * h, 30.0, 200)


# ---------------------------------------------------------------- 界面状态
class UiState:
    """与物理无关的界面状态：暂停、拖拽、提示语。"""

    def __init__(self) -> None:
        self.paused = False
        self.spawn_timer = 0.0
        self.drag_start: tuple[float, float] | None = None
        self.drag_now: tuple[float, float] | None = None
        self.message = ""
        self.message_ttl = 0.0

    def notify(self, text: str, ttl: float = 2.5) -> None:
        """在面板底部显示一条临时提示。"""
        self.message = text
        self.message_ttl = ttl

    def drag_rect(self) -> tuple[float, float, float, float] | None:
        """当前右键拖拽对应的矩形 ``(x, y, w, h)``；没有拖拽时返回 None。"""
        if self.drag_start is None or self.drag_now is None:
            return None
        x0, y0 = self.drag_start
        x1, y1 = self.drag_now
        return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)

    def tick_message(self, dt: float) -> None:
        if self.message_ttl > 0.0:
            self.message_ttl = max(0.0, self.message_ttl - dt)
            if self.message_ttl == 0.0:
                self.message = ""


# ---------------------------------------------------------------- 小工具
def to_world(pos: tuple[int, int]) -> tuple[float, float] | None:
    """屏幕坐标 -> 世界坐标；落在右侧面板或世界之外时返回 None。"""
    x, y = pos
    if not (0 <= x < config.WORLD_WIDTH and 0 <= y < config.WORLD_HEIGHT):
        return None
    return float(x), float(y)


def spawn_water(system: ParticleSystem, ui: UiState, x: float, y: float) -> None:
    """在世界坐标 (x, y) 落一批水粒子；撞上上限时提示并忽略。"""
    added = system.add_water(x, y, config.DEFAULT_SPAWN_RADIUS, config.DEFAULT_SPAWN_COUNT)
    if added == 0:
        ui.notify(f"已达粒子上限 {config.MAX_PARTICLES}")


def load_font(size: int):
    """优先找能显示中文的系统字体，找不到就退回默认字体（标签改用英文）。"""
    names = (
        "microsoftyahei", "msyh", "simhei", "dengxian", "pingfangsc",
        "notosanscjksc", "arialunicodems", "wqymicrohei",
    )
    for name in names:
        path = pygame.font.match_font(name)
        if not path:
            continue
        try:
            return pygame.font.Font(path, size), True
        except OSError:
            continue
    return pygame.font.Font(None, size + 4), False


def make_droplet(radius: float, fill, edge):
    """预渲染一滴水的贴图，避免每帧对上千个粒子重复画圆。"""
    r = max(1, int(round(radius)))
    d = 2 * r + 2
    surf = pygame.Surface((d, d), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    pygame.draw.circle(surf, fill, (d // 2, d // 2), r)
    pygame.draw.circle(surf, edge, (d // 2, d // 2), r, 1)
    return surf


_LABELS_CJK = {
    "title": "二维液态水仿真",
    "particles": "粒子数",
    "fps": "帧率",
    "energy": "动能",
    "iter": "约束迭代",
    "state": "状态",
    "running": "运行中",
    "paused": "已暂停",
    "help": [
        "左键拖拽：加液体",
        "右键拖拽：画固体",
        "C 清液体  X 清固体",
        "Space 暂停 / 继续",
        "R 重置   Esc 退出",
    ],
}
_LABELS_EN = {
    "title": "Fluid Particle Sim",
    "particles": "Particles",
    "fps": "FPS",
    "energy": "Energy",
    "iter": "Iterations",
    "state": "State",
    "running": "running",
    "paused": "paused",
    "help": [
        "LMB drag: add water",
        "RMB drag: draw solid",
        "C clear water  X solids",
        "Space pause / resume",
        "R reset   Esc quit",
    ],
}


# ---------------------------------------------------------------- 渲染
def draw_frame(screen, system: ParticleSystem, ui: UiState, ctx: dict, fps: float) -> None:
    """画一帧：世界背景、固体、粒子、拖拽预览、右侧信息面板。"""
    labels = ctx["labels"]
    font_big = ctx["font_big"]
    font_small = ctx["font_small"]
    droplet = ctx["droplet"]
    world = ctx["world_rect"]

    screen.fill(config.COLOR_PANEL)
    pygame.draw.rect(screen, config.COLOR_BG, world)
    pygame.draw.rect(screen, config.COLOR_SOLID_EDGE, world, 1)

    for sx, sy, sw, sh in system.solids:
        rect = pygame.Rect(int(sx), int(sy), max(1, int(sw)), max(1, int(sh)))
        pygame.draw.rect(screen, config.COLOR_SOLID, rect)
        pygame.draw.rect(screen, config.COLOR_SOLID_EDGE, rect, 2)

    half = droplet.get_width() // 2
    blit = screen.blit
    for x, y, _vx, _vy in system.particles:
        blit(droplet, (int(x) - half, int(y) - half))

    drag = ui.drag_rect()
    if drag is not None and drag[2] > 0.0 and drag[3] > 0.0:
        pygame.draw.rect(
            screen,
            config.COLOR_SOLID_EDGE,
            pygame.Rect(int(drag[0]), int(drag[1]), max(1, int(drag[2])), max(1, int(drag[3]))),
            1,
        )

    # --- 侧栏
    x0 = config.WORLD_WIDTH + PANEL_PAD
    right = config.WORLD_WIDTH + config.PANEL_WIDTH - PANEL_PAD
    y = PANEL_PAD
    title = font_big.render(labels["title"], True, config.COLOR_TEXT)
    screen.blit(title, (x0, y))
    y += title.get_height() + 10
    pygame.draw.line(screen, config.COLOR_SOLID, (x0, y), (right, y))
    y += 12

    rows = [
        (labels["particles"], f"{system.count} / {config.MAX_PARTICLES}"),
        (labels["fps"], f"{fps:5.1f}"),
        (labels["energy"], f"{system.kinetic_energy:,.0f}"),
        (labels["iter"], str(config.CONSTRAINT_ITERATIONS)),
        (labels["state"], labels["paused"] if ui.paused else labels["running"]),
    ]
    lh = font_small.get_height() + 5
    for name, value in rows:
        screen.blit(font_small.render(name, True, config.COLOR_SOLID_EDGE), (x0, y))
        surf = font_small.render(value, True, config.COLOR_TEXT)
        screen.blit(surf, (right - surf.get_width(), y))
        y += lh

    y += 8
    for line in labels["help"]:
        screen.blit(font_small.render(line, True, config.COLOR_SOLID_EDGE), (x0, y))
        y += lh

    if ui.message:
        y += 6
        screen.blit(font_small.render(ui.message, True, config.COLOR_WATER_EDGE), (x0, y))


# ---------------------------------------------------------------- 主流程
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="二维液态水粒子仿真（pygame）")
    parser.add_argument("--headless", action="store_true", help="使用 dummy 视频驱动，不建真实窗口")
    parser.add_argument("--frames", type=int, default=0, help="跑满 N 帧后自动退出（0 = 一直运行）")
    parser.add_argument("--auto", action="store_true", help="自检模式：自动加液体/画固体，验证交互路径")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """建窗口并进入主循环；返回进程退出码。"""
    if pygame is None:
        print("缺少依赖 pygame，请先执行：python -m pip install pygame", file=sys.stderr)
        return 2
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()
    pygame.display.set_caption("Fluid Particles (PBF)")
    screen = pygame.display.set_mode(
        (config.WORLD_WIDTH + config.PANEL_WIDTH, config.WORLD_HEIGHT)
    )
    clock = pygame.time.Clock()

    font_big, cjk = load_font(20)
    font_small, _ = load_font(14)
    ctx = {
        "labels": _LABELS_CJK if cjk else _LABELS_EN,
        "font_big": font_big,
        "font_small": font_small,
        "droplet": make_droplet(
            config.PARTICLE_RADIUS,
            (*config.COLOR_WATER, 210),
            (*config.COLOR_WATER_EDGE, 150),
        ),
        "world_rect": pygame.Rect(0, 0, config.WORLD_WIDTH, config.WORLD_HEIGHT),
    }

    system = ParticleSystem(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    build_default_scene(system)

    ui = UiState()
    accumulator = 0.0
    frames = 0
    running = True

    while running:
        # --- 单帧时间：夹住上限，避免断点恢复后物理一次补太多
        frame_dt = clock.tick(config.FPS) / 1000.0
        frame_dt = max(0.0, min(frame_dt, MAX_FRAME_DT))
        frames += 1

        # --- 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    system.clear_water()
                    ui.notify("已清空液体")
                elif event.key == pygame.K_x:
                    system.clear_solids()
                    ui.notify("已清空固体")
                elif event.key == pygame.K_SPACE:
                    ui.paused = not ui.paused
                elif event.key == pygame.K_r:
                    build_default_scene(system)
                    ui.spawn_timer = 0.0
                    ui.drag_start = ui.drag_now = None
                    ui.notify("场景已重置")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = to_world(event.pos)
                if pos is None:
                    continue
                if event.button == 1:
                    spawn_water(system, ui, *pos)
                    ui.spawn_timer = 0.0
                elif event.button == 3:
                    ui.drag_start = pos
                    ui.drag_now = pos
            elif event.type == pygame.MOUSEMOTION and ui.drag_start is not None:
                pos = to_world(event.pos)
                if pos is not None:
                    ui.drag_now = pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                rect = ui.drag_rect()
                ui.drag_start = ui.drag_now = None
                if rect is not None and rect[2] >= MIN_DRAG_SIZE and rect[3] >= MIN_DRAG_SIZE:
                    system.add_solid(rect)
                    ui.notify(f"固体 {int(rect[2])}x{int(rect[3])}")
                elif rect is not None:
                    ui.notify("太小的矩形已忽略", 1.2)

        # --- 自检模式：无鼠标时也走一遍"加液体 / 画固体"的代码路径
        if args.auto:
            if frames == 5:
                system.add_solid((0.30 * config.WORLD_WIDTH, 0.40 * config.WORLD_HEIGHT, 90.0, 10.0))
            if frames < 40:
                sx = 0.5 * config.WORLD_WIDTH + 40.0 * math.sin(frames * 0.2)
                spawn_water(system, ui, sx, 70.0)

        # --- 左键按住：按固定间隔连续落水（不依赖帧率）
        if pygame.mouse.get_pressed()[0]:
            pos = to_world(pygame.mouse.get_pos())
            if pos is not None:
                ui.spawn_timer += frame_dt
                if ui.spawn_timer >= SPAWN_INTERVAL:
                    ui.spawn_timer = 0.0
                    spawn_water(system, ui, *pos)
        else:
            ui.spawn_timer = 0.0
        if pygame.mouse.get_pressed()[2] and ui.drag_start is not None:
            pos = to_world(pygame.mouse.get_pos())
            if pos is not None:
                ui.drag_now = pos

        # --- 物理：固定步长 + 累加器，每帧最多 MAX_SUBSTEPS 个子步
        if not ui.paused:
            accumulator += frame_dt
            steps = 0
            while accumulator >= config.PHYSICS_DT and steps < config.MAX_SUBSTEPS:
                system.step(config.PHYSICS_DT)
                accumulator -= config.PHYSICS_DT
                steps += 1
            if accumulator >= config.PHYSICS_DT:
                # 补到上限还欠时间：丢弃积压，宁可物理慢一点也不追帧爆炸
                accumulator = 0.0

        ui.tick_message(frame_dt)
        draw_frame(screen, system, ui, ctx, clock.get_fps())
        pygame.display.flip()

        if args.frames and frames >= args.frames:
            running = False

    summary = {
        "frames": frames,
        "particles": system.count,
        "solids": len(system.solids),
        "mass": round(system.total_mass, 4),
        "kinetic_energy": round(system.kinetic_energy, 4),
        "paused": ui.paused,
    }
    pygame.quit()
    if args.frames:
        print(f"退出摘要：{summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口：只在直接运行时被调用，import 本模块不会建窗口。"""
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
