"""二维液态水的纯物理层：位置约束粒子法（PBF / Position-Based Fluids 风格）。

设计约束（来自项目规范）：

* 本模块**不 import pygame**，可以被无头驱动：验收工具直接实例化 ``ParticleSystem``
  并调用 ``step``，不依赖任何窗口。
* 密度约束在**位置层面**求解，迭代若干次之后由 ``(p_new - p_prev) / dt`` 反推速度，
  从不直接给速度叠加"压力加速度"。
* 邻域搜索使用均匀空间网格（格子边长 = 核函数支撑半径 h），复杂度约 O(n)。
* 状态保存在并行的 float 列表里，只读属性返回副本，外部无法就地篡改。

每个 step 的流程：

    重力 -> 位置预测 -> [密度约束迭代 -> 边界/固体投影] x N
         -> 由位置差反推速度 -> XSPH 粘性 -> 阻尼
"""

from __future__ import annotations

import math
import random
from typing import Iterator

try:  # 作为包导入：fluid.core
    from . import config
except ImportError:  # 作为顶层模块导入：cwd 位于 fluid/ 内
    import config  # type: ignore[no-redef]

__all__ = ["ParticleSystem", "poly6", "kernel_gradient", "lattice_density", "REST_DENSITY"]

# --------------------------------------------------------------------------- 核函数
H = float(config.SMOOTHING_RADIUS)   # 支撑半径
H2 = H * H
_PI = math.pi

# 归一化的二维 poly6 核：W(r) = 4/(pi h^8) * (h^2 - r^2)^3，满足 ∫W dA = 1
POLY6_COEF = 4.0 / (_PI * H**8)
# 二维 spiky 核梯度幅值系数：|∇W(r)| = 45/(pi h^6) * (h - r)^2
SPIKY_COEF = 45.0 / (_PI * H**6)


def poly6(r: float) -> float:
    """poly6 核在距离 r 处的取值（r >= h 时为 0）。用于密度求和与邻域加权。"""
    if r >= H:
        return 0.0
    d = H2 - r * r
    return POLY6_COEF * d * d * d


def kernel_gradient(r: float) -> float:
    """spiky 核梯度的**幅值**（方向由调用方按单位向量决定）。"""
    if r <= 1e-9 or r >= H:
        return 0.0
    t = H - r
    return SPIKY_COEF * t * t


def lattice_density(spacing: float) -> float:
    """间距为 ``spacing`` 的无限正方点阵上，一个粒子的 poly6 密度总和。

    ``rho0`` 就用它来定：取"粒子按 ``config.REST_SPACING`` 均匀排布"时的密度。
    这样 C_i = rho_i / rho0 - 1 在这种静止排布下严格为 0，不会被连续近似
    (1/spacing^2) 的系统偏差顶得整个流体永远处于"被压缩"状态。
    """
    span = int(math.ceil(H / spacing))
    total = poly6(0.0)
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            if i == 0 and j == 0:
                continue
            r = spacing * math.hypot(i, j)
            if r < H:
                total += poly6(r)
    return total


#: 静止密度：粒子按 REST_SPACING 均匀排布时的 poly6 密度
REST_DENSITY = lattice_density(float(config.REST_SPACING))


# --------------------------------------------------------------------------- 系统
class ParticleSystem:
    """二维液态水的粒子系统（纯物理，无渲染）。"""

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        self.width = float(width)
        self.height = float(height)
        self._rng = random.Random(seed)

        # 并行状态数组：第 i 个粒子为 (x, y, vx, vy)
        self._x: list[float] = []
        self._y: list[float] = []
        self._vx: list[float] = []
        self._vy: list[float] = []
        # 固体矩形 (x, y, w, h)
        self._solids: list[tuple[float, float, float, float]] = []

        #: 密度约束的静止密度（只读暴露，供界面显示）
        self.rho0 = REST_DENSITY

    # ------------------------------------------------------------------ 液体
    def add_water(self, x: float, y: float, radius: float, count: int) -> int:
        """在 ``(x, y)`` 附近、以 ``radius`` 为散布半径添加 ``count`` 个水粒子。

        返回实际添加的数量。超出 ``MAX_PARTICLES`` 上限时不添加。
        """
        cx = float(x)
        cy = float(y)
        spread = max(0.0, float(radius))
        want = int(count)
        if want <= 0:
            return 0

        room = config.MAX_PARTICLES - len(self._x)
        want = min(want, max(0, room))
        if want <= 0:
            return 0

        margin = self._margin()
        rng = self._rng
        added = 0
        for _ in range(want):
            # 圆盘内均匀采样（开方保证面积均匀）
            ang = rng.random() * 2.0 * _PI
            dist = spread * math.sqrt(rng.random())
            px = cx + dist * math.cos(ang)
            py = cy + dist * math.sin(ang)
            self._x.append(self._clip(px, margin, self.width - margin))
            self._y.append(self._clip(py, margin, self.height - margin))
            self._vx.append(0.0)
            self._vy.append(0.0)
            added += 1

        if added:
            # 出生点若落在固体/墙里，立刻投影到合法位置
            self._project_positions()
        return added

    def clear_water(self) -> None:
        """清空所有水粒子。"""
        self._x.clear()
        self._y.clear()
        self._vx.clear()
        self._vy.clear()

    # ------------------------------------------------------------------ 固体
    def add_solid(self, rect: tuple[float, float, float, float]) -> None:
        """添加一个固体矩形 ``(x, y, w, h)``；负宽高会被规范化。"""
        x, y, w, h = (float(v) for v in rect)
        if w < 0.0:
            x += w
            w = -w
        if h < 0.0:
            y += h
            h = -h
        if w <= 0.0 or h <= 0.0:
            return
        # 夹到世界内部，避免固体跑到墙外导致粒子被夹死
        x0 = max(0.0, min(x, self.width))
        y0 = max(0.0, min(y, self.height))
        x1 = max(0.0, min(x + w, self.width))
        y1 = max(0.0, min(y + h, self.height))
        if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
            return
        self._solids.append((x0, y0, x1 - x0, y1 - y0))
        if self._x:
            self._project_positions()

    def clear_solids(self) -> None:
        """清空所有固体。"""
        self._solids.clear()

    # ------------------------------------------------------------------ 推进
    def step(self, dt: float) -> None:
        """推进一个物理子步。

        重力 -> 位置预测 -> 约束迭代（密度约束 + 最小间距约束 + 边界投影）
        -> 由位置差反推速度 -> XSPH 粘性 -> 阻尼。
        """
        frame_dt = float(dt)
        if frame_dt <= 0.0:
            return
        n = len(self._x)
        if n == 0:
            return
        frame_dt = min(frame_dt, config.MAX_DT)

        prev_x = list(self._x)
        prev_y = list(self._y)
        xs = self._x
        ys = self._y
        vx = self._vx
        vy = self._vy

        # 1) 外力：重力累加到速度
        gravity_dv = config.GRAVITY * frame_dt
        for i in range(n):
            vy[i] += gravity_dv

        # 2) 位置预测（先按当前速度外推，之后全在位置层面修正）
        for i in range(n):
            xs[i] += vx[i] * frame_dt
            ys[i] += vy[i] * frame_dt

        # 3) 约束迭代：密度约束 -> 边界/固体投影 -> 最小间距（非穿透）
        #    邻域列表每轮只枚举一次，两个约束共用同一份候选对
        for _ in range(config.CONSTRAINT_ITERATIONS):
            pairs = self._neighbor_pairs()
            self._solve_density(pairs)
            self._project_positions()
            self._solve_contacts(pairs[0], pairs[1])

        # 4) 速度由位置差反推（PBF 的核心：不是给速度加加速度）
        inv_dt = 1.0 / frame_dt
        for i in range(n):
            vx[i] = (xs[i] - prev_x[i]) * inv_dt
            vy[i] = (ys[i] - prev_y[i]) * inv_dt

        # 5) XSPH 粘性：耗散粒子间的**相对**速度，压掉约束抖动
        self._apply_viscosity()

        # 6) 轻微阻尼（保留惯性，不是把速度归零）
        damping = config.DAMPING
        for i in range(n):
            vx[i] *= damping
            vy[i] *= damping

        # 7) 数值保护：NaN / 位移爆炸都不允许传播出去
        self._sanitize()

    # -------------------------------------------------------------- 空间网格
    def _build_grid(self) -> dict[tuple[int, int], list[int]]:
        """均匀网格；格边长 = 核函数支撑半径 h，查 3x3 邻域即覆盖全部 r < h 的邻居。"""
        cell = float(config.CELL_SIZE)
        cell_inv = 1.0 / cell
        grid: dict[tuple[int, int], list[int]] = {}
        xs = self._x
        ys = self._y
        for i in range(len(xs)):
            key = (int(xs[i] * cell_inv), int(ys[i] * cell_inv))
            bucket = grid.get(key)
            if bucket is None:
                grid[key] = [i]
            else:
                bucket.append(i)
        return grid

    def _neighbor_pairs(self) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
        """枚举所有 r < h 的无序粒子对，返回 (i, j, dx, dy, r)。

        ``dx/dy = p_i - p_j``。每对只出现一次，调用方据此一次遍历完成两遍统计。
        """
        cell = float(config.CELL_SIZE)
        cell_inv = 1.0 / cell
        xs = self._x
        ys = self._y
        grid = self._build_grid()
        get_cell = grid.get
        pair_i: list[int] = []
        pair_j: list[int] = []
        pair_dx: list[float] = []
        pair_dy: list[float] = []
        pair_r: list[float] = []
        for i in range(len(xs)):
            xi = xs[i]
            yi = ys[i]
            cxi = int(xi * cell_inv)
            cyi = int(yi * cell_inv)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    bucket = get_cell((cxi + ox, cyi + oy))
                    if not bucket:
                        continue
                    for j in bucket:
                        if j <= i:          # 每个无序对只处理一次
                            continue
                        dx = xi - xs[j]
                        dy = yi - ys[j]
                        r2 = dx * dx + dy * dy
                        if r2 >= H2:
                            continue
                        if r2 <= 1e-12:
                            # 退化对（位置几乎重合）：确定性微推一下，
                            # 否则 1/r 方向未定义，两颗粒子会永久黏在一起
                            ang = 2.399963229728653 * (i + j)
                            xs[j] += 1e-3 * math.cos(ang)
                            ys[j] += 1e-3 * math.sin(ang)
                            dx = xi - xs[j]
                            dy = yi - ys[j]
                            r2 = dx * dx + dy * dy
                            if r2 <= 1e-12:
                                continue
                        pair_i.append(i)
                        pair_j.append(j)
                        pair_dx.append(dx)
                        pair_dy.append(dy)
                        pair_r.append(math.sqrt(r2))
        return pair_i, pair_j, pair_dx, pair_dy, pair_r

    # -------------------------------------------------------- 密度约束求解器
    def _solve_density(self, pairs=None) -> None:
        """一次密度约束迭代：统计密度 -> 求 lambda -> 沿梯度做位置修正。

        ``pairs`` 可由调用方传入（复用邻域枚举结果）；为 None 时自行枚举。
        """
        n = len(self._x)
        if n == 0:
            return
        xs = self._x
        ys = self._y
        inv_rho0 = 1.0 / self.rho0
        inv_rho0_sq = inv_rho0 * inv_rho0

        if pairs is None:
            pairs = self._neighbor_pairs()
        pair_i, pair_j, pair_dx, pair_dy, pair_r = pairs

        # --- 第一遍：密度和梯度向量和
        w0 = poly6(0.0)
        rho = [w0] * n
        gx = [0.0] * n
        gy = [0.0] * n
        npairs = len(pair_i)
        for k in range(npairs):
            i = pair_i[k]
            j = pair_j[k]
            dx = pair_dx[k]
            dy = pair_dy[k]
            r = pair_r[k]
            w = poly6(r)
            rho[i] += w
            rho[j] += w
            gm = SPIKY_COEF * (H - r) * (H - r)
            inv_r = 1.0 / r
            # ∇_{p_i} W 指向 j，故为 -(gm) * (dx, dy)/r
            ux = gm * dx * inv_r
            uy = gm * dy * inv_r
            gx[i] -= ux
            gy[i] -= uy
            gx[j] += ux
            gy[j] += uy

        # --- lambda：只处理"被压缩"的粒子（C = max(0, rho/rho0 - 1)）
        eps = config.DENSITY_EPSILON
        lam = [0.0] * n
        for i in range(n):
            c = rho[i] * inv_rho0 - 1.0
            if c <= 0.0:
                continue
            grad_sq = (gx[i] * gx[i] + gy[i] * gy[i]) * inv_rho0_sq
            lam[i] = -c / (grad_sq + eps)

        # --- 第二遍：Δp_i = (1/rho0) * Σ_j (λ_i + λ_j) ∇W_ij
        dpx = [0.0] * n
        dpy = [0.0] * n
        for k in range(npairs):
            i = pair_i[k]
            j = pair_j[k]
            s = lam[i] + lam[j]
            if s == 0.0:
                continue
            r = pair_r[k]
            coef = s * SPIKY_COEF * (H - r) * (H - r) * inv_rho0 / r
            cx = coef * pair_dx[k]
            cy = coef * pair_dy[k]
            dpx[i] -= cx
            dpy[i] -= cy
            dpx[j] += cx
            dpy[j] += cy

        # --- 应用修正（带单步位移上限，防止初始重叠时数值爆炸）
        limit = config.MAX_CORRECTION_FACTOR * float(config.REST_SPACING)
        relax = config.RELAXATION
        for i in range(n):
            mx = dpx[i] * relax
            my = dpy[i] * relax
            if mx == 0.0 and my == 0.0:
                continue
            length = math.hypot(mx, my)
            if length > limit:
                scale = limit / length
                mx *= scale
                my *= scale
            xs[i] += mx
            ys[i] += my

    # ------------------------------------------------------------ XSPH 粘性
    def _apply_viscosity(self) -> None:
        """XSPH：v_i += c · Σ_j (v_j - v_i) · W_ij / rho0。

        只把相对速度按核权重拉平（相邻粒子趋向同速），不改变质心速度，
        更不是"每步把速度乘 0"。没有它，位置约束的残差会以高频抖动的形式
        长期驻留在能量里，液体看起来永远在沸腾。
        """
        visc = float(config.VISCOSITY)
        n = len(self._x)
        if visc <= 0.0 or n < 2:
            return
        vx = self._vx
        vy = self._vy
        pair_i, pair_j, _, _, pair_r = self._neighbor_pairs()
        ax = [0.0] * n
        ay = [0.0] * n
        for k in range(len(pair_i)):
            i = pair_i[k]
            j = pair_j[k]
            w = poly6(pair_r[k])
            dvx = (vx[j] - vx[i]) * w
            dvy = (vy[j] - vy[i]) * w
            ax[i] += dvx
            ay[i] += dvy
            ax[j] -= dvx
            ay[j] -= dvy
        scale = visc / self.rho0
        for i in range(n):
            vx[i] += scale * ax[i]
            vy[i] += scale * ay[i]

    # ------------------------------------------------------------ 碰撞投影
    def _margin(self) -> float:
        """粒子中心与世界墙的最小距离（世界太小时自动收缩）。"""
        margin = float(config.COLLISION_MARGIN)
        return max(0.0, min(margin, self.width * 0.5, self.height * 0.5))

    def _clamp_point(self, x: float, y: float) -> tuple[float, float]:
        """把一个位置夹回合法区域：世界矩形内 + 固体矩形外。"""
        margin = self._margin()
        lo_x = margin
        hi_x = self.width - margin
        lo_y = margin
        hi_y = self.height - margin

        # 世界边界
        if x < lo_x:
            x = lo_x
        elif x > hi_x:
            x = hi_x
        if y < lo_y:
            y = lo_y
        elif y > hi_y:
            y = hi_y

        # 固体：沿最近的面推到外面
        for sx, sy, sw, sh in self._solids:
            if not (sx < x < sx + sw and sy < y < sy + sh):
                continue
            left = x - sx
            right = sx + sw - x
            top = y - sy
            bottom = sy + sh - y
            nearest = min(left, right, top, bottom)
            if nearest == left:
                x = sx - margin
            elif nearest == right:
                x = sx + sw + margin
            elif nearest == top:
                y = sy - margin
            else:
                y = sy + sh + margin
            # 推开之后仍可能在墙外（贴墙的固体），再夹一次
            if x < lo_x:
                x = lo_x
            elif x > hi_x:
                x = hi_x
            if y < lo_y:
                y = lo_y
            elif y > hi_y:
                y = hi_y
        return x, y

    def _project_positions(self) -> None:
        """把粒子投影回合法区域：世界矩形内 + 固体矩形外（位置投影，不改速度）。"""
        xs = self._x
        ys = self._y
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            nx, ny = self._clamp_point(x, y)
            if nx != x:
                xs[i] = nx
            if ny != y:
                ys[i] = ny

    # --------------------------------------------------- 最小间距（非穿透）约束
    @staticmethod
    def _separate_dir(i: int, j: int) -> tuple[float, float]:
        """完全重合粒子对的确定性分离方向（黄金角散布，避免总往同一侧撞墙）。"""
        ang = 2.399963229728653 * (i + j + 1)
        return math.cos(ang), math.sin(ang)

    def _shift(self, i: int, dx: float, dy: float) -> tuple[float, float]:
        """把粒子 i 平移 (dx,dy)，夹回合法区域后写入，返回**实际**发生的位移。"""
        xs = self._x
        ys = self._y
        ox = xs[i]
        oy = ys[i]
        nx, ny = self._clamp_point(ox + dx, oy + dy)
        xs[i] = nx
        ys[i] = ny
        return nx - ox, ny - oy

    def _solve_contacts(self, pair_i: list, pair_j: list) -> None:
        """最小间距约束：把间距小于 MIN_DISTANCE 的粒子对沿连线推开。

        为什么单靠密度约束不够：约束式是 ``C = max(0, rho/rho0 - 1)``，只在
        **被压缩**时才有推力。贴墙的那一层和自由液面附近，局部密度长期低于
        rho0（实测 rho/rho0 = 0.80~0.97），于是 C = 0、lambda = 0，
        即使两颗粒子已经重合也没有任何恢复力 —— 这正是长时间运行后重叠累积的
        原因。这里补一条与密度无关的纯几何约束，等价于"粒子不许穿模"。

        推开量两端各分一半；某一端被墙/固体挡住时，把没走完的份额转给另一端。
        否则墙角处两颗粒子会被两面墙夹在同一个点上，永远分不开。
        """
        dmin = float(config.MIN_DISTANCE)
        if dmin <= 0.0 or not pair_i:
            return
        xs = self._x
        ys = self._y
        limit = float(config.MAX_CONTACT_CORRECTION) * dmin  # 单粒子单次推开上限
        dmin_sq = dmin * dmin

        for k in range(len(pair_i)):
            i = pair_i[k]
            j = pair_j[k]
            # 距离按当前位置重算（候选对是在投影前枚举的，位置已变过）
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            r2 = dx * dx + dy * dy
            if r2 >= dmin_sq:
                continue
            r = math.sqrt(r2)
            if r > 1e-9:
                ux = dx / r
                uy = dy / r
            else:
                ux, uy = self._separate_dir(i, j)
            need = min(dmin - r, 2.0 * limit)
            half = 0.5 * need

            # i 沿 +u 走一半，j 沿 -u 走一半
            mx, my = self._shift(i, ux * half, uy * half)
            done = mx * ux + my * uy
            mx, my = self._shift(j, -ux * half, -uy * half)
            done -= mx * ux + my * uy

            # 被墙/固体夹住、没走完的份额转给另一端
            rest = min(need - done, limit)
            if rest > 1e-9:
                mx, my = self._shift(j, -ux * rest, -uy * rest)
                done -= mx * ux + my * uy
                rest = min(need - done, limit)
            if rest > 1e-9:
                self._shift(i, ux * rest, uy * rest)


    # -------------------------------------------------------------- 数值保护
    def _sanitize(self) -> None:
        """兜底：非有限值复位、速度截断。正常情况下不会触发。"""
        n = len(self._x)
        if n == 0:
            return
        margin = self._margin()
        cx = self._clip(self.width * 0.5, margin, self.width - margin)
        cy = self._clip(self.height * 0.5, margin, self.height - margin)
        max_speed = float(config.MAX_SPEED)
        xs = self._x
        ys = self._y
        vx = self._vx
        vy = self._vy

        for i in range(n):
            if not (math.isfinite(xs[i]) and math.isfinite(ys[i])):
                xs[i] = cx
                ys[i] = cy
            if not (math.isfinite(vx[i]) and math.isfinite(vy[i])):
                vx[i] = 0.0
                vy[i] = 0.0
                continue
            speed = math.hypot(vx[i], vy[i])
            if speed > max_speed:
                scale = max_speed / speed
                vx[i] *= scale
                vy[i] *= scale

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        if hi < lo:
            return (lo + hi) * 0.5
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    # -------------------------------------------------------------- 只读状态
    @property
    def particles(self) -> list[tuple[float, float, float, float]]:
        """``[(x, y, vx, vy), ...]`` 的副本。"""
        return list(zip(self._x, self._y, self._vx, self._vy))

    @property
    def solids(self) -> list[tuple[float, float, float, float]]:
        """固体矩形 ``[(x, y, w, h), ...]`` 的副本。"""
        return list(self._solids)

    @property
    def total_mass(self) -> float:
        """粒子数 x 单位质量；除 add_water / clear_water 外不应变化。"""
        return len(self._x) * float(config.UNIT_MASS)

    @property
    def kinetic_energy(self) -> float:
        """Σ 0.5·|v|²；液体静置后应趋近 0。"""
        vx = self._vx
        vy = self._vy
        total = 0.0
        for i in range(len(vx)):
            total += vx[i] * vx[i] + vy[i] * vy[i]
        return 0.5 * total

    @property
    def count(self) -> int:
        """当前水粒子数。"""
        return len(self._x)

    def positions(self) -> Iterator[tuple[float, float]]:
        """迭代 ``(x, y)``，渲染时避免构造完整副本。"""
        return zip(self._x, self._y)
