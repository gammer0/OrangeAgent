---
name: fluid-particle-sim
description: 用 pygame + 粒子法做二维液态水仿真：密度约束不可压缩、重力积分、边界碰撞、固体遮挡。用于编写或验收 fluid 项目。
---

# 二维液态水粒子仿真规范

## 目标

用**粒子法**（类似 PBF / Position-Based Fluids）在 pygame 里模拟液态水：

- 粒子受重力下落，彼此排斥以维持**不可压缩性**
- 用户可以放置**固体**（阻碍水流）和**液体**（增加粒子）
- 所有约束在**位置层面**求解，速度由位置差反推——这是 PBF 的核心，也是"符合物理"的关键

## 目录结构（必须）

```
fluid/
├── main.py        # pygame 主循环：事件、固定步长、渲染
├── core.py        # 纯物理：ParticleSystem，不 import pygame
├── config.py      # 常量：半径、重力、刚度、迭代次数、颜色
└── README.md      # 运行方式、操作说明、物理模型说明
```

**`core.py` 不得 import pygame。** 这是硬性要求——物理层必须能被无头测试直接驱动，
不能靠"看图判断对不对"。

## 必须提供的 API（验收契约）

```python
# core.py
class ParticleSystem:
    def __init__(self, width: int, height: int, seed: int | None = None): ...
        # width/height 为世界尺寸（像素或格子）

    # --- 液体 ---
    def add_water(self, x: float, y: float, radius: float, count: int) -> int: ...
        # 在 (x, y) 附近以 radius 为散布半径添加 count 个粒子，返回实际添加数
    def clear_water(self) -> None: ...

    # --- 固体 ---
    def add_solid(self, rect: tuple[float, float, float, float]) -> None: ...
        # (x, y, w, h)；固体参与碰撞，不被液体穿透
    def clear_solids(self) -> None: ...

    # --- 推进 ---
    def step(self, dt: float) -> None: ...
        # 一次物理子步：重力 → 密度约束迭代 → 边界/固体碰撞 → 速度更新

    # --- 只读状态 ---
    @property
    def particles(self) -> list[tuple[float, float, float, float]]: ...
        # [(x, y, vx, vy), ...]，只读副本
    @property
    def solids(self) -> list[tuple[float, float, float, float]]: ...
    @property
    def total_mass(self) -> float: ...
        # 粒子数 × 单位质量，质量守恒的观测值
    @property
    def kinetic_energy(self) -> float: ...
        # Σ 0.5·|v|²，静置后应趋近于 0
```

## 物理要求（会被工具自动检验）

### 1. 不可压缩性

- 使用**密度约束**：对每个粒子统计邻域内粒子数，若超过静止密度则沿连线推开。
- 每个 `step()` 至少 **3 次**约束迭代（config 里可配）。
- 验收：静置 300 步后，粒子间平均距离不得小于 `1.6 × 半径`（不重叠）。

### 2. 质量守恒

- `add_water` / `clear_water` 之外**不得增删粒子**。
- 验收：跑 500 步前后 `total_mass` 完全相等。

### 3. 边界与固体

- 粒子必须留在世界矩形内，**不得穿墙**。
- 粒子不得进入固体矩形内部。
- 验收：跑 500 步后，所有粒子坐标都在合法区域内，越界个数为 0。

### 4. 重力与静置

- 重力向下，`g` 在 config 中可配（建议 900 px/s²）。
- 液体应向下流动并逐渐静止。
- 验收：从高处落下 500 步后，`kinetic_energy` 相比第 50 步下降 ≥ 60%。

### 5. 阻尼

- 每步对速度施加轻微阻尼（建议 0.98~0.995），否则粒子会永远抖动。
- 阻尼不得写死的"瞬移归零"，必须保留惯性。

## 关键技术点（容易写错的地方）

### 邻域搜索必须有空间网格

朴素 O(n²) 在 2000 粒子时会卡死。用**均匀网格**（cell size = 2×半径）：

```python
cell = 2.0 * radius
grid = {}
for i, (x, y, *_) in enumerate(pos):
    key = (int(x // cell), int(y // cell))
    grid.setdefault(key, []).append(i)
```

查邻域时只遍历 3×3 个相邻格子。

### 密度计算用核函数

推荐 poly6 核或简化的线性衰减：

```
rho_i = Σ_j W(|p_i - p_j|)          # j 包含 3×3 格内所有粒子
C_i   = max(0, rho_i / rho0 - 1)     # 密度约束残差
```

`rho0` 取"粒子以 1.1×半径间距均匀排布"时的密度。

### 约束推进用位置修正，不要直接改速度

```
Δp_i = -k · Σ_j (C_i + C_j) · ∇W · 方向
```

最后再 `v = (p_new - p_old) / dt`。**直接给速度加速度是错的**，
那会破坏不可压缩性并产生爆炸。

### 固体处理用"位置投影"

粒子进入固体后，把它沿最近的面推出去，并把该方向速度归零（或反向乘摩擦系数）。

## 不能用简单写法糊弄

以下做法会被验收工具判为不合格：

- 粒子只是自由落体 + 撞地弹跳（没有密度约束）
- 用 `max(0, min(W, x))` 之类的方式把粒子"夹"在格子里
- 把所有粒子的速度每步乘 0 让它"看起来静下来了"
- 只在 `add_water` 时排布一次，`step()` 里不做任何粒子间作用

## 主循环要求（pygame）

- 固定物理步长（如 1/60 秒），渲染与物理分离
- 用累加器处理可变帧率，避免物理速度依赖帧率
- 每帧最多补 3 个物理子步，防止卡顿后"追帧爆炸"
- 左侧/右侧面板显示：粒子数、FPS、动能、约束迭代数

## 交互要求

| 操作 | 效果 |
|---|---|
| 左键拖拽 | 连续添加液体粒子 |
| 右键拖拽 | 绘制固体矩形（按下到抬起为一个矩形） |
| `C` | 清空液体 |
| `X` | 清空固体 |
| `Space` | 暂停 / 继续 |
| `R` | 重置场景 |
| `Esc` / 关闭窗口 | 退出 |

添加粒子时要限制在上限内（建议 3000），超出时忽略并提示。

## 自检流程（提交前必须全绿）

1. `core.py` 能独立 import，**不依赖 pygame**
2. 无头跑 500 步：质量守恒、无越界、无 NaN
3. 静置后动能下降 ≥ 60%
4. 粒子平均间距 ≥ 1.6×半径
5. pygame 窗口能启动、能加液体、能画固体、能退出

## 交付说明

`README.md` 必须写明：

- 启动命令与依赖安装
- 操作说明
- **物理模型说明**：用了什么约束、迭代几次、参数含义
- 已知限制（如二维、无表面张力、无粘性）
