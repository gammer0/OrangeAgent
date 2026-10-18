# 数据结构设计

> 本目录只定义**数据结构**：字段、类型、约束、关系。
> 不包含设计动机、方案比较、历史决策——那些见 `docs/01-系统架构设计.md`。
>
> **快速上手**：先看 [全系统架构与数据流](13-全系统架构与数据流.md)，再按需查阅各组文档。

---

## 1. 总览

| 组 | 结构 | 归属模块 | 可变性 | 文档 |
|----|------|---------|--------|------|
| G | `Inbox` / `UserInput` / `InterruptRequest` | 接入层 | 队列可变 | [接入层](01-接入层.md) |
| H | `Session` / `SessionRegistry` / `EvictionPolicy` / `SessionRecord` | 会话层 | 会话级可变 | [会话层](02-会话层.md) |
| A | `Message` / `SystemMsg` / `UserMsg` / `AssistantMsg` / `ToolMsg` / `ToolCall` | 上下文层 | 不可变 | [消息](03-消息.md) |
| B | `FlatStore` / `ToolDef` / `ToolTree` / `ToolNode` / `Permission` / `TreeQueryState` / `DiscoveredTools` / `AgentIdentity` / `AgentPermissions` | 能力层 | `FlatStore` 与树只读；会话状态可变 | [工具与权限](04-工具与权限.md) |
| C | `ProjectMemory` / `MemoryEntry` / `MemoryExtraction` | 记忆层 | 长期记忆只追加 | [记忆](05-记忆.md) |
| D | `BatchPlan` / `ExecGroup` / `ExecResult` | 执行层 | 单批次生命周期 | [执行](06-执行.md) |
| E | `TraceEvent` | 观测层 | 只追加 | [轨迹](07-轨迹.md) |
| F | `Watermark` / `JudgeResult` / `RouteDecision` | 上下文层 / 路由层 | 会话级可变 | [运行时状态](08-运行时状态.md) |
| K | `RateWindow` / `RateEvent` / `ProviderHealth` | 模型层 | 全局可变 | [模型层](10-模型层.md) |

---

## 2. 文档索引

| # | 文件 | 内容 |
|---|------|------|
| 01 | [接入层](01-接入层.md) | 双通道语义、安全点编号、`Inbox` / `UserInput` / `InterruptRequest` |
| 02 | [会话层](02-会话层.md) | `Session` / `SessionRegistry` / `EvictionPolicy` / SQLite 持久化 |
| 03 | [消息](03-消息.md) | `Message` 家族、`SystemMsg` 五段组成、校验器、序列化投影 |
| 04 | [工具与权限](04-工具与权限.md) | 扁平存储与树视图、构建算法、挂载与卸载、权限模型、schema 下发 |
| 05 | [记忆](05-记忆.md) | `WORK.md` / `MEMORY.md` / 结算提取 |
| 06 | [执行](06-执行.md) | `BatchPlan` / `ExecGroup` / 子批次安全点 |
| 07 | [轨迹](07-轨迹.md) | `TraceEvent` 与 payload |
| 08 | [运行时状态](08-运行时状态.md) | `Watermark` / `JudgeResult` / `RouteDecision` |
| 09 | [参数校验](09-参数校验.md) | `arguments` 四道关卡与 `3d` |
| 10 | [模型层](10-模型层.md) | `RateWindow` / `ProviderHealth` / 成本归集 |
| 11 | [结构关系](11-结构关系.md) | 全局关系图 |
| 12 | [生命周期](12-生命周期.md) | 各结构创建与销毁时机 |
| 13 | [全系统架构与数据流](13-全系统架构与数据流.md) | 架构图、时序图、生命周期矩阵、持久化落点 |

---

## 3. 约束编号总表

| 前缀 | 范围 | 所属文档 |
|------|------|---------|
| `A0a`–`A0e` | `SystemMsg` 组成 | 03-消息 |
| `A1`–`A10` | 消息 | 03-消息 |
| `A11`–`A14` | 临时注入 | 03-消息 |
| `G1`–`G12` | 接入层 | 01-接入层 |
| `H1`–`H14` | 会话层 | 02-会话层 |
| `B1`–`B25` | 工具与权限 | 04-工具与权限 |
| `C1`–`C8` | 记忆 | 05-记忆 |
| `D1`–`D8` | 执行 | 06-执行 |
| `E1`–`E3` | 轨迹 | 07-轨迹 |
| `F1`–`F2` | 运行时状态 | 08-运行时状态 |
| `J1`–`J4` | 参数校验 | 09-参数校验 |
| `K1`–`K7` | 模型层 | 10-模型层 |

---

## 4. 待定议题

当前所有已识别结构均已完成设计。后续新增能力时在此追加。
