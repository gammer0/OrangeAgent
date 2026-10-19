# Orange

一个可长期运行、工具能力可扩展的智能体平台。它在 ReAct 循环里调用工具并根据结果继续推理，工具数量增长到数百个时模型仍能找到正确的那个，换模型只改配置不改代码。

```
用户消息 → 上下文装配 → 模型推理 ⇄ 工具执行 → 轨迹落盘
                ↑                        ↓
           会话存储 ←─────────────── SSE 实时推送
```

## 为什么是 Orange

大多数 agent 框架在工具数量增长后会失效：把所有工具的完整定义塞进提示词，几百个工具就撑爆上下文，模型也开始选错。Orange 的做法是把能力组织成一棵**可按需展开的树**——模型先看到一级目录，再用 `tree_query` 下钻到具体路径。前缀稳定，新增能力不会破坏已建立的上下文缓存。

配套的几个设计取向：

| 特性 | 实现方式 |
|---|---|
| **前缀稳定性优先** | `messages[0]` 逐字节不变；能力树展开走工具返回值，不重排版提示词 |
| **失败即数据** | 工具异常不中断循环，降级为模型可观察的结果，由它决定重试或换路 |
| **优化可回退** | 路由失效时无损退化为全量索引 |
| **能力归属唯一** | 工具树路径即归属，一个能力只有一个可达路径 |
| **可复原** | 每步下钻都有轨迹，落盘到 SQLite，崩溃后可定位 |

## 内置能力

本发布版**内置**的只有工作区文件访问，不含命令执行、Python 运行与网络请求——这些能力会扩大攻击面。需要时可通过外置工具或 MCP 服务器自行挂载，见[加载外置工具](#加载外置工具)。

| 工具 | 说明 |
|---|---|
| `/tool/os/file/read` | 按行读取工作目录内的文本文件 |
| `/tool/os/file/list` | 列出目录条目（文件/目录 + 大小） |
| `/tool/os/file/write` | 整体写入文件（覆盖），自动建目录 |
| `/tool/os/file/append` | 追加写入文件 |

另有**树查询**能力：每个中间路径（如 `/tool`、`/tool/os`）本身就是可调用节点，调用即返回下一层子节点。这是系统机制，不是普通工具，因此不占工具槽位。

首次启动会自动写入 `capabilities.json` 并挂载上述 4 个工具，无需手工配置。

## 加载外置工具

除内置工具外，还可以挂载两类外来能力：**Python 函数**（进程内调用）与 **MCP 服务器**（JSON-RPC over stdio）。

### 一、Python 外置工具

#### 1. 准备一个 Python 文件

文件里需要一个函数，默认名字是 `run`：

```python
# ./ext/sort_numbers.py
def run(numbers: list[int]) -> str:
    return str(sorted(numbers))
```

约定：

- **参数以关键字传入**（内部执行 `fn(**kwargs)`），返回值会被转成字符串交给模型。
- 若签名里声明了 `workdir`，运行时会把当前工作目录注入进去，模型看不到这个参数：

  ```python
  def run(path: str, workdir: str) -> str:
      return f"{workdir}/{path}"
  ```

- `--target` 支持两种形式：

  | 形式 | 示例 | 解析方式 |
  |---|---|---|
  | 脚本路径 | `./ext/sort_numbers.py` | 相对 `--workdir` 解析 |
  | 模块名 | `mypkg.tools.hello` | 从 `sys.path` 导入 |

#### 2. 准备参数 schema（强烈建议）

不给 schema 时工具会得到一个空参数表，模型不知道传什么，等于不可用。

```json
{
  "type": "object",
  "properties": {
    "numbers": {
      "type": "array",
      "items": { "type": "integer" },
      "description": "待排序的整数列表"
    }
  },
  "required": ["numbers"]
}
```

#### 3. 装载

```powershell
orange tools add-ext /tool/demo/sort `
  --target ./ext/sort_numbers.py `
  --description "排序整数列表" `
  --caps read `
  --schema ./ext/schema.json `
  --workdir .
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `path`（位置参数） | 是 | 工具树路径，如 `/tool/demo/sort`；分组从路径推导 |
| `--target` | 是 | 脚本路径或模块名 |
| `--callable` | 否 | 函数名，默认 `run` |
| `--description` | 否 | 单行描述。模型靠它选择工具，建议必填 |
| `--caps` | 否 | 能力位，可重复，默认 `exec`；常用 `read` / `write` / `net` |
| `--schema` | 否 | 参数 schema 的 JSON 文件 |
| `--idempotent` | 否 | 标记为幂等 |
| `--group` | 否 | 省略时从 `path` 推导 |

装载后清单写入工作目录的 `capabilities.json`（无本地清单时写 `~/.orange/capabilities.json`）。

> **装载时即校验**：`add-ext` 会立刻导入模块并查找函数，文件不存在或找不到函数会当场报错，而不是等模型调用时才崩。
>
> **没有沙箱**：外置工具与主进程同权限运行，安全性取决于你对代码来源的信任。

### 二、MCP 服务器

```powershell
orange mcp add fs --command npx --arg=-y --arg "@modelcontextprotocol/server-filesystem" --arg ./ws
```

| 参数 | 说明 |
|---|---|
| `name`（位置参数） | 服务器名，同时作为 `/mcp/<name>` 分组名 |
| `--command` | 可执行文件，如 `npx`、`uvx`、`python` |
| `--arg` | 命令参数，可重复。**以 `--` 开头的值须写成 `--arg=--root`**，否则 argparse 会把它当成新选项 |
| `--env` | 环境变量 `K=V`，可重复 |
| `--args-file` | 从文件读取参数，每行一个；参数很多时比反复写 `--arg` 清晰 |

`mcp add` 会立即拉起服务器、握手并拉取工具清单，逐个挂到 `/mcp/<name>/` 下。上面这条命令实测输出：

```
已添加 MCP 服务器: fs
  发现 14 个工具，已装载 14 个到 /mcp/fs/
```

管理命令：

```powershell
orange mcp list     # 列出已声明的服务器与工具
orange mcp remove fs
```

### 三、查看装载结果

```powershell
orange tools list              # 已装载的全部工具
orange tools list --available  # 内置可用工具与预设
orange tools remove <path>     # 摘除某个工具
```

## 安装

需要 Python ≥ 3.12。

```powershell
uv tool install orange-0.1.0-py3-none-any.whl
```

装完后 `orange` 是一个独立命令，与项目目录无关：

```powershell
orange --help
```

从源码安装（开发用）：

```powershell
git clone <仓库地址> orange
cd orange
uv sync
uv run orange --help
```

## 快速开始

**1. 配置密钥**

```powershell
cd 你的工作目录
orange config init          # 生成 .env
```

编辑 `.env`，填入至少这三项：

```dotenv
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**2. 自检**

```powershell
orange service doctor
```

逐项检查密钥、模型连通性、工具树、存储与上下文容量，不消耗额度。

**3. 启动控制台**

```powershell
orange serve
```

打开 http://127.0.0.1:8012 ，在输入框发一条消息即可。界面包含会话列表、消息流（含流式输出）、下钻轨迹与设置面板。

## 命令一览

| 命令 | 用途 |
|---|---|
| `orange serve` | 启动 Web 控制台 |
| `orange service` | 服务生命周期与状态（`doctor` / `status` / `sweep`） |
| `orange config` | 超参数与配置（`show` / `get` / `check` / `env` / `init`） |
| `orange tools` | 工具装载与查看（`list` / `add` / `remove` / `add-ext`） |
| `orange skills` | skill 装载与查看 |
| `orange groups` | 工具树分组管理 |
| `orange mcp` | MCP 服务器管理（JSON-RPC over stdio） |
| `orange perms` | 工具权限管理 |

所有命令都接受 `--workdir` 指定工作目录、`--json` 输出机器可读结果。

## 配置

优先级：`~/.orange/settings.json` > 环境变量 / `.env` > 内置默认值。

Web 界面里的设置面板只暴露常用项（如 `loop.max_steps`、`context.capacity`），完整参数用 CLI：

```powershell
orange config show                      # 全部默认值
orange config get context.capacity      # 查单项
orange config check                     # 校验合法性
```

数据目录默认 `~/.orange`，可用环境变量 `ORANGE_HOME` 覆盖。工作目录下的 `WORK.md` 是跨会话共享的工作记忆。

## 项目结构

```
src/orange/          核心编排层（零 Web 框架依赖）
  loop.py            ReAct 循环
  context.py         上下文装配与压缩
  tool_tree.py       能力树与下钻
  executor.py        工具执行
  session_store.py   SQLite 持久化
  web/               前端静态资源
apps/server/         FastAPI + SSE 服务层
docs/                设计文档
tests/               测试
```

核心层禁止导入任何 Web 框架或编排框架，这条约束由 `scripts/check_constraints.py` 强制执行。

## 开发

```powershell
uv sync
uv run pytest -q                          # 794 项测试
uv run ruff check src/orange apps/server tests
uv run python scripts/check_constraints.py
uv build --wheel
```

## 文档

| 文档 | 内容 |
|---|---|
| `docs/01-系统架构设计.md` | 产品定位、设计原则、总体架构、关键机制 |
| `docs/02-数据结构设计/` | 逐模块数据结构定义 |
| `docs/10-CLI使用手册.md` | 命令行完整参考 |
| `docs/11-前端使用手册.md` | 控制台界面说明 |
| `docs/12-安装与使用指南.md` | 安装与故障排查 |

## 许可

未声明。
