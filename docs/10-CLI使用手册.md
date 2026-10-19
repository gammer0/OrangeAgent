# Orange CLI 使用手册

> 面向日常使用。命令都在项目根目录执行，前缀 `uv run python -m orange.cli`。
> 装好之后可以直接用短命令 `orange`（见 [§9 安装](#9-安装为命令行工具)）。

---

## 1. 五分钟上手

```bash
# 1. 生成配置模板，填入 API key
uv run python -m orange.cli config init
#    → 在当前目录写出 .env，编辑 DEEPSEEK_API_KEY

# 2. 校验配置是否可用（不会打印密钥）
uv run python -m orange.cli config check

# 3. 看一眼工具树，确认能力已装载
uv run python -m orange.cli service tree

# 4. 启动 Web 控制台
uv run uvicorn apps.server.main:app --port 8012 --host 127.0.0.1
#    → 浏览器打开 http://127.0.0.1:8012
```

**全局参数**（可放在子命令前后皆可）：

| 参数 | 作用 |
|---|---|
| `--workdir PATH` | 工作目录；`.env` 与工具根目录相对它解析，默认当前目录 |
| `--json` | 输出机器可读 JSON，便于脚本处理 |

---

## 2. `service` —— 服务状态与自检

### `service status`

各组件一句话状态：配置、能力清单、会话库、工具数。

```bash
uv run python -m orange.cli service status
uv run python -m orange.cli service status --json
```

### `service doctor`

逐项自检并给出修复建议。**出问题时先跑这个。**

检查项包括：`.env` 是否存在、密钥是否读出、能力清单能否解析、工具树是否为空、数据库能否打开。每项失败都附带回下一步怎么做。

```bash
uv run python -m orange.cli service doctor
```

### `service tree`

打印模型看到的工具树。`--depth` 控制展开层数。

```bash
uv run python -m orange.cli service tree
uv run python -m orange.cli service tree --depth 2
```

> **注意**：这里打印的是**核心层**默认深度（全展开）。Web 服务刻意只暴露 L1，让模型逐级下钻。

### `service sweep`

手动触发一次空闲会话回收。服务运行时通常由后台 sweeper 自动完成。

```bash
uv run python -m orange.cli service sweep
```

---

## 3. `config` —— 超参数

### `config show`

列出全部超参数的分组默认值。

```bash
uv run python -m orange.cli config show
uv run python -m orange.cli config show --with-env   # 附带模型配置（密钥已打码）
```

### `config get <key>`

查单个值。可以只写字段名，唯一时自动补全分组。

```bash
uv run python -m orange.cli config get loop.max_steps
uv run python -m orange.cli config get max_steps     # 同理
```

### `config check`

校验 `.env` 与数值合法性。密钥只显示首尾各 4 位。

```bash
uv run python -m orange.cli config check
```

### `config env`

列出所需环境变量与当前取值（密钥打码）。用来确认「我以为设了，其实没设」。

```bash
uv run python -m orange.cli config env
```

### `config init`

生成 `.env` 模板到当前工作目录。

```bash
uv run python -m orange.cli config init
```

> **修改超参数有两种方式**
>
> | 方式 | 适用 | 生效 |
> |---|---|---|
> | Web 设置界面 | 日常调参 | 多数立即生效，标「重启生效」的需重启 |
> | 环境变量 / `.env` | 部署固化 | 重启后生效 |
>
> Web 界面写入的是 `~/.orange/settings.json`，**优先级高于** `.env`。两边都设了值时以 `settings.json` 为准。

---

## 4. `tools` —— 内置与外来工具

```bash
uv run python -m orange.cli tools list
uv run python -m orange.cli tools add-ext /tool/mine/hello --target ./hello.py
uv run python -m orange.cli tools remove /tool/mine/hello
```

`add-ext` 后，工具立刻出现在 `service tree` 里，无需重启。

---

## 5. `skills` —— 技能包

```bash
uv run python -m orange.cli skills list
uv run python -m orange.cli skills add /skill/doc-writer --target ./doc-writer.md
```

技能是**只读说明文档**，不改工具树结构。模型读取内容作为使用指引。

---

## 6. `groups` —— 分组

分组由路径派生，只有写了描述的分组才会出现在索引里。

```bash
uv run python -m orange.cli groups list
uv run python -m orange.cli groups create tool/net --description "网络访问"
uv run python -m orange.cli groups remove tool/net           # 有成员时拒绝
uv run python -m orange.cli groups remove tool/net --force
```

---

## 7. `mcp` —— 外部 MCP 服务

```bash
uv run python -m orange.cli mcp list
uv run python -m orange.cli mcp add filesystem --command npx --arg -y --arg @modelcontextprotocol/server-filesystem
uv run python -m orange.cli mcp remove filesystem
```

装载后其工具出现在 `/mcp/<服务名>/...` 下，与其他能力共用同一棵树。

---

## 8. `perms` —— 权限

拒绝一个路径会**同时**影响查询与调用：树里看不见，调用抛 `PermissionDenied`。

```bash
uv run python -m orange.cli perms list
uv run python -m orange.cli perms deny /tool/net
uv run python -m orange.cli perms deny /tool/os/shell/run --agent web   # 只拒 Web 侧
uv run python -m orange.cli perms grant /tool/net
uv run python -m orange.cli perms show /tool/net
```

权限文件：`~/.orange/permissions.json`（项目内 `.agent/permissions.json` 若存在则优先）。

---

## 9. 安装为命令行工具

让 `orange` 成为全局命令，不再需要写 `uv run python -m orange.cli`：

```bash
uv tool install .
orange --help
```

升级：

```bash
uv tool install . --force
```

卸载：

```bash
uv tool uninstall orange
```

---

## 10. 常见问题

**`配置无效: 缺少 DEEPSEEK_API_KEY`**
`.env` 没生成，或键名写错。跑 `config init` 后编辑，或直接设环境变量。

**`service tree` 是空的**
能力清单未装载。先 `service doctor` 看 `capability_error`；清单文件在 `~/.orange/capabilities.json`（项目内 `.agent/capabilities.json` 优先）。

**改了超参数没生效**
Web 界面会明确提示哪些项需要重启。环境变量改了一律需要重启。

**`Unknown command`**
命令拼错，或子命令不存在。`uv run python -m orange.cli --help` 看全部子命令。

**想回退某次配置**
删掉 `~/.orange/settings.json` 中对应键，或在 Web 设置界面点「恢复」。
