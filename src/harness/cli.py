"""Harness 服务管理 CLI。

三个子命令组：

* ``harness service`` —— 服务生命周期与状态（``status`` / ``doctor`` / ``sweep`` / ``tree``）
* ``harness config``  —— 配置与超参数（``show`` / ``get`` / ``check`` / ``env``）
* ``harness config init`` —— 生成配置模板

设计约定：

* **只用标准库 ``argparse``**，不引入 click/typer 等额外依赖。
* **默认输出人类可读**；``--json`` 输出机器可读，便于脚本与 CI 消费。
* **不打印密钥**：任何路径都只输出 ``masked_key()``。
* 退出码：``0`` 成功 / ``1`` 运行时问题（如状态 degraded）/ ``2`` 参数错误。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from harness.capabilities import (
    BUILTIN_SKILLS,
    BUILTIN_TOOL_PATHS,
    CapabilityError,
    CapabilityManifest,
    GroupDef,
    SkillDef,
    SourceSpec,
    _build_external_tool,
    build_builtin_tool,
    group_of,
    load_manifest,
    save_manifest,
    write_skill_files,
)
from harness.config import ConfigError, load_llm_config
from harness.defaults import as_dict as defaults_as_dict
from harness.service import ServiceStatus, build_service
from harness.tools_builtin import TOOL_SPECS

__all__ = ["main", "build_parser"]

#: 生成模板的目标文件名。
ENV_TEMPLATE_NAME = ".env.example"

#: 模型类配置项（直接从 LLMConfig 读取，不进入 defaults）。
_MODEL_KEYS = (
    "base_url",
    "model",
    "temperature",
    "max_tokens",
    "timeout",
    "max_retries",
)


# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------


def _print_json(payload: Any) -> None:
    """以 UTF-8 友好的 JSON 输出。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _state_mark(state: str) -> str:
    """把状态映射为终端标记。"""
    return {
        "ready": "OK",
        "degraded": "WARN",
        "failed": "FAIL",
        "not_loaded": "SKIP",
    }.get(state, state.upper())


def _print_status(status: ServiceStatus, *, as_json: bool) -> None:
    """打印服务状态。"""
    if as_json:
        _print_json(status.to_dict())
        return

    print(f"工作目录: {status.workdir}")
    print(f"整体状态: {_state_mark(status.state)}")
    print()
    for layer in status.layers:
        print(f"[{_state_mark(layer.state):4}] {layer.name:<11} {layer.detail}")
        for key, value in layer.metrics.items():
            print(f"         {key}: {value}")
    if status.problems:
        print("\n问题:")
        for item in status.problems:
            print(f"  - {item}")


def _flatten_defaults() -> dict[str, Any]:
    """把分组默认值摊平为 ``group.field`` → value。"""
    flat: dict[str, Any] = {}
    for group, values in defaults_as_dict().items():
        for name, value in values.items():
            flat[f"{group}.{name}"] = value
    return flat


def _resolve_key(key: str, flat: dict[str, Any]) -> tuple[str, Any]:
    """解析 ``group.field`` 或裸字段名。

    Raises:
        KeyError: 键不存在或存在歧义（多个分组同名）。
    """
    if key in flat:
        return key, flat[key]
    matches = [name for name in flat if name.endswith(f".{key}")]
    if not matches:
        raise KeyError(key)
    if len(matches) > 1:
        raise KeyError(f"{key} 有歧义，候选: {', '.join(sorted(matches))}")
    return matches[0], flat[matches[0]]


def _env_template(dest: Path) -> None:
    """生成 ``.env.example``：优先复制仓库内模板，否则写入内置内容。"""
    bundled = Path(__file__).resolve().parents[2] / ENV_TEMPLATE_NAME
    if bundled.is_file() and bundled.resolve() != dest.resolve():
        shutil.copyfile(bundled, dest)
        return
    dest.write_text(_BUILTIN_ENV_TEMPLATE, encoding="utf-8")


_BUILTIN_ENV_TEMPLATE = """\
# Harness 配置模板 —— 复制为 .env 后填入真实值
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2
"""


# ---------------------------------------------------------------------------
# service 子命令
# ---------------------------------------------------------------------------


def _cmd_service_status(args: argparse.Namespace) -> int:
    """``harness service status``：打印各组件状态。"""
    service = build_service(args.workdir, capacity=args.capacity, load_model=not args.no_model)
    try:
        status = service.status()
        _print_status(status, as_json=args.json)
        return 0 if status.ok or args.allow_degraded else 1
    finally:
        service.close()


def _cmd_service_doctor(args: argparse.Namespace) -> int:
    """``harness service doctor``：逐项自检并给出修复建议。"""
    service = build_service(args.workdir, capacity=args.capacity, load_model=True)
    checks: list[dict[str, Any]] = []
    try:
        status = service.status()

        env_file = Path(args.workdir).resolve() / ".env"
        checks.append(
            {
                "item": ".env 存在",
                "ok": env_file.is_file(),
                "hint": "" if env_file.is_file() else "运行 `harness config init` 生成模板",
            }
        )
        checks.append(
            {
                "item": "API Key 可用",
                "ok": service.config is not None,
                "hint": ""
                if service.config is not None
                else "在 .env 或进程环境设置 DEEPSEEK_API_KEY",
            }
        )
        checks.append(
            {
                "item": "工具树可构建",
                "ok": status.layer("capability").state == "ready"
                if status.layer("capability")
                else False,
                "hint": ""
                if status.layer("capability") and status.layer("capability").state == "ready"
                else "检查工具注册是否存在重名或非法前缀",
            }
        )
        store_ok = bool(
            status.layer("store")
            and status.layer("store").state == "ready"
            and status.layer("store").metrics.get("journal_mode") == "wal"
        )
        checks.append(
            {
                "item": "SQLite 可写（WAL）",
                "ok": store_ok,
                "hint": "" if store_ok else "确认工作目录可写",
            }
        )
        checks.append(
            {
                "item": "上下文容量",
                "ok": service.capacity > 0,
                "hint": "" if service.capacity > 0 else "容量必须为正数且与模型上下文一致",
            }
        )
        compress_ok = bool(
            service.assembler is not None and service.assembler.summarizer is not None
        )
        checks.append(
            {
                "item": "C3 压缩可用",
                "ok": compress_ok,
                "hint": ""
                if compress_ok
                else "未注入 summarizer 时压缩静默跳过（I4 无损退化，非错误）",
            }
        )

        if args.json:
            _print_json({"status": status.to_dict(), "checks": checks})
        else:
            print("自检项:")
            for check in checks:
                mark = "OK  " if check["ok"] else "WARN"
                print(f"  [{mark}] {check['item']}")
                if check["hint"]:
                    print(f"         → {check['hint']}")
            print()
            _print_status(status, as_json=False)

        # 只有"硬失败"才返回 1；压缩未启用属于提示级
        hard_failures = [
            c for c in checks if not c["ok"] and c["item"] in {"API Key 可用", "工具树可构建"}
        ]
        return 1 if hard_failures else 0
    finally:
        service.close()


def _cmd_service_sweep(args: argparse.Namespace) -> int:
    """``harness service sweep``：驱逐空闲会话。"""
    service = build_service(args.workdir, load_model=False)
    try:
        evicted = service.sweep()
        if args.json:
            _print_json({"evicted": evicted, "count": len(evicted)})
        else:
            print(f"已驱逐 {len(evicted)} 个会话")
            for session_id in evicted:
                print(f"  - {session_id}")
        return 0
    finally:
        service.close()


def _cmd_service_tree(args: argparse.Namespace) -> int:
    """``harness service tree``：打印工具树索引。"""
    service = build_service(args.workdir, load_model=False)
    try:
        if service.tree is None:
            print("工具树不可用：" + "; ".join(service.problems), file=sys.stderr)
            return 1
        if args.json:
            _print_json(
                {
                    "paths": service.tree.paths,
                    "version": service.tree.version,
                }
            )
        else:
            index = service.tree.tree_index(depth=args.depth)
            print(index if index else "（工具树为空）")
        return 0
    finally:
        service.close()


# ---------------------------------------------------------------------------
# config 子命令
# ---------------------------------------------------------------------------


def _cmd_config_show(args: argparse.Namespace) -> int:
    """``harness config show``：展示全部默认值（可选合并模型配置）。"""
    groups = defaults_as_dict()
    payload: dict[str, Any] = {"defaults": groups}

    if args.with_env:
        try:
            config = load_llm_config(dotenv_path=Path(args.workdir).resolve() / ".env")
            payload["model"] = {
                "base_url": config.base_url,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "api_key": config.masked_key(),
            }
        except ConfigError as exc:
            payload["model_error"] = str(exc)

    if args.json:
        _print_json(payload)
        return 0

    for group, values in groups.items():
        print(f"[{group}]")
        for name, value in values.items():
            print(f"  {name} = {value}")
        print()

    if "model" in payload:
        print("[model] （来自环境变量）")
        for name, value in payload["model"].items():
            print(f"  {name} = {value}")
    elif "model_error" in payload:
        print(f"[model] 未加载: {payload['model_error']}")
    return 0


def _cmd_config_get(args: argparse.Namespace) -> int:
    """``harness config get <key>``：查询单个超参数。"""
    flat = _flatten_defaults()
    try:
        resolved, value = _resolve_key(args.key, flat)
    except KeyError as exc:
        message = exc.args[0] if exc.args else args.key
        print(f"未知配置项: {message}", file=sys.stderr)
        return 2

    if args.json:
        _print_json({"key": resolved, "value": value})
    else:
        print(f"{resolved} = {value}")
    return 0


def _cmd_config_check(args: argparse.Namespace) -> int:
    """``harness config check``：校验环境变量与数值合法性。"""
    try:
        config = load_llm_config(dotenv_path=Path(args.workdir).resolve() / ".env")
    except ConfigError as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            print(f"配置无效: {exc}", file=sys.stderr)
            print("提示: 运行 `harness config init` 生成模板", file=sys.stderr)
        return 1

    payload = {
        "ok": True,
        "base_url": config.base_url,
        "model": config.model,
        "api_key": config.masked_key(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    if args.json:
        _print_json(payload)
    else:
        print("配置有效")
        for name, value in payload.items():
            if name != "ok":
                print(f"  {name} = {value}")
    return 0


def _cmd_config_env(args: argparse.Namespace) -> int:
    """``harness config env``：列出所需环境变量与当前取值（密钥掩码）。"""
    rows = [
        ("DEEPSEEK_API_KEY", "必填，密钥"),
        ("DEEPSEEK_BASE_URL", "端点基址"),
        ("DEEPSEEK_MODEL", "模型名"),
        ("LLM_TEMPERATURE", "采样温度"),
        ("LLM_MAX_TOKENS", "输出上限"),
        ("LLM_TIMEOUT", "HTTP 超时（秒）"),
        ("LLM_MAX_RETRIES", "重试次数"),
    ]
    import os

    payload = []
    for name, desc in rows:
        raw = os.getenv(name)
        if name == "DEEPSEEK_API_KEY" and raw:
            shown = f"{raw[:4]}...{raw[-4:]}" if len(raw) > 8 else "***"
        else:
            shown = raw if raw is not None else "(未设置)"
        payload.append({"name": name, "value": shown, "description": desc})

    if args.json:
        _print_json({"variables": payload})
        return 0

    print(f"{'变量':<22}{'当前值':<28}说明")
    for row in payload:
        print(f"{row['name']:<22}{row['value']:<28}{row['description']}")
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    """``harness config init``：生成 ``.env`` 模板。"""
    target = Path(args.workdir).resolve() / args.output
    if target.exists() and not args.force:
        print(f"目标已存在: {target}（使用 --force 覆盖）", file=sys.stderr)
        return 2

    _env_template(target)
    print(f"已生成: {target}")
    print("下一步：填入真实 DEEPSEEK_API_KEY，然后运行 `harness config check`")
    return 0


# ---------------------------------------------------------------------------
# tools / skills 子命令
# ---------------------------------------------------------------------------


#: 预设分组：``harness tools add --preset <名>``。
TOOL_PRESETS: dict[str, tuple[str, ...]] = {
    "core": ("/tool/os/file/read", "/tool/os/file/list", "/tool/os/file/write", "/tool/os/echo"),
    "coding": (
        "/tool/os/file/read",
        "/tool/os/file/list",
        "/tool/os/file/write",
        "/tool/os/file/append",
        "/tool/code/py/check",
        "/tool/code/py/run_brief",
        "/tool/os/shell/run",
    ),
    "game": (
        "/tool/game/snake/check",
        "/tool/game/snake/demo",
        "/tool/code/py/check",
        "/tool/code/py/run_brief",
    ),
    "mcp-filesystem": ("/mcp/fs/list", "/mcp/fs/read", "/mcp/fs/write"),
    "all": tuple(BUILTIN_TOOL_PATHS),
}

#: 预设名称 → 说明（保持与 ``TOOL_PRESETS`` 同步）。
PRESET_HINTS: dict[str, str] = {
    "core": "最小文件读写（4 个）",
    "coding": "编码闭环：读写 + 语法检查 + 短运行 + shell",
    "game": "游戏验收：结构检查 + 无头自推演（依赖 coding 的 py 工具）",
    "mcp-filesystem": "MCP filesystem 命名空间（3 个）",
    "all": "全部内置工具",
}


def _load_or_empty(workdir: str) -> CapabilityManifest:
    """读取清单；不存在时返回空清单（供 ``add`` 首次使用）。"""
    try:
        return load_manifest(workdir)
    except CapabilityError as exc:
        if "不存在" in str(exc):
            return CapabilityManifest()
        raise


def _save_and_report(manifest: CapabilityManifest, workdir: str, *, as_json: bool) -> None:
    """保存清单并输出摘要。"""
    target = save_manifest(manifest, workdir)
    if as_json:
        _print_json({**manifest.to_dict(), "file": str(target)})
        return
    print(f"已写入: {target}")
    print(f"工具 {len(manifest.tools)} 个，skill {len(manifest.skills)} 个")


def _print_capabilities(manifest: CapabilityManifest, *, as_json: bool) -> None:
    """打印当前清单（按分组聚合）。"""
    if as_json:
        _print_json(manifest.to_dict())
        return
    if not manifest.tools and not manifest.skills:
        print("（空清单）使用 `harness tools add --preset all` 装载工具")
        return

    buckets: dict[str, list[str]] = {}
    for tool in sorted(manifest.tools, key=lambda t: t.path):
        marks = f"caps={','.join(tool.caps) or '-'} {tool.concurrency}"
        if tool.idempotent:
            marks += " idem"
        if tool.path in manifest.sources:
            marks += " [ext]"
        buckets.setdefault(group_of(tool.path) or "(root)", []).append(
            f"  {tool.path:<30} {marks:<28} {tool.description}"
        )
    for skill in sorted(manifest.skills, key=lambda s: s.path):
        buckets.setdefault(group_of(skill.path) or "(root)", []).append(
            f"  {skill.path:<30} source={skill.source:<20} {skill.description}"
        )

    for group_id in sorted(buckets):
        meta = manifest.get_group(group_id)
        label = f" — {meta.description}" if meta and meta.description else ""
        print(f"[{group_id}]{label}")
        for line in buckets[group_id]:
            print(line)


# ---------------------------------------------------------------------------
# 分组命令
# ---------------------------------------------------------------------------


def _cmd_groups_list(args: argparse.Namespace) -> int:
    """``harness groups list``：列出分组及其成员数。"""
    manifest = _load_or_empty(args.workdir)
    rows = []
    for group_id in manifest.group_ids():
        prefix = f"/{group_id}/"
        members = [t.path for t in manifest.tools if t.path.startswith(prefix)]
        members += [s.path for s in manifest.skills if s.path.startswith(prefix)]
        meta = manifest.get_group(group_id)
        rows.append(
            {
                "group_id": group_id,
                "path": f"/{group_id}",
                "members": len(members),
                "description": meta.description if meta else "",
                "declared": meta is not None,
            }
        )
    if args.json:
        _print_json({"groups": rows})
        return 0
    if not rows:
        print("（无分组）使用 `harness groups create tool/<名> --description ...` 新建")
        return 0
    for row in rows:
        flag = "" if row["declared"] else " (派生)"
        print(
            f"{row['path']:<28} 成员 {row['members']:>2} 个{flag:<8} {row['description']}"
        )
    return 0


def _cmd_groups_create(args: argparse.Namespace) -> int:
    """``harness groups create``：声明一个分组（工具树上作为中间节点出现）。"""
    manifest = _load_or_empty(args.workdir)
    try:
        group = GroupDef(group_id=args.group_id, description=args.description or "")
        manifest.add_group(group)
    except CapabilityError as exc:
        print(f"无法创建分组: {exc}", file=sys.stderr)
        return 2
    target = save_manifest(manifest, args.workdir)
    if args.json:
        _print_json({"created": group.group_id, "path": group.path, "file": str(target)})
        return 0
    print(f"已创建分组: {group.path}")
    print(f"已写入: {target}")
    return 0


def _cmd_groups_remove(args: argparse.Namespace) -> int:
    """``harness groups remove``：删除分组声明（不影响成员，除非先移除）。"""
    manifest = _load_or_empty(args.workdir)
    try:
        removed = manifest.remove_group(args.group_id, force=args.force)
    except CapabilityError as exc:
        print(f"无法删除分组: {exc}（加 --force 可逆）", file=sys.stderr)
        return 2
    if not removed:
        print(f"没有该分组: {args.group_id}", file=sys.stderr)
        return 1
    target = save_manifest(manifest, args.workdir)
    if args.json:
        _print_json({"removed": args.group_id, "file": str(target)})
        return 0
    print(f"已删除分组声明: {args.group_id}")
    return 0


# ---------------------------------------------------------------------------
# 外来工具命令
# ---------------------------------------------------------------------------


def _cmd_tools_add_ext(args: argparse.Namespace) -> int:
    """``harness tools add-ext``：装载外来工具（执行本体代码，供应链负责）。"""
    path = args.path
    if not path.startswith("/"):
        head = args.group.replace("/", "") or "tool"
        path = f"/{head}/{path.lstrip('/')}"
    try:
        source = SourceSpec(
            kind=args.kind, target=args.target, callable=args.callable
        )
    except CapabilityError as exc:
        print(f"来源声明非法: {exc}", file=sys.stderr)
        return 2

    manifest = _load_or_empty(args.workdir)
    schema: dict = {"type": "object", "properties": {}, "required": []}
    if args.schema:
        try:
            schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"无法读取 schema 文件: {exc}", file=sys.stderr)
            return 2

    entry = {
        "path": path,
        "description": args.description or f"外来工具 {path}",
        "caps": args.caps or ["exec"],
        "idempotent": bool(args.idempotent),
        "schema": schema,
        "source": source.to_dict(),
    }
    try:
        tool = _build_external_tool(path, entry, source, args.workdir)
        if manifest.has_path(path):
            manifest.remove(path)
        manifest.add_tool(tool)
        manifest.sources[path] = source
    except CapabilityError as exc:
        print(f"无法装载外来工具: {exc}", file=sys.stderr)
        return 2

    target = save_manifest(manifest, args.workdir)
    if args.json:
        _print_json({"added": path, "source": source.to_dict(), "file": str(target)})
        return 0
    print(f"已装载外来工具: {path}")
    print(f"  来源: {source.kind}:{source.target}:{source.callable}")
    print(f"  分组: /{group_of(path)}")
    print(f"已写入: {target}")
    return 0


# ---------------------------------------------------------------------------
# MCP 命令
# ---------------------------------------------------------------------------


def _cmd_mcp_add(args: argparse.Namespace) -> int:
    """``harness mcp add``：声明 MCP 服务器并拉取工具清单。"""
    from harness.mcp_client import McpError, McpServerConfig, list_tools

    workdir = args.workdir
    config = McpServerConfig(
        name=args.name,
        command=args.command,
        args=tuple(args.arg or ()),
        env=tuple(
            tuple(item.split("=", 1)) for item in (args.env or []) if "=" in item
        ),
    )

    try:
        discovered = list_tools(config, workdir)
    except McpError as exc:
        print(f"无法连接 MCP 服务器: {exc}", file=sys.stderr)
        return 2

    manifest = _load_or_empty(workdir)
    manifest.mcp_servers = [
        s for s in manifest.mcp_servers if s.get("name") != config.name
    ]
    manifest.mcp_servers.append(config.to_dict())

    added = 0
    for remote in discovered:
        tool_name = remote.get("name", "")
        if not tool_name:
            continue
        path = f"/mcp/{config.name}/{tool_name}"
        entry = {
            "path": path,
            "description": (remote.get("description") or f"MCP {config.name}/{tool_name}")
            .split("\n")[0],
            "caps": ["exec"],
            "idempotent": False,
            "schema": remote.get("inputSchema") or {"type": "object", "properties": {}},
            "source": {"kind": "mcp", "target": config.name, "callable": tool_name},
            "remote_name": tool_name,
        }
        try:
            tool = _build_external_tool(
                path, entry, SourceSpec(kind="mcp", target=config.name, callable=tool_name), workdir
            )
            if manifest.has_path(path):
                manifest.remove(path)
            manifest.add_tool(tool)
            manifest.sources[path] = SourceSpec(
                kind="mcp", target=config.name, callable=tool_name
            )
            added += 1
        except CapabilityError as exc:
            print(f"跳过 {path}: {exc}", file=sys.stderr)

    target = save_manifest(manifest, workdir)
    if args.json:
        _print_json(
            {
                "server": config.name,
                "tools": [t.get("name") for t in discovered],
                "added": added,
                "file": str(target),
            }
        )
        return 0
    print(f"已添加 MCP 服务器: {config.name}")
    print(f"  发现 {len(discovered)} 个工具，已装载 {added} 个到 /mcp/{config.name}/")
    return 0


def _cmd_mcp_list(args: argparse.Namespace) -> int:
    """``harness mcp list``：列出 MCP 服务器及其工具数。"""
    manifest = _load_or_empty(args.workdir)
    rows = []
    for server in manifest.mcp_servers:
        name = server.get("name", "")
        prefix = f"/mcp/{name}/"
        tools = [t.path for t in manifest.tools if t.path.startswith(prefix)]
        rows.append(
            {
                "name": name,
                "command": server.get("command", ""),
                "args": server.get("args") or [],
                "tools": len(tools),
            }
        )
    if args.json:
        _print_json({"servers": rows})
        return 0
    if not rows:
        print("（无 MCP 服务器）使用 `harness mcp add <名> --command <cmd>` 添加")
        return 0
    for row in rows:
        cmd = " ".join([row["command"], *row["args"]])
        print(f"/mcp/{row['name']:<22} 工具 {row['tools']:>2} 个  {cmd}")
    return 0


def _cmd_mcp_remove(args: argparse.Namespace) -> int:
    """``harness mcp remove``：删除服务器声明及其工具。"""
    manifest = _load_or_empty(args.workdir)
    prefix = f"/mcp/{args.name}/"
    before = len(manifest.tools)
    manifest.tools = [t for t in manifest.tools if not t.path.startswith(prefix)]
    removed_tools = before - len(manifest.tools)
    before_servers = len(manifest.mcp_servers)
    manifest.mcp_servers = [
        s for s in manifest.mcp_servers if s.get("name") != args.name
    ]
    if len(manifest.mcp_servers) == before_servers:
        print(f"没有该 MCP 服务器: {args.name}", file=sys.stderr)
        return 1
    target = save_manifest(manifest, args.workdir)
    if args.json:
        _print_json({"removed": args.name, "tools": removed_tools, "file": str(target)})
        return 0
    print(f"已删除 MCP 服务器 {args.name}（连带 {removed_tools} 个工具）")
    return 0


# ---------------------------------------------------------------------------
# 权限命令
# ---------------------------------------------------------------------------


def _cmd_perms_list(args: argparse.Namespace) -> int:
    """``harness perms list``：列出拒绝记录。"""
    from harness.permission_store import list_grants

    data = list_grants(args.workdir)
    if args.agent:
        data = {k: v for k, v in data.items() if k == args.agent}
    if args.json:
        _print_json({"agents": data})
        return 0
    if not data:
        print("（无拒绝记录）所有路径默认允许（B15）")
        return 0
    for agent_id, grants in sorted(data.items()):
        denied = [p for p, allowed in grants.items() if not allowed]
        print(f"[{agent_id}] 拒绝 {len(denied)} 条")
        for path in sorted(denied):
            print(f"  {path}")
    return 0


def _cmd_perms_deny(args: argparse.Namespace) -> int:
    """``harness perms deny``：拒绝某路径（含其子树）的查询与调用。"""
    from harness.permission_store import set_grant

    workdir = args.workdir
    agents = args.agent or ["local", "web"]
    done = []
    for agent_id in agents:
        try:
            set_grant(workdir, args.path, allowed=False, agent_id=agent_id)
            done.append(agent_id)
        except ValueError as exc:
            print(f"{agent_id}: {exc}", file=sys.stderr)
            return 2
    if args.json:
        _print_json({"denied": args.path, "agents": done})
        return 0
    print(f"已拒绝: {args.path}（agent: {', '.join(done)}）")
    print("该路径及其子树将不再出现在工具树索引中，调用返回 3a PermissionDenied")
    return 0


def _cmd_perms_grant(args: argparse.Namespace) -> int:
    """``harness perms grant``：恢复允许（删除例外）。"""
    from harness.permission_store import set_grant

    workdir = args.workdir
    agents = args.agent or ["local", "web"]
    done = []
    for agent_id in agents:
        try:
            set_grant(workdir, args.path, allowed=True, agent_id=agent_id)
            done.append(agent_id)
        except ValueError as exc:
            print(f"{agent_id}: {exc}", file=sys.stderr)
            return 2
    if args.json:
        _print_json({"granted": args.path, "agents": done})
        return 0
    print(f"已恢复允许: {args.path}（agent: {', '.join(done)}）")
    return 0


def _cmd_perms_show(args: argparse.Namespace) -> int:
    """``harness perms show``：查某路径的实际判定结果。"""
    from harness.permission import AgentPermissions, visible
    from harness.permission_store import load_permissions

    workdir = args.workdir
    agents = args.agent or ["local", "web"]
    rows = []
    for agent_id in agents:
        perms = load_permissions(workdir, agent_id)
        rows.append(
            {
                "agent": agent_id,
                "allowed": visible(args.path, perms),
                "explicit": perms.grants.get(args.path),
            }
        )
    if args.json:
        _print_json({"path": args.path, "results": rows})
        return 0
    print(f"路径: {args.path}")
    for row in rows:
        state = "允许" if row["allowed"] else "拒绝"
        explicit = row["explicit"]
        reason = "显式拒绝" if explicit is False else "默认允许（B15）"
        if explicit is None and not row["allowed"]:
            reason = "祖先被拒，整棵子树隐藏（B5）"
        print(f"  [{row['agent']}] {state} — {reason}")
    return 0


def _cmd_tools_list(args: argparse.Namespace) -> int:
    """``harness tools list``：列出已装载与可用工具。"""
    manifest = _load_or_empty(args.workdir)
    if args.available:
        payload = {
            "available": [
                {
                    "path": spec["path"],
                    "description": spec["description"],
                    "caps": list(spec["caps"]),
                }
                for _, spec in sorted(TOOL_SPECS.items())
            ],
            "presets": {name: list(paths) for name, paths in TOOL_PRESETS.items()},
        }
        if args.json:
            _print_json(payload)
            return 0
        print("[可用内置工具]")
        for item in payload["available"]:
            loaded = " *" if manifest.has_path(item["path"]) else ""
            print(f"  {item['path']:<28} {item['description']}{loaded}")
        print("\n[预设]")
        for name, paths in TOOL_PRESETS.items():
            print(f"  {name:<16} {PRESET_HINTS.get(name, '')} ({len(paths)} 个)")
        print("\n（* 表示已装载）")
        return 0

    _print_capabilities(manifest, as_json=args.json)
    return 0


def _cmd_tools_add(args: argparse.Namespace) -> int:
    """``harness tools add``：按路径或预设追加工具。"""
    paths: list[str] = []
    for group in args.preset or []:
        if group not in TOOL_PRESETS:
            print(
                f"未知预设: {group}（可用: {', '.join(sorted(TOOL_PRESETS))}）",
                file=sys.stderr,
            )
            return 2
        paths.extend(TOOL_PRESETS[group])
    paths.extend(args.paths or [])
    if not paths:
        print("至少给出一个路径或 --preset", file=sys.stderr)
        return 2

    manifest = _load_or_empty(args.workdir)
    added, skipped = [], []
    for path in dict.fromkeys(paths):
        if path not in TOOL_SPECS:
            print(f"未知工具路径: {path}", file=sys.stderr)
            return 2
        if manifest.has_path(path):
            skipped.append(path)
            continue
        manifest.add_tool(build_builtin_tool(path, args.workdir))
        added.append(path)

    if added:
        _save_and_report(manifest, args.workdir, as_json=args.json)
    elif args.json:
        _print_json({"added": [], "skipped": skipped})
    if skipped and not args.json:
        print(f"已存在，跳过 {len(skipped)} 个: {', '.join(skipped)}")
    if not added and not skipped and not args.json:
        print("没有变化")
    return 0


def _cmd_tools_remove(args: argparse.Namespace) -> int:
    """``harness tools remove``：按路径移除工具。"""
    manifest = _load_or_empty(args.workdir)
    removed = [path for path in args.paths if manifest.remove(path)]
    if not removed:
        print("没有匹配的条目", file=sys.stderr)
        return 1
    _save_and_report(manifest, args.workdir, as_json=args.json)
    return 0


def _cmd_skills_list(args: argparse.Namespace) -> int:
    """``harness skills list``：列出已装载 skill 及目录状态。"""
    manifest = _load_or_empty(args.workdir)
    rows = []
    for skill in sorted(manifest.skills, key=lambda s: s.name):
        target = Path(args.workdir).resolve() / skill.skill_md
        rows.append(
            {
                "name": skill.name,
                "path": skill.path,
                "source": skill.source,
                "dir": skill.dir,
                "skill_md": target.is_file(),
                "description": skill.description,
            }
        )
    if args.json:
        _print_json({"skills": rows})
        return 0
    if not rows:
        print("（没有 skill）使用 `harness skills add <来源> --name <名>` 装载")
        return 0
    for row in rows:
        flag = "OK" if row["skill_md"] else "MISSING SKILL.md"
        print(f"{row['path']:<28} {flag:<18} source={row['source']}")
        print(f"  {row['description']}")
    return 0


def _cmd_skills_add(args: argparse.Namespace) -> int:
    """``harness skills add``：从内置模板 / 本地文件 / URL 装载 skill。"""
    workdir = args.workdir
    files: dict[str, str] = {}
    source = args.source

    if source.startswith(("http://", "https://")):
        text, files = _fetch_skill_source(source)
    elif source in BUILTIN_SKILLS:
        text = BUILTIN_SKILLS[source][0]
    else:
        local = Path(source)
        if not local.is_file():
            print(f"找不到来源: {source}", file=sys.stderr)
            print(
                f"可用内置模板: {', '.join(sorted(BUILTIN_SKILLS)) or '（无）'}",
                file=sys.stderr,
            )
            return 2
        text = local.read_text(encoding="utf-8", errors="replace")

    description = args.description or _frontmatter_value(text, "description") or ""
    if not description:
        print("无法确定 description：用 --description 显式给出", file=sys.stderr)
        return 2
    name = args.name or _frontmatter_value(text, "name") or ""
    if not name:
        print("无法确定 skill 名：用 --name 显式给出", file=sys.stderr)
        return 2

    manifest = _load_or_empty(workdir)
    skill = SkillDef(name=name, description=description, source=source)
    reload = args.force
    if manifest.get_skill(name):
        if not reload:
            print(f"skill 已存在: {name}（使用 --force 覆盖内容）", file=sys.stderr)
            return 2
        manifest.skills = [s for s in manifest.skills if s.name != name]
    manifest.add_skill(skill)

    payload = {"SKILL.md": _strip_to_skill(text)}
    for filename, content in files.items():
        payload[filename] = content
    target = write_skill_files(workdir, name, payload)
    _save_and_report(manifest, workdir, as_json=args.json)
    if not args.json:
        print(f"skill 文件: {target / 'SKILL.md'}")
    return 0


def _cmd_skills_remove(args: argparse.Namespace) -> int:
    """``harness skills remove``：从清单摘除 skill（不删目录）。"""
    manifest = _load_or_empty(args.workdir)
    if not manifest.remove(f"/skill/{args.name}"):
        print(f"没有该 skill: {args.name}", file=sys.stderr)
        return 1
    _save_and_report(manifest, args.workdir, as_json=args.json)
    return 0


def _cmd_skills_show(args: argparse.Namespace) -> int:
    """``harness skills show``：打印 skill 文件内容。"""
    manifest = _load_or_empty(args.workdir)
    skill = manifest.get_skill(args.name)
    if skill is None:
        print(f"没有该 skill: {args.name}", file=sys.stderr)
        return 1
    target = Path(args.workdir).resolve() / skill.dir / args.file
    if not target.is_file():
        print(f"文件不存在: {target}", file=sys.stderr)
        return 1
    print(target.read_text(encoding="utf-8", errors="replace"))
    return 0


def _frontmatter_value(text: str, key: str) -> str:
    """从 ``SKILL.md`` 的 YAML frontmatter 读取单个标量值。"""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else text[3:]
    for line in block.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        if name.strip() == key:
            return value.strip().strip("\"'")
    return ""


def _strip_to_skill(text: str) -> str:
    """保证落盘内容以 frontmatter 开头（去 BOM / 前导空行）。"""
    return text.lstrip("\ufeff").lstrip()


def _fetch_skill_source(url: str) -> tuple[str, dict[str, str]]:
    """下载 skill 来源；``*.md`` 单文件，其余视作 zip 包（仅取顶层 md）。"""
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - 用户显式给出的 URL
        raw = response.read()

    if not url.lower().endswith(".zip"):
        return raw.decode("utf-8", errors="replace"), {}

    files: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            name = Path(info.filename).name
            if not name.endswith(".md"):
                continue
            with archive.open(info) as handle:
                files[name] = handle.read().decode("utf-8", errors="replace")
    skill_md = files.pop("SKILL.md", "")
    return skill_md, files


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """为子命令补充常用参数，使其既能放在子命令前也能放在其后。

    ``argparse`` 的父级选项必须在子命令之前给出；这里在每个子命令上再声明一次，
    冲突由 ``argparse.SUPPRESS`` 默认值处理——子命令未给出时不会覆盖全局值。
    """
    parser.add_argument(
        "--workdir",
        default=argparse.SUPPRESS,
        help="工作目录（也可放在子命令之前）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="输出 JSON（也可放在子命令之前）",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器。"""
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Harness 服务管理 / 超参数配置 / 配置加载",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="工作目录（.env 与 .agent/ 相对它解析），默认当前目录",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- service ----
    service = sub.add_parser("service", help="服务管理与状态")
    _add_common_options(service)
    service_sub = service.add_subparsers(dest="action", required=True)

    status = service_sub.add_parser("status", help="打印各组件状态")
    _add_common_options(status)
    status.add_argument("--capacity", type=int, default=None, help="覆盖上下文容量")
    status.add_argument(
        "--no-model", action="store_true", help="跳过模型通道（不读密钥、不发请求）"
    )
    status.add_argument(
        "--allow-degraded", action="store_true", help="degraded 也返回 0（默认返回 1）"
    )
    status.set_defaults(func=_cmd_service_status)

    doctor = service_sub.add_parser("doctor", help="逐项自检并给出修复建议")
    _add_common_options(doctor)
    doctor.add_argument("--capacity", type=int, default=None, help="覆盖上下文容量")
    doctor.set_defaults(func=_cmd_service_doctor)

    sweep = service_sub.add_parser("sweep", help="驱逐空闲会话")
    _add_common_options(sweep)
    sweep.set_defaults(func=_cmd_service_sweep)

    tree = service_sub.add_parser("tree", help="打印工具树索引")
    _add_common_options(tree)
    tree.add_argument(
        "--depth",
        type=int,
        default=0,
        help="索引深度；0 表示全展开（默认），1 只列 L1",
    )
    tree.set_defaults(func=_cmd_service_tree)

    # ---- config ----
    config = sub.add_parser("config", help="超参数与配置加载")
    _add_common_options(config)
    config_sub = config.add_subparsers(dest="action", required=True)

    show = config_sub.add_parser("show", help="展示全部默认超参数")
    _add_common_options(show)
    show.add_argument(
        "--with-env", action="store_true", help="同时展示环境变量解析出的模型配置"
    )
    show.set_defaults(func=_cmd_config_show)

    get = config_sub.add_parser("get", help="查询单个超参数（如 context.capacity）")
    _add_common_options(get)
    get.add_argument("key", help="配置项名，支持 group.field 或裸字段名")
    get.set_defaults(func=_cmd_config_get)

    check = config_sub.add_parser("check", help="校验环境变量与数值合法性")
    _add_common_options(check)
    check.set_defaults(func=_cmd_config_check)

    env = config_sub.add_parser("env", help="列出所需环境变量与当前取值")
    _add_common_options(env)
    env.set_defaults(func=_cmd_config_env)

    init = config_sub.add_parser("init", help="生成 .env 模板")
    _add_common_options(init)
    init.add_argument("--output", default=".env.example", help="输出文件名")
    init.add_argument("--force", action="store_true", help="覆盖已存在文件")
    init.set_defaults(func=_cmd_config_init)

    # ---- tools ----
    tools = sub.add_parser("tools", help="工具装载与查看")
    _add_common_options(tools)
    tools_sub = tools.add_subparsers(dest="action", required=True)

    tools_list = tools_sub.add_parser("list", help="列出已装载工具")
    _add_common_options(tools_list)
    tools_list.add_argument("--available", action="store_true", help="同时列出可用与预设")
    tools_list.set_defaults(func=_cmd_tools_list)

    tools_add = tools_sub.add_parser("add", help="按路径或预设追加工具")
    _add_common_options(tools_add)
    tools_add.add_argument("paths", nargs="*", help="工具路径，如 /tool/code/py/check")
    tools_add.add_argument(
        "--preset",
        action="append",
        choices=sorted(TOOL_PRESETS),
        help="按预设批量装载（可重复）",
    )
    tools_add.set_defaults(func=_cmd_tools_add)

    tools_remove = tools_sub.add_parser("remove", help="按路径移除工具")
    _add_common_options(tools_remove)
    tools_remove.add_argument("paths", nargs="+", help="工具路径")
    tools_remove.set_defaults(func=_cmd_tools_remove)

    # ---- skills ----
    skills = sub.add_parser("skills", help="skill 装载与查看")
    _add_common_options(skills)
    skills_sub = skills.add_subparsers(dest="action", required=True)

    skills_list = skills_sub.add_parser("list", help="列出已装载 skill")
    _add_common_options(skills_list)
    skills_list.set_defaults(func=_cmd_skills_list)

    skills_add = skills_sub.add_parser("add", help="从内置模板/文件/URL 装载 skill")
    _add_common_options(skills_add)
    skills_add.add_argument("source", help="内置模板名 / SKILL.md 路径 / http(s) URL")
    skills_add.add_argument("--name", default=None, help="skill 名，默认取 frontmatter")
    skills_add.add_argument("--description", default=None, help="覆盖 frontmatter 描述")
    skills_add.add_argument("--force", action="store_true", help="已存在时覆盖内容")
    skills_add.set_defaults(func=_cmd_skills_add)

    skills_remove = skills_sub.add_parser("remove", help="从清单摘除 skill")
    _add_common_options(skills_remove)
    skills_remove.add_argument("name", help="skill 名")
    skills_remove.set_defaults(func=_cmd_skills_remove)

    skills_show = skills_sub.add_parser("show", help="打印 skill 文件内容")
    _add_common_options(skills_show)
    skills_show.add_argument("name", help="skill 名")
    skills_show.add_argument("--file", default="SKILL.md", help="相对文件名")
    skills_show.set_defaults(func=_cmd_skills_show)

    # ---- groups ----
    groups = sub.add_parser("groups", help="工具树分组管理")
    _add_common_options(groups)
    groups_sub = groups.add_subparsers(dest="action", required=True)

    groups_list = groups_sub.add_parser("list", help="列出分组及成员数")
    _add_common_options(groups_list)
    groups_list.set_defaults(func=_cmd_groups_list)

    groups_create = groups_sub.add_parser("create", help="创建分组（工具树中间节点）")
    _add_common_options(groups_create)
    groups_create.add_argument("group_id", help="如 tool/mytools 或 mcp/filesystem")
    groups_create.add_argument("--description", default="", help="单行描述")
    groups_create.set_defaults(func=_cmd_groups_create)

    groups_remove = groups_sub.add_parser("remove", help="删除分组声明")
    _add_common_options(groups_remove)
    groups_remove.add_argument("group_id", help="分组标识")
    groups_remove.add_argument("--force", action="store_true", help="成员非空也删描述")
    groups_remove.set_defaults(func=_cmd_groups_remove)

    # ---- tools add-ext ----
    tools_add_ext = tools_sub.add_parser(
        "add-ext", help="装载外来工具（允许执行本体代码）"
    )
    _add_common_options(tools_add_ext)
    tools_add_ext.add_argument("path", help="工具路径，如 /tool/mytools/hello")
    tools_add_ext.add_argument("--target", required=True, help="./ext/hello.py 或 模块名")
    tools_add_ext.add_argument("--callable", default="run", help="目标函数名，默认 run")
    tools_add_ext.add_argument("--kind", default="python", choices=["python"], help="来源类型")
    tools_add_ext.add_argument("--description", default="", help="单行描述")
    tools_add_ext.add_argument("--caps", action="append", default=None, help="能力位，可重复")
    tools_add_ext.add_argument("--idempotent", action="store_true", help="标记为幂等")
    tools_add_ext.add_argument("--schema", default=None, help="参数 schema 的 JSON 文件")
    tools_add_ext.add_argument("--group", default="", help="省略时从 path 推导")
    tools_add_ext.set_defaults(func=_cmd_tools_add_ext)

    # ---- mcp ----
    mcp = sub.add_parser("mcp", help="MCP 服务器管理（JSON-RPC over stdio）")
    _add_common_options(mcp)
    mcp_sub = mcp.add_subparsers(dest="action", required=True)

    mcp_add = mcp_sub.add_parser("add", help="声明服务器并拉取工具清单")
    _add_common_options(mcp_add)
    mcp_add.add_argument("name", help="服务器名（同时作为 /mcp/<名> 分组）")
    mcp_add.add_argument("--command", required=True, help="可执行文件，如 npx")
    mcp_add.add_argument("--arg", action="append", default=None, help="命令参数，可重复")
    mcp_add.add_argument("--env", action="append", default=None, help="K=V，可重复")
    mcp_add.set_defaults(func=_cmd_mcp_add)

    mcp_list = mcp_sub.add_parser("list", help="列出 MCP 服务器")
    _add_common_options(mcp_list)
    mcp_list.set_defaults(func=_cmd_mcp_list)

    mcp_remove = mcp_sub.add_parser("remove", help="删除服务器及其工具")
    _add_common_options(mcp_remove)
    mcp_remove.add_argument("name", help="服务器名")
    mcp_remove.set_defaults(func=_cmd_mcp_remove)

    # ---- perms ----
    perms = sub.add_parser("perms", help="工具权限管理（真实屏蔽查询与调用）")
    _add_common_options(perms)
    perms_sub = perms.add_subparsers(dest="action", required=True)

    perms_list = perms_sub.add_parser("list", help="列出拒绝记录")
    _add_common_options(perms_list)
    perms_list.add_argument("--agent", action="append", default=None, help="限定 agent，可重复")
    perms_list.set_defaults(func=_cmd_perms_list)

    perms_deny = perms_sub.add_parser("deny", help="拒绝某路径（含子树）")
    _add_common_options(perms_deny)
    perms_deny.add_argument("path", help="绝对路径，如 /tool/os/shell/run")
    perms_deny.add_argument("--agent", action="append", default=None, help="默认 local+web")
    perms_deny.set_defaults(func=_cmd_perms_deny)

    perms_grant = perms_sub.add_parser("grant", help="恢复允许（删除例外）")
    _add_common_options(perms_grant)
    perms_grant.add_argument("path", help="绝对路径")
    perms_grant.add_argument("--agent", action="append", default=None, help="默认 local+web")
    perms_grant.set_defaults(func=_cmd_perms_grant)

    perms_show = perms_sub.add_parser("show", help="查某路径的实际判定")
    _add_common_options(perms_show)
    perms_show.add_argument("path", help="绝对路径")
    perms_show.add_argument("--agent", action="append", default=None, help="默认 local+web")
    perms_show.set_defaults(func=_cmd_perms_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口，返回退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - 交互中断
        print("\n已取消", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
