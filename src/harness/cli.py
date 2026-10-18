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

from harness.config import ConfigError, load_llm_config
from harness.defaults import as_dict as defaults_as_dict
from harness.service import ServiceStatus, build_service

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
    tree.add_argument("--depth", type=int, default=2, help="索引深度，默认 2")
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
