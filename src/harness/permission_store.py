"""权限持久化：把 ``AgentPermissions.grants`` 落到 ``.agent/permissions.json``。

对应文档：`docs/06-工具分组与外来能力.md` 第五节。

设计要点：

* **只存例外**：``grants`` 的键是被拒绝的绝对路径，值是 ``false``。
  未记录的路径默认允许（B15），因此文件通常很小。
* **按 agent 分区**：服务组合根用 ``local``，Web 会话用 ``web``。同一份
  策略通常两边都要，但允许分别配置（例如只在 Web 侧禁用 shell）。
* **损坏降级**：文件损坏时**不**抛异常中断服务，而是返回空表并记录原因。
  安全性上这等价于「全部允许」——但把不可读的配置当成「禁止一切」会让
  服务直接不可用，权衡后选择降级 + 显式报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.permission import AgentPermissions, is_absolute_path

__all__ = [
    "PERMISSION_DIR_NAME",
    "PERMISSION_FILENAME",
    "PERMISSION_VERSION",
    "permissions_path",
    "load_permissions",
    "save_permissions",
    "set_grant",
]

#: 权限文件所在目录（与能力清单同处 ``.agent``）。
PERMISSION_DIR_NAME = ".agent"

#: 权限文件名。
PERMISSION_FILENAME = "permissions.json"

#: 文件格式版本。
PERMISSION_VERSION = 1

#: 服务组合根使用的 agent 标识。
LOCAL_AGENT = "local"

#: Web 会话使用的 agent 标识。
WEB_AGENT = "web"


def permissions_path(workdir: str | Path) -> Path:
    """权限文件路径：``<workdir>/.agent/permissions.json``。"""
    return (
        Path(workdir).resolve() / PERMISSION_DIR_NAME / PERMISSION_FILENAME
    )


def _read_raw(workdir: str | Path) -> dict[str, Any]:
    """读取文件为 dict；文件不存在视为空配置。"""
    target = permissions_path(workdir)
    if not target.is_file():
        return {"version": PERMISSION_VERSION, "agents": {}}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": PERMISSION_VERSION, "agents": {}}
    if not isinstance(data, dict):
        return {"version": PERMISSION_VERSION, "agents": {}}
    data.setdefault("agents", {})
    return data


def load_permissions(
    workdir: str | Path,
    agent_id: str = LOCAL_AGENT,
    *,
    problems: list[str] | None = None,
) -> AgentPermissions:
    """读取某 agent 的权限表。

    文件缺失或损坏时返回空表（= 全部允许），并把原因写入 ``problems``。
    非法路径（非绝对路径）会被跳过，避免一条坏记录让整份策略失效。

    Args:
        workdir: 工作目录。
        agent_id: ``local`` / ``web``，或用自定义标识。
        problems: 可选列表，用于收集降级原因。

    Returns:
        该 agent 的权限表。
    """
    raw = _read_raw(workdir)
    agents = raw.get("agents") or {}
    entry = agents.get(agent_id) or {}
    grants_raw = entry.get("grants") or {}

    grants: dict[str, bool] = {}
    for path, allowed in grants_raw.items():
        if not is_absolute_path(path):
            if problems is not None:
                problems.append(f"permissions: 忽略非法路径 {path!r}")
            continue
        if allowed is not True:
            # 只保留拒绝项；显式 true 与「无记录」等价，不占空间
            grants[path] = False

    try:
        return AgentPermissions(agent_id=agent_id, grants=grants)
    except Exception as exc:  # noqa: BLE001 - 校验失败时降级
        if problems is not None:
            problems.append(f"permissions: {exc}")
        return AgentPermissions(agent_id=agent_id)


def save_permissions(
    permissions: AgentPermissions, workdir: str | Path
) -> Path:
    """原子写入某 agent 的权限表（保留其他 agent 的配置）。"""
    raw = _read_raw(workdir)
    agents = raw.setdefault("agents", {})
    agents[permissions.agent_id] = {
        "grants": {path: False for path in sorted(permissions.grants) if not permissions.grants[path]}
    }
    raw["version"] = PERMISSION_VERSION

    target = permissions_path(workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def set_grant(
    workdir: str | Path,
    path: str,
    *,
    allowed: bool,
    agent_id: str = LOCAL_AGENT,
) -> AgentPermissions:
    """设置单条权限并落盘，返回新的权限表。

    ``allowed=False`` 表示拒绝该路径（及其子树，由 ``visible`` 保证）；
    ``allowed=True`` 表示删除该例外，回到「默认允许」。

    Raises:
        ValueError: 路径不是合法绝对路径（B16）。
    """
    if not is_absolute_path(path):
        raise ValueError(f"B16: 权限判定只接受完整绝对路径，实际为 {path!r}")

    current = load_permissions(workdir, agent_id)
    updated = current.grant(path, True) if allowed else current.grant(path, False)
    save_permissions(updated, workdir)
    return updated


def list_grants(workdir: str | Path) -> dict[str, dict[str, bool]]:
    """列出所有 agent 的拒绝记录，供 CLI 展示。"""
    raw = _read_raw(workdir)
    result: dict[str, dict[str, bool]] = {}
    for agent_id, entry in (raw.get("agents") or {}).items():
        grants = (entry or {}).get("grants") or {}
        result[agent_id] = {path: bool(value) for path, value in grants.items()}
    return result
