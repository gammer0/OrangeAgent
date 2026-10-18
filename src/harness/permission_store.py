"""Permission persistence: store ``AgentPermissions.grants`` in ``.agent/permissions.json``.

See section 5 of the tool-groups-and-external-capabilities doc.

Design notes:

* **Exceptions only**: keys in ``grants`` are denied absolute paths with value
  ``false``. Unrecorded paths default to allowed (B15), so the file stays small.
* **Partitioned by agent**: the service composition root uses ``local`` and web
  sessions use ``web``. Both usually need the same policy, but they can be
  configured separately (e.g. disable shell only on the web side).
* **Degrade on corruption**: a corrupt file does **not** raise and take the
  service down; we return an empty table and record the reason. Security-wise
  this is equivalent to "allow everything" -- but treating an unreadable config
  as "deny everything" would make the service unusable, so degrading plus an
  explicit error report is the trade-off we chose.
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

#: Directory holding the permission file (the same ``.agent`` as the capability manifest).
PERMISSION_DIR_NAME = ".agent"

#: Permission file name.
PERMISSION_FILENAME = "permissions.json"

#: File format version.
PERMISSION_VERSION = 1

#: Agent identifier used by the service composition root.
LOCAL_AGENT = "local"

#: Agent identifier used by web sessions.
WEB_AGENT = "web"


def permissions_path(workdir: str | Path) -> Path:
    """Permission file path: ``<workdir>/.agent/permissions.json``."""
    return (
        Path(workdir).resolve() / PERMISSION_DIR_NAME / PERMISSION_FILENAME
    )


def _read_raw(workdir: str | Path) -> dict[str, Any]:
    """Read the file as a dict; a missing file counts as empty configuration."""
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
    """Load one agent's permission table.

    A missing or corrupt file yields an empty table (= allow everything) and the
    reason is appended to ``problems``. Invalid paths (non-absolute) are skipped
    so that one bad record cannot invalidate the whole policy.

    Args:
        workdir: Working directory.
        agent_id: ``local`` / ``web``, or a custom identifier.
        problems: Optional list collecting degradation reasons.

    Returns:
        The permission table for that agent.
    """
    raw = _read_raw(workdir)
    agents = raw.get("agents") or {}
    entry = agents.get(agent_id) or {}
    grants_raw = entry.get("grants") or {}

    grants: dict[str, bool] = {}
    for path, allowed in grants_raw.items():
        if not is_absolute_path(path):
            if problems is not None:
                problems.append(f"permissions: ignoring invalid path {path!r}")
            continue
        if allowed is not True:
            # Keep denials only; an explicit true is equivalent to "no record"
            # and would waste space.
            grants[path] = False

    try:
        return AgentPermissions(agent_id=agent_id, grants=grants)
    except Exception as exc:  # noqa: BLE001 - degrade when validation fails
        if problems is not None:
            problems.append(f"permissions: {exc}")
        return AgentPermissions(agent_id=agent_id)


def save_permissions(
    permissions: AgentPermissions, workdir: str | Path
) -> Path:
    """Atomically write one agent's permission table (preserving other agents)."""
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
    """Set a single permission and persist it, returning the new table.

    ``allowed=False`` denies that path (and its subtree, guaranteed by
    ``visible``); ``allowed=True`` removes the exception and returns to
    "allowed by default".

    Raises:
        ValueError: The path is not a valid absolute path (B16).
    """
    if not is_absolute_path(path):
        raise ValueError(
            f"B16: permission checks accept full absolute paths only, got {path!r}"
        )

    current = load_permissions(workdir, agent_id)
    updated = current.grant(path, True) if allowed else current.grant(path, False)
    save_permissions(updated, workdir)
    return updated


def list_grants(workdir: str | Path) -> dict[str, dict[str, bool]]:
    """List denial records for every agent, for CLI display."""
    raw = _read_raw(workdir)
    result: dict[str, dict[str, bool]] = {}
    for agent_id, entry in (raw.get("agents") or {}).items():
        grants = (entry or {}).get("grants") or {}
        result[agent_id] = {path: bool(value) for path, value in grants.items()}
    return result
