"""User-editable runtime settings (written from the settings UI or CLI).

Why a separate layer: :mod:`orange.defaults` holds the *shipped* defaults and is
frozen on purpose -- tests assert docs and code agree on it. A settings screen
must not mutate that file, and it must not silently rewrite ``.env`` either
(that file mixes secrets with tunables and is often under version control).

So user overrides live in their own JSON file under the data directory:

    ~/.OAgent/settings.json

Precedence::

    settings.json  >  environment variables / .env  >  orange.defaults

That order is deliberate. An override file is the user's explicit, deliberate
choice made through the UI; environment variables are the deployment's choice;
defaults are the fallback. Secrets never enter ``settings.json`` -- the settings
API refuses them and the file is plain text on disk.

Reload semantics: the file is read at request time (no restart needed to see a
change) and cached by mtime, so editing it by hand works too. Whether a value
takes effect immediately depends on what it configures -- see
:data:`RESTART_REQUIRED`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from orange import defaults as _defaults

__all__ = [
    "SETTINGS_FILENAME",
    "RESTART_REQUIRED",
    "SettingsError",
    "settings_path",
    "read_overrides",
    "write_overrides",
    "effective_config",
    "describe_schema",
    "clear_overrides",
]

#: File name inside the orange data directory.
SETTINGS_FILENAME = "settings.json"

#: Keys exposed in the settings UI.
#:
#: Deliberately short. The rest of the schema (memory file names, rate-window
#: internals, sqlite tuning, description lengths) are **deployment** concerns --
#: they belong to ``.env`` / the capability manifest and are managed from the
#: CLI, where they can be version-controlled and reviewed. Putting them on a
#: slider invites people to break their install with no way to see what changed.
#:
#: The rule for inclusion: a user would plausibly want a different value *for a
#: single conversation* or *for this machine's load*, not for the installation.
KEY_SETTINGS: tuple[str, ...] = (
    "loop.max_steps",
    "loop.max_injects",
    "context.capacity",
    "execution.max_workers",
    "tools.index_depth",
    "tools.max_discovered",
)

#: Config keys whose change needs a service restart. Everything else is read per
#: request (or per session) and therefore applies immediately.
RESTART_REQUIRED: frozenset[str] = frozenset(
    {
        "context.capacity",
        "sessions.max_resident",
        "sessions.idle_timeout_s",
        "sessions.sweep_interval_s",
        "sessions.sqlite_timeout",
        "rate.window_s",
        "rate.max_requests",
        "rate.max_tokens",
        "rate.failure_threshold",
        "rate.health_backoff_s",
        "rate.request_backoff_s",
        "execution.max_workers",
        "execution.chat_capacity",
        "memory.agent_dir",
        "memory.work_memory",
        "memory.long_term_memory",
        "memory.extract_principle",
        "memory.trace_dir",
    }
)

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"mtime": None, "data": {}}


class SettingsError(ValueError):
    """A settings write was rejected (unknown key, wrong type, out of range)."""


def settings_path(home: Path | None = None) -> Path:
    """Location of the override file (``<home>/settings.json``)."""
    if home is not None:
        return Path(home) / SETTINGS_FILENAME
    from orange import paths

    return paths.orange_home() / SETTINGS_FILENAME


def _groups() -> dict[str, Any]:
    """Group name -> frozen dataclass instance of shipped defaults."""
    return {
        "context": _defaults.DEFAULT_CONTEXT,
        "loop": _defaults.DEFAULT_LOOP,
        "execution": _defaults.DEFAULT_EXECUTION,
        "tools": _defaults.DEFAULT_TOOLS,
        "sessions": _defaults.DEFAULT_SESSIONS,
        "rate": _defaults.DEFAULT_RATE,
        "memory": _defaults.DEFAULT_MEMORY,
    }


def _schema() -> dict[str, dict[str, Any]]:
    """Flattened ``group.name`` -> ``{default, type, group, name}``."""
    schema: dict[str, dict[str, Any]] = {}
    for group, instance in _groups().items():
        if not is_dataclass(instance):
            continue
        for item in fields(instance):
            value = getattr(instance, item.name)
            schema[f"{group}.{item.name}"] = {
                "group": group,
                "name": item.name,
                "default": value,
                # ``type(value)`` is the arbiter of what an override may contain:
                # accepting a float where an int is expected would silently
                # break range checks elsewhere.
                "type": type(value).__name__,
                "restart_required": f"{group}.{item.name}" in RESTART_REQUIRED,
            }
    return schema


def describe_schema() -> list[dict[str, Any]]:
    """Machine-readable catalogue for the settings UI.

    Returns:
        One dict per tunable, sorted by group then name, carrying the shipped
        default, the expected type, and whether changing it needs a restart.
    """
    schema = _schema()
    return [
        {"key": key, **meta}
        for key, meta in sorted(schema.items(), key=lambda kv: (kv[1]["group"], kv[1]["name"]))
    ]


def read_overrides(home: Path | None = None) -> dict[str, Any]:
    """Read the override file (``{}`` when absent or unreadable).

    A corrupt file degrades to "no overrides" rather than failing the service:
    the same reasoning as the permission file -- bad configuration should not
    make the app unusable.
    """
    path = settings_path(home)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    with _LOCK:
        if _CACHE["mtime"] == mtime:
            return dict(_CACHE["data"])
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        data = raw if isinstance(raw, dict) else {}
        valid = {k: v for k, v in data.items() if isinstance(k, str) and "." in k}
        _CACHE["mtime"] = mtime
        _CACHE["data"] = valid
        return dict(valid)


def _coerce(key: str, value: Any, meta: dict[str, Any]) -> Any:
    """Validate and normalise one value against the schema entry."""
    expected = meta["type"]
    if expected == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{key} 需要整数，收到 {type(value).__name__}")
        if isinstance(value, float) and not value.is_integer():
            raise SettingsError(f"{key} 需要整数，收到 {value}")
        return int(value)
    if expected == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{key} 需要数值，收到 {type(value).__name__}")
        return float(value)
    if expected == "str":
        if not isinstance(value, str) or not value.strip():
            raise SettingsError(f"{key} 需要非空字符串")
        return value.strip()
    raise SettingsError(f"{key} 的类型 {expected} 不支持在线修改")


def write_overrides(
    updates: dict[str, Any],
    home: Path | None = None,
    *,
    only_keys: bool = False,
) -> dict[str, Any]:
    """Merge ``updates`` into the override file and return the new contents.

    Entries that equal the shipped default are **removed** rather than stored:
    keeping them would freeze today's default into the user's file, so a future
    change to the shipped default would silently not apply.

    Args:
        updates: ``{"loop.max_steps": 60}`` style mapping.
        home: Override data directory (tests).
        only_keys: Reject anything outside :data:`KEY_SETTINGS`. The settings UI
            sets this; the CLI does not, because a deployment may legitimately
            tune internals that the UI deliberately hides.

    Returns:
        The full override map after merging.

    Raises:
        SettingsError: Unknown key, wrong type, or non-positive value where a
            positive number is required.
    """
    schema = _schema()
    unknown = sorted(k for k in updates if k not in schema)
    if unknown:
        raise SettingsError("未知配置项: " + ", ".join(unknown))

    if only_keys:
        hidden = sorted(k for k in updates if k not in KEY_SETTINGS)
        if hidden:
            raise SettingsError(
                "以下配置项不在界面可改范围，请用 CLI："
                + ", ".join(hidden)
                + "（例如 orange config set）"
            )

    current = read_overrides(home)
    normalised: dict[str, Any] = {}
    for key, value in updates.items():
        meta = schema[key]
        coerced = _coerce(key, value, meta)
        # Every tunable in this system is a positive quantity; zero or negative
        # values would disable the feature in a way the caller almost never means.
        if isinstance(coerced, (int, float)) and coerced <= 0:
            raise SettingsError(f"{key} 必须大于 0，收到 {coerced}")
        normalised[key] = coerced

    merged = {**current, **normalised}
    # Drop no-ops so the file only records genuine deviations from the defaults.
    merged = {
        key: value
        for key, value in merged.items()
        if key in schema and value != schema[key]["default"]
    }
    return _rewrite(merged, home)


def clear_overrides(keys: list[str] | None = None, home: Path | None = None) -> dict[str, Any]:
    """Drop the given overrides (all of them when ``keys`` is ``None``).

    Args:
        keys: Keys to remove; ``None`` removes every override.
        home: Override data directory (tests).

    Returns:
        The override map that remains.
    """
    if keys is None:
        return _rewrite({}, home)
    current = read_overrides(home)
    for key in keys:
        current.pop(key, None)
    return _rewrite(current, home)


def _rewrite(data: dict[str, Any], home: Path | None) -> dict[str, Any]:
    """Replace the file wholesale (used by :func:`clear_overrides`)."""
    path = settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    with _LOCK:
        _CACHE["mtime"] = None
        _CACHE["data"] = {}
    return data


def effective_config(
    home: Path | None = None, *, only_keys: bool = True
) -> dict[str, list[dict[str, Any]]]:
    """Grouped effective values plus provenance, for the settings UI.

    Each entry reports ``default``, ``value`` and ``source`` so the UI can show
    which values the user has overridden.

    Args:
        home: Override data directory (tests).
        only_keys: Restrict to :data:`KEY_SETTINGS`. ``False`` returns the full
            schema, which the CLI uses for ``config show``.

    Returns:
        ``{group: [{key, name, value, default, type, restart_required, overridden}]}``
    """
    overrides = read_overrides(home)
    grouped: dict[str, list[dict[str, Any]]] = {}
    allowed = set(KEY_SETTINGS)
    for key, meta in sorted(_schema().items(), key=lambda kv: (kv[1]["group"], kv[1]["name"])):
        if only_keys and key not in allowed:
            continue
        value = overrides.get(key, meta["default"])
        grouped.setdefault(meta["group"], []).append(
            {
                "key": key,
                "name": meta["name"],
                "value": value,
                "default": meta["default"],
                "type": meta["type"],
                "restart_required": meta["restart_required"],
                "overridden": key in overrides,
            }
        )
    return grouped
