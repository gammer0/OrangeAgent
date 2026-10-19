"""用户可编辑的超参数层（settings.json）。

这一层的三条硬性要求，任何改动都不得破坏：
  * 只记录与出厂默认不同的项 —— 否则未来的默认值变更会对老用户失效；
  * 类型与取值必须按 schema 校验 —— 否则下游会拿到无法使用的值；
  * 文件损坏时降级为"无覆盖" —— 配置写坏不该让服务起不来。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orange.settings import (
    KEY_SETTINGS,
    RESTART_REQUIRED,
    SettingsError,
    clear_overrides,
    describe_schema,
    effective_config,
    read_overrides,
    settings_path,
    write_overrides,
)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    return tmp_path


def test_defaults_apply_when_no_file(home: Path) -> None:
    assert read_overrides(home) == {}
    groups = effective_config(home)
    assert "loop" in groups
    assert all(not item["overridden"] for items in groups.values() for item in items)


def test_write_then_read_roundtrip(home: Path) -> None:
    write_overrides({"loop.max_steps": 55}, home)
    assert read_overrides(home) == {"loop.max_steps": 55}
    assert settings_path(home).is_file()


def test_only_deviations_are_stored(home: Path) -> None:
    """Writing a value equal to the default must not persist anything.

    Otherwise today's default gets frozen into the user's file, and a future
    release that improves the default would silently fail to reach them.
    """
    write_overrides({"loop.max_steps": 40}, home)  # 40 is the shipped default
    assert read_overrides(home) == {}
    assert json.loads(settings_path(home).read_text(encoding="utf-8")) == {}


def test_reset_to_default_removes_the_override(home: Path) -> None:
    write_overrides({"loop.max_steps": 55}, home)
    write_overrides({"loop.max_steps": 40}, home)
    assert read_overrides(home) == {}


def test_unknown_key_is_rejected(home: Path) -> None:
    with pytest.raises(SettingsError):
        write_overrides({"loop.nope": 1}, home)


def test_wrong_type_is_rejected(home: Path) -> None:
    with pytest.raises(SettingsError):
        write_overrides({"loop.max_steps": "many"}, home)
    with pytest.raises(SettingsError):
        write_overrides({"context.auto_1": "high"}, home)


def test_float_rejected_for_int_field(home: Path) -> None:
    """2.5 steps is not a thing; silently truncating would hide the typo."""
    with pytest.raises(SettingsError):
        write_overrides({"loop.max_steps": 2.5}, home)


def test_non_positive_is_rejected(home: Path) -> None:
    with pytest.raises(SettingsError):
        write_overrides({"loop.max_steps": 0}, home)
    with pytest.raises(SettingsError):
        write_overrides({"rate.max_requests": -1}, home)


def test_corrupt_file_degrades_to_no_overrides(home: Path) -> None:
    path = settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert read_overrides(home) == {}


def test_clear_single_key(home: Path) -> None:
    write_overrides({"loop.max_steps": 55, "loop.max_injects": 20}, home)
    remaining = clear_overrides(["loop.max_steps"], home)
    assert remaining == {"loop.max_injects": 20}


def test_clear_all(home: Path) -> None:
    write_overrides({"loop.max_steps": 55}, home)
    assert clear_overrides(None, home) == {}
    assert read_overrides(home) == {}


def test_schema_reports_types_and_restart_flags(home: Path) -> None:
    rows = {row["key"]: row for row in describe_schema()}
    assert rows["loop.max_steps"]["type"] == "int"
    assert rows["context.auto_1"]["type"] == "float"
    assert rows["loop.max_steps"]["restart_required"] is False
    assert rows["context.capacity"]["restart_required"] is True


def test_restart_flags_only_name_real_keys() -> None:
    known = {row["key"] for row in describe_schema()}
    assert known >= RESTART_REQUIRED, "a restart flag must not name a phantom key"


def test_no_secrets_in_schema() -> None:
    """The settings file is plain text; an API key must never be writable there.

    Note ``rate.max_tokens`` is a legitimate *quota* setting, not a secret -- the
    check therefore looks for credential-shaped names only.
    """
    keys = {row["key"].lower() for row in describe_schema()}
    forbidden = ("api_key", "apikey", "secret", "password", "authorization")
    assert not any(any(bad in key for bad in forbidden) for key in keys)


def test_effective_config_marks_overrides(home: Path) -> None:
    write_overrides({"loop.max_steps": 55}, home)
    groups = effective_config(home)
    entry = next(i for i in groups["loop"] if i["key"] == "loop.max_steps")
    assert entry["value"] == 55
    assert entry["default"] == 40
    assert entry["overridden"] is True


def test_effective_config_hides_internal_keys_by_default(home: Path) -> None:
    """界面只看关键项；其余交给 CLI。"""
    groups = effective_config(home)
    keys = {item["key"] for items in groups.values() for item in items}
    assert keys == set(KEY_SETTINGS)
    assert "memory.trace_dir" not in keys
    assert "rate.window_s" not in keys


def test_full_schema_still_available(home: Path) -> None:
    groups = effective_config(home, only_keys=False)
    keys = {item["key"] for items in groups.values() for item in items}
    assert "memory.trace_dir" in keys
    assert len(keys) > len(KEY_SETTINGS)


def test_ui_cannot_write_hidden_keys(home: Path) -> None:
    """界面层写入超出白名单的项必须被拒绝，并指向 CLI。"""
    with pytest.raises(SettingsError, match="CLI"):
        write_overrides({"memory.trace_dir": "t2"}, home, only_keys=True)
    # The CLI path may still set it.
    write_overrides({"memory.trace_dir": "t2"}, home)
    assert read_overrides(home)["memory.trace_dir"] == "t2"


def test_atomic_write_leaves_no_temp_file(home: Path) -> None:
    write_overrides({"loop.max_steps": 55}, home)
    leftovers = list(home.glob("*.tmp"))
    assert leftovers == [], "a temp file must never survive a successful write"
