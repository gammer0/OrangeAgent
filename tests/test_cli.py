"""CLI 测试：解析、退出码、JSON 输出与密钥掩码。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.cli import build_parser, main


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestParser:
    def test_requires_command(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_service_requires_action(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["service"])

    def test_workdir_is_global(self) -> None:
        args = build_parser().parse_args(["--workdir", "/tmp/x", "config", "show"])
        assert args.workdir == "/tmp/x"

    def test_json_flag_global(self) -> None:
        args = build_parser().parse_args(["--json", "config", "get", "loop.max_steps"])
        assert args.json is True


class TestConfigCommands:
    def test_get_qualified_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["config", "get", "context.capacity"], capsys)
        assert code == 0
        assert "32768" in out

    def test_get_bare_key_is_resolved(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["config", "get", "max_steps"], capsys)
        assert code == 0
        assert "loop.max_steps" in out
        assert "20" in out

    def test_unknown_key_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["config", "get", "not_a_real_key"], capsys)
        assert code == 2
        assert "未知配置项" in out

    def test_json_output_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["--json", "config", "get", "context.auto_1"], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["key"] == "context.auto_1"
        assert payload["value"] == 0.5

    def test_show_contains_all_groups(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["config", "show"], capsys)
        assert code == 0
        for group in ("context", "loop", "execution", "tools", "sessions", "rate", "memory"):
            assert f"[{group}]" in out

    def test_show_json_matches_defaults(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["--json", "config", "show"], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["defaults"]["context"]["capacity"] == 32768

    def test_env_lists_required_vars(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["config", "env"], capsys)
        assert code == 0
        assert "DEEPSEEK_API_KEY" in out
        assert "LLM_TEMPERATURE" in out

    def test_env_masks_api_key(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-supersecret1234567890")
        code, out = _run(["--json", "config", "env"], capsys)
        assert code == 0
        assert "supersecret" not in out
        payload = json.loads(out)
        key_row = next(v for v in payload["variables"] if v["name"] == "DEEPSEEK_API_KEY")
        assert key_row["value"] == "sk-s...7890"

    def test_init_creates_template(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out = _run(
            ["config", "init", "--output", "env.tpl", "--workdir", str(tmp_path)], capsys
        )
        assert code == 0
        created = tmp_path / "env.tpl"
        assert created.is_file()
        assert "DEEPSEEK_API_KEY" in created.read_text(encoding="utf-8")

    def test_init_refuses_overwrite(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "env.tpl"
        target.write_text("old", encoding="utf-8")
        code, out = _run(
            ["config", "init", "--output", "env.tpl", "--workdir", str(tmp_path)], capsys
        )
        assert code == 2
        assert target.read_text(encoding="utf-8") == "old"

    def test_init_force_overwrites(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "env.tpl"
        target.write_text("old", encoding="utf-8")
        code, _ = _run(
            ["config", "init", "--output", "env.tpl", "--force", "--workdir", str(tmp_path)],
            capsys,
        )
        assert code == 0
        assert "DEEPSEEK_API_KEY" in target.read_text(encoding="utf-8")

    def test_check_reports_missing_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        code, out = _run(["config", "check", "--workdir", str(tmp_path)], capsys)
        assert code == 1
        assert "配置无效" in out
        assert "sk-" not in out.split("配置无效")[1]

    def test_check_passes_with_valid_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-valid-key-123456")
        code, out = _run(["--json", "config", "check", "--workdir", str(tmp_path)], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["ok"] is True


class TestServiceCommands:
    def test_status_without_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["service", "status", "--no-model"], capsys)
        assert code == 0
        assert "capability" in out
        assert "store" in out

    def test_status_json_has_layers(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["--json", "service", "status", "--no-model"], capsys)
        assert code == 0
        payload = json.loads(out)
        names = {layer["name"] for layer in payload["layers"]}
        assert {"capability", "session", "store", "context"} <= names

    def test_status_degraded_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """缺少密钥时应报告 degraded 并以 1 退出。"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        code, out = _run(["service", "status", "--workdir", str(tmp_path)], capsys)
        assert code == 1
        assert "WARN" in out or "FAIL" in out

    def test_status_allow_degraded(self, tmp_path: Path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        code, _ = _run(
            ["service", "status", "--workdir", str(tmp_path), "--allow-degraded"], capsys
        )
        assert code == 0

    def test_tree_prints_index(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["service", "tree"], capsys)
        assert code == 0
        assert "工具树" in out or "/" in out

    def test_sweep_reports_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["service", "sweep"], capsys)
        assert code == 0
        assert "已驱逐" in out

    def test_doctor_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["service", "doctor"], capsys)
        assert code in {0, 1}
        assert "自检项" in out

    def test_doctor_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(["--json", "service", "doctor"], capsys)
        assert code in {0, 1}
        payload = json.loads(out)
        assert "checks" in payload
        assert "status" in payload
