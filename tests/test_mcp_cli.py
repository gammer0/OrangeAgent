"""MCP 命令行装载的测试：参数传递与清单落盘。

覆盖两个曾经踩坑的点：

* ``--arg`` 传入以 ``--`` 开头的值（标准 MCP 启动参数几乎都带 ``--``）；
* 参数文件带 UTF-8 BOM（PowerShell ``Out-File`` 的默认行为）时不能带脏字符。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness.capabilities import load_manifest
from harness.cli import main

SERVER = Path(__file__).resolve().parents[1] / "mcp_servers" / "workspace_server.py"


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """准备一个空工作区，并放一个可用的 MCP 服务器副本。"""
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "capabilities.json").write_text(
        '{"version": 2, "tools": [], "skills": [], "groups": []}', encoding="utf-8"
    )
    target = tmp_path / "server.py"
    target.write_text(SERVER.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello", encoding="utf-8")
    return tmp_path


def _add(workdir: Path, *extra: str) -> int:
    return main(
        [
            "mcp",
            "add",
            "ws",
            "--workdir",
            str(workdir),
            "--command",
            sys.executable,
            *extra,
        ]
    )


class TestArgPassing:
    def test_equals_form_accepts_dash_values(self, workdir: Path) -> None:
        """``--arg=--root`` 形式必须可用（否则无法传 -- 开头的参数）。"""
        code = _add(workdir, "--arg=server.py", "--arg=--root", "--arg=.")
        assert code == 0
        servers = load_manifest(workdir).mcp_servers
        assert servers[0]["args"] == ["server.py", "--root", "."]

    def test_args_file_handles_bom(self, workdir: Path) -> None:
        """参数文件带 BOM 时首行不能带 \\ufeff。"""
        args_file = workdir / "args.txt"
        # 显式写入 BOM，模拟 PowerShell Out-File 的产物
        args_file.write_text("server.py\n--root\n.\n", encoding="utf-8-sig")
        assert args_file.read_bytes().startswith(b"\xef\xbb\xbf")

        code = _add(workdir, "--args-file", str(args_file))
        assert code == 0
        servers = load_manifest(workdir).mcp_servers
        assert servers[0]["args"] == ["server.py", "--root", "."]

    def test_args_file_skips_blank_lines(self, workdir: Path) -> None:
        args_file = workdir / "args.txt"
        args_file.write_text("server.py\n\n--root\n\n.\n\n", encoding="utf-8")
        assert _add(workdir, "--args-file", str(args_file)) == 0
        assert load_manifest(workdir).mcp_servers[0]["args"] == ["server.py", "--root", "."]

    def test_missing_args_file(self, workdir: Path) -> None:
        assert _add(workdir, "--args-file", str(workdir / "nope.txt")) == 2


class TestMount:
    def test_tools_mounted_under_server_group(self, workdir: Path) -> None:
        assert _add(workdir, "--arg=server.py", "--arg=--root", "--arg=.") == 0

        manifest = load_manifest(workdir)
        paths = manifest.tool_paths()
        expected = [
            "/mcp/ws/list_dir",
            "/mcp/ws/read_file",
            "/mcp/ws/search_text",
            "/mcp/ws/stat_path",
            "/mcp/ws/write_file",
        ]
        assert paths == expected

        for path in expected:
            source = manifest.sources[path]
            assert source.kind == "mcp"
            assert source.target == "ws"

    def test_schema_carried_from_server(self, workdir: Path) -> None:
        """服务器的 inputSchema 必须原样进入清单，否则模型不知道参数。"""
        assert _add(workdir, "--arg=server.py", "--arg=--root", "--arg=.") == 0

        manifest = load_manifest(workdir)
        tool = manifest.get_tool("/mcp/ws/read_file")
        assert tool is not None
        props = tool.schema.get("properties") or {}
        assert "path" in props
        assert tool.schema.get("required") == ["path"]

    def test_remove_clears_tools_and_server(self, workdir: Path) -> None:
        assert _add(workdir, "--arg=server.py", "--arg=--root", "--arg=.") == 0
        assert main(["mcp", "remove", "ws", "--workdir", str(workdir)]) == 0

        manifest = load_manifest(workdir)
        assert manifest.mcp_servers == []
        assert not [p for p in manifest.tool_paths() if p.startswith("/mcp/ws/")]

    def test_mcp_list_reports_tool_count(self, workdir: Path, capsys) -> None:
        assert _add(workdir, "--arg=server.py", "--arg=--root", "--arg=.") == 0
        capsys.readouterr()
        assert main(["mcp", "list", "--workdir", str(workdir), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["servers"][0]["name"] == "ws"
        assert payload["servers"][0]["tools"] == 5


class TestBadServer:
    def test_unreachable_command(self, workdir: Path) -> None:
        code = main(
            [
                "mcp",
                "add",
                "bad",
                "--workdir",
                str(workdir),
                "--command",
                "definitely-not-a-real-command-xyz",
            ]
        )
        assert code == 2
        assert load_manifest(workdir).mcp_servers == []

    def test_server_that_exits_immediately(self, workdir: Path) -> None:
        """服务器立即退出时应报错且不写入清单。"""
        code = main(
            [
                "mcp",
                "add",
                "bad",
                "--workdir",
                str(workdir),
                "--command",
                sys.executable,
                "--arg=-c",
                "--arg=import sys; sys.exit(3)",
            ]
        )
        assert code == 2
        assert load_manifest(workdir).mcp_servers == []


class TestCommandResolution:
    """Windows 上 npm 工具是 .cmd 包装脚本，直接传名字会 FileNotFoundError。"""

    def test_posix_passes_through(self, monkeypatch) -> None:
        from harness import mcp_client

        monkeypatch.setattr(mcp_client.os, "name", "posix")
        assert mcp_client._resolve_command("npx") == ["npx"]

    def test_windows_cmd_wrapped_via_cmd_exe(self, monkeypatch, tmp_path) -> None:
        from harness import mcp_client

        fake = tmp_path / "npx.cmd"
        fake.write_text("@echo off\n", encoding="utf-8")

        monkeypatch.setattr(mcp_client.os, "name", "nt")
        monkeypatch.setattr(mcp_client.shutil, "which", lambda name: str(fake))

        assert mcp_client._resolve_command("npx") == ["cmd.exe", "/c", str(fake)]

    def test_windows_exe_used_directly(self, monkeypatch, tmp_path) -> None:
        from harness import mcp_client

        fake = tmp_path / "python.exe"
        fake.write_text("", encoding="utf-8")

        monkeypatch.setattr(mcp_client.os, "name", "nt")
        monkeypatch.setattr(mcp_client.shutil, "which", lambda name: str(fake))

        assert mcp_client._resolve_command("python") == [str(fake)]

    def test_windows_already_has_suffix(self, monkeypatch) -> None:
        from harness import mcp_client

        monkeypatch.setattr(mcp_client.os, "name", "nt")
        # 带扩展名时不再走 which，原样返回
        assert mcp_client._resolve_command("tool.exe") == ["tool.exe"]

    def test_windows_missing_command_kept_for_error(self, monkeypatch) -> None:
        """找不到时原样返回，让 Popen 抛 FileNotFoundError 并翻译成 McpError。"""
        from harness import mcp_client

        monkeypatch.setattr(mcp_client.os, "name", "nt")
        monkeypatch.setattr(mcp_client.shutil, "which", lambda name: None)

        assert mcp_client._resolve_command("definitely-missing") == ["definitely-missing"]

