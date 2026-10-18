"""参数四道关卡测试（J1–J4）。

对应 docs/02-数据结构设计/09-参数校验.md。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import pytest

from harness.args_validate import (
    ArgumentValidation,
    parse_call_arguments,
    validate_arguments,
)
from harness.execution import ExecResult
from harness.message import ToolCall

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "default": 0},
        "mode": {"type": "string", "enum": ["safe", "exclusive"]},
    },
    "required": ["path"],
}


class ArgumentParams(Protocol):
    """成功分支的返回类型（仅测试用，避免依赖具体类）。"""

    params: dict[str, Any]
    canonical: dict[str, Any]
    ignored: list[str]


def _ok(raw: str, schema: dict[str, Any] | None = None) -> ArgumentValidation:
    result, error = validate_arguments(raw, schema if schema is not None else SCHEMA)
    assert error is None, error
    assert result is not None
    return result


def _err(raw: str, schema: dict[str, Any] | None = None) -> tuple[dict[str, Any], ExecResult]:
    result, error = validate_arguments(raw, schema if schema is not None else SCHEMA)
    assert result is None
    assert error is not None
    assert error.ok is False
    assert error.error_type == "3d"
    assert isinstance(error.content, str)
    payload = json.loads(error.content)  # J2：必须是合法 JSON
    assert "stage" in payload
    assert isinstance(payload["errors"], list)
    return payload, error


class TestGateParse:
    def test_truncated_json_fails_at_parse_stage(self) -> None:
        """① 关：截断 JSON → stage=parse，附原始字符串与错误位置。"""
        raw = '{"path": "a.txt"'
        payload, error = _err(raw)

        assert payload["stage"] == "parse"
        assert payload["raw"] == raw
        assert payload["position"]["line"] >= 1
        assert payload["position"]["column"] >= 1
        assert error.retryable is False

    def test_trailing_comma_is_not_auto_repaired(self) -> None:
        """J1：尾逗号不得被自动修复。"""
        payload, _ = _err('{"path": "a.txt",}')
        assert payload["stage"] == "parse"

    def test_single_quotes_are_not_repaired(self) -> None:
        """J1：单引号不是合法 JSON，不得自动转换。"""
        payload, _ = _err("{'path': 'a.txt'}")
        assert payload["stage"] == "parse"

    def test_non_object_json_is_schema_failure(self) -> None:
        """① 关通过但根不是对象 → schema 关失败，kind=type。"""
        payload, _ = _err('["a.txt"]')
        assert payload["stage"] == "schema"
        assert payload["errors"][0]["kind"] == "type"
        assert payload["errors"][0]["actual"] == "array"


class TestGateSchema:
    def test_missing_required_field(self) -> None:
        """② 关：缺必填字段 → kind=missing + field + expected。"""
        payload, _ = _err('{"offset": 1}')
        assert payload["stage"] == "schema"
        issue = payload["errors"][0]
        assert issue == {"field": "path", "kind": "missing", "expected": "string"}

    def test_type_mismatch_reports_actual(self) -> None:
        """② 关：类型不符 → kind=type + expected + actual。"""
        payload, _ = _err('{"path": 123}')
        issue = payload["errors"][0]
        assert issue == {
            "field": "path",
            "kind": "type",
            "expected": "string",
            "actual": "integer",
        }

    def test_enum_violation_reports_allowed(self) -> None:
        """② 关：枚举非法 → kind=enum + allowed。"""
        payload, _ = _err('{"path": "a", "mode": "turbo"}')
        issue = payload["errors"][0]
        assert issue["field"] == "mode"
        assert issue["kind"] == "enum"
        assert issue["allowed"] == ["safe", "exclusive"]

    def test_multiple_issues_reported_together(self) -> None:
        """J2：一次回填全部错误，减少往返。"""
        payload, _ = _err('{"offset": "x", "mode": "turbo"}')
        kinds = {(e["field"], e["kind"]) for e in payload["errors"]}
        assert ("path", "missing") in kinds
        assert ("offset", "type") in kinds
        assert ("mode", "enum") in kinds

    def test_extra_fields_allowed_but_reported(self) -> None:
        """J3：多余字段允许忽略，但必须能看到被忽略的字段。"""
        result = _ok('{"path": "a.txt", "extra": 1, "another": true}')
        assert result.params["path"] == "a.txt"
        assert sorted(result.ignored) == ["another", "extra"]
        assert "extra" not in result.params
        assert result.canonical["extra"] == 1  # 原始参数保留给轨迹

    def test_empty_schema_accepts_everything(self) -> None:
        result = _ok('{"anything": 1}', schema={})
        assert result.params == {"anything": 1}


class TestGateCoerce:
    def test_numeric_string_converted_to_integer(self) -> None:
        """③ 关：数字字符串转 int 成功。"""
        result = _ok('{"path": "a.txt", "offset": "12"}')
        assert result.params["offset"] == 12
        assert isinstance(result.params["offset"], int)

    def test_unconvertible_string_fails_at_coerce(self) -> None:
        """③ 关：数字字符串转 int 失败 → stage=coerce，仍是 3d。"""
        payload, _ = _err('{"path": "a.txt", "offset": "abc"}')
        assert payload["stage"] == "coerce"
        issue = payload["errors"][0]
        assert issue["field"] == "offset"
        assert issue["kind"] == "type"
        assert issue["expected"] == "integer"
        assert issue["actual"] == "string"

    def test_default_is_filled(self) -> None:
        """③ 关：缺省填充。"""
        result = _ok('{"path": "a.txt"}')
        assert result.params["offset"] == 0

    def test_boolean_string_conversion(self) -> None:
        schema = {
            "type": "object",
            "properties": {"force": {"type": "boolean"}},
            "required": ["force"],
        }
        result = _ok('{"force": "true"}', schema=schema)
        assert result.params["force"] is True

    def test_bool_is_not_integer(self) -> None:
        """JSON 语义：true 不是 integer。"""
        payload, _ = _err('{"path": "a.txt", "offset": true}')
        assert payload["stage"] == "schema"
        assert payload["errors"][0]["actual"] == "boolean"


class TestResultShape:
    def test_3d_is_never_retryable(self) -> None:
        """J4：3d 不自动重试。"""
        _, error = _err("{truncated")
        assert error.retryable is False

    def test_3d_duration_is_zero(self) -> None:
        """工具从未执行，耗时恒为 0。"""
        _, error = _err('{"offset": 1}')
        assert error.duration_ms == 0

    @pytest.mark.parametrize(
        "raw",
        [
            '{"path": "a.txt", "mode": "safe"}',
            '{"path": "a.txt", "offset": "7", "mode": "exclusive"}',
        ],
    )
    def test_success_returns_params(self, raw: str) -> None:
        result = _ok(raw)
        assert isinstance(result.params, dict)
        assert result.ignored == []


class TestParseCallArguments:
    def test_parses_object(self) -> None:
        call = ToolCall(id="c1", name="/tool/os/file/read", arguments='{"path": "a"}')
        assert parse_call_arguments(call) == {"path": "a"}

    def test_empty_arguments_is_empty_dict(self) -> None:
        call = ToolCall(id="c1", name="/tool/os/file/read", arguments="")
        assert parse_call_arguments(call) == {}

    def test_null_arguments_is_empty_dict(self) -> None:
        call = ToolCall(id="c1", name="/tool/os/file/read", arguments="null")
        assert parse_call_arguments(call) == {}

    def test_invalid_json_raises(self) -> None:
        call = ToolCall(id="c1", name="/tool/os/file/read", arguments="{oops}")
        with pytest.raises(ValueError):
            parse_call_arguments(call)

    def test_non_object_raises(self) -> None:
        call = ToolCall(id="c1", name="/tool/os/file/read", arguments="[1, 2]")
        with pytest.raises(ValueError):
            parse_call_arguments(call)
