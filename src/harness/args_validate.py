"""参数校验：四道关卡与 ``3d`` 回填。

对应文档：docs/02-数据结构设计/09-参数校验.md

| 关 | 动作 | 归属 |
|----|------|------|
| ① | JSON 解析 | 本模块 |
| ② | Schema 校验（必填 / 类型 / 枚举 / 多余字段忽略） | 本模块 |
| ③ | 类型转换与缺省填充 | 本模块 |
| ④ | 调用 ``fn`` | 执行器，失败归 ``3c`` |

约束：

* J1：``arguments`` 不得被自动修复或补全，只能原样解析；
* J2：``3d`` 的 ``content`` 必须是合法 JSON，含 ``stage`` 与 ``errors``；
* J3：多余字段允许忽略，但必须让上层拿到"被忽略的字段"以便写 ``TraceEvent``；
* J4：``3d`` 不单独计数重试（本模块不含重试逻辑）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harness.execution import ExecResult
from harness.message import ToolCall

__all__ = [
    "ArgIssue",
    "ArgumentValidation",
    "ValidationOutcome",
    "parse_call_arguments",
    "validate_arguments",
]

# JSON Schema 类型名 → 模型可读写法（回填给模型看，用于自我修正）
_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "string": ("string", "str"),
    "integer": ("integer", "int"),
    "number": ("number", "float", "int"),
    "boolean": ("boolean", "bool"),
    "array": ("array", "list"),
    "object": ("object", "dict"),
    "null": ("null", "none"),
}


def _json_type_of(value: Any) -> str:
    """返回 JSON 类型名（``bool`` 必须先于 ``int`` 判断）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    """判断 Python 值是否符合 JSON Schema 的 ``type``。"""
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True  # 未知 type 不阻断


@dataclass(frozen=True, slots=True)
class ArgIssue:
    """单条可自我修正的参数错误（J2 的 ``errors[]`` 元素）。

    Attributes:
        field: 出错字段名。
        kind: ``missing`` / ``type`` / ``enum``。
        expected: 期望类型（``kind == "type"`` / ``"missing"``）。
        actual: 实际类型（``kind == "type"``）。
        allowed: 允许取值（``kind == "enum"``）。
    """

    field: str
    kind: str
    expected: str | None = None
    actual: str | None = None
    allowed: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"field": self.field, "kind": self.kind}
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        if self.allowed is not None:
            payload["allowed"] = list(self.allowed)
        return payload


@dataclass(frozen=True, slots=True)
class ArgumentValidation:
    """成功分支的产物。

    Attributes:
        params: 已转换、已填缺省的关键字参数。
        canonical: 统一为 dict 后的原始参数（未填缺省），用于轨迹记录。
        ignored: 被忽略的多余字段（J3，上层据此写 ``TraceEvent``）。
    """

    params: dict[str, Any]
    canonical: dict[str, Any]
    ignored: list[str] = field(default_factory=list)


ValidationOutcome = tuple[ArgumentValidation | None, ExecResult | None]
"""``(成功产物, 3d 失败结果)``，二者恰有一个非 ``None``。"""


def _fail(stage: str, errors: list[ArgIssue], **extra: Any) -> ExecResult:
    """构造 ``3d`` 的 ``ExecResult``（J2：``content`` 为合法 JSON，含 stage/errors）。"""
    payload: dict[str, Any] = {
        "stage": stage,
        "errors": [issue.to_dict() for issue in errors],
    }
    payload.update(extra)
    return ExecResult(
        ok=False,
        content=json.dumps(payload, ensure_ascii=False),
        error_type="3d",
        retryable=False,  # J4 / §9.3：3d 不自动重试
        duration_ms=0,
    )


def _parse_error_stage(raw: str, exc: json.JSONDecodeError) -> ExecResult:
    """① 关失败：提供原始字符串与错误位置（不回填"修复后"的内容）。"""
    return _fail(
        "parse",
        [ArgIssue(field="<arguments>", kind="parse", expected="json")],
        raw=raw,
        position={"line": exc.lineno, "column": exc.colno, "offset": exc.pos},
        message=exc.msg,
    )


def _coerce(value: Any, expected: str) -> tuple[Any, bool]:
    """③ 关：单值类型转换；返回 ``(转换后值, 是否成功)``。

    只做安全的字面量转换，不做字符串清洗等"自动修复"（J1 精神）。
    """
    if _matches_type(value, expected):
        return value, True

    if expected == "integer" and isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10), True
        except ValueError:
            return value, False

    if expected == "number" and isinstance(value, str):
        text = value.strip()
        try:
            return float(text), True
        except ValueError:
            return value, False

    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true", True
        return value, False

    return value, False


def _defer_to_coerce(value: Any, expected: Any) -> bool:
    """② 关是否需要把该字段"让给"③ 关（§9.1）。

    只有**字符串字面量**需要类型转换（``"12"`` → 12、``"true"`` → true）时
    才延后到 ③ 关；其余类型不符一律在 ② 关报 ``kind="type"``。
    """
    return isinstance(value, str) and expected in {"integer", "number", "boolean"}


def _schema_errors(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[list[ArgIssue], list[str]]:
    """② 关：必填 / 类型 / 枚举检查；同时收集多余字段（J3）。"""
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    issues: list[ArgIssue] = []

    for name in required:
        if name not in data:
            prop = properties.get(name) or {}
            issues.append(
                ArgIssue(
                    field=name,
                    kind="missing",
                    expected=str(prop.get("type", "any")),
                )
            )

    for name, prop in properties.items():
        if name not in data:
            continue
        value = data[name]
        expected = prop.get("type")
        expected_types = [expected] if isinstance(expected, str) else list(expected or [])
        if expected_types and not any(_matches_type(value, t) for t in expected_types):
            # 需要类型转换的字面量属于 ③ 关职责，② 关不重复报告
            if _defer_to_coerce(value, expected):
                continue
            issues.append(
                ArgIssue(
                    field=name,
                    kind="type",
                    expected="|".join(expected_types),
                    actual=_json_type_of(value),
                )
            )
            continue
        enum = prop.get("enum")
        if enum is not None and value not in enum:
            issues.append(ArgIssue(field=name, kind="enum", allowed=list(enum)))

    ignored = [name for name in data if name not in properties]
    return issues, ignored


def _coerce_params(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[ArgIssue]]:
    """③ 关：类型转换 + 缺省填充；返回 ``(params, issues)``。"""
    properties: dict[str, Any] = schema.get("properties") or {}
    params = dict(data)
    issues: list[ArgIssue] = []

    for name, prop in properties.items():
        expected = prop.get("type")
        if isinstance(expected, list):
            expected = expected[0] if expected else None

        if name in params and isinstance(expected, str):
            converted, ok = _coerce(params[name], expected)
            if not ok:
                issues.append(
                    ArgIssue(
                        field=name,
                        kind="type",
                        expected=expected,
                        actual=_json_type_of(params[name]),
                    )
                )
                continue
            params[name] = converted

        if name not in params and "default" in prop:
            params[name] = prop["default"]

    if issues:
        return None, issues
    return params, issues


def _prune_extras(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """丢弃多余字段（J3：允许并忽略，不报 3d）。"""
    properties: dict[str, Any] = schema.get("properties") or {}
    if not properties:
        return dict(data)
    return {name: value for name, value in data.items() if name in properties}


def validate_arguments(raw: str, schema: dict[str, Any]) -> ValidationOutcome:
    """执行 ①②③ 三道关卡（第 ④ 关交由执行器）。

    Args:
        raw: ``ToolCall.arguments`` 原始 JSON 字符串。
        schema: 工具定义的 JSON Schema；``schema`` 为空时按"无约束"处理。

    Returns:
        ``(ArgumentValidation, None)`` 成功，或 ``(None, ExecResult)`` 失败；
        失败结果为 ``3d``，``content`` 是含 ``stage`` 与 ``errors`` 的合法 JSON。

    Note:
        不自动修复 JSON（J1）：截断、尾逗号、单引号等一切非法输入直接判 ① 关失败。
    """
    schema = schema or {}

    # ① JSON 解析
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, _parse_error_stage(raw, exc)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        return None, _fail(
            "schema",
            [ArgIssue(field="<root>", kind="type", expected="object", actual=_json_type_of(data))],
        )

    canonical = dict(data)

    # ② Schema 校验
    schema_issues, ignored = _schema_errors(data, schema)

    # ③ 类型转换与缺省填充（多余字段先剔除，避免污染 kwargs）
    # 三关合并：类型转换失败属于 ③ 关，但仍与 ② 关的错误一起回填（J2：一次给全）
    pruned = _prune_extras(data, schema)
    params, conversion_issues = _coerce_params(pruned, schema)
    if schema_issues or params is None:
        stage = "schema" if schema_issues else "coerce"
        return None, _fail(stage, [*schema_issues, *conversion_issues])

    return ArgumentValidation(params=params, canonical=canonical, ignored=ignored), None


# 兼容执行器的简化入口 -------------------------------------------------------


def parse_call_arguments(call: ToolCall) -> dict[str, Any]:
    """把 ``ToolCall.arguments`` 解析为 dict（执行器的轻量路径）。

    该路径**不做** schema 校验（编排层应在执行前调用 :func:`validate_arguments`）。
    解析失败时抛 ``ValueError``，由执行器统一收敛为 ``3c``。
    """
    try:
        data = json.loads(call.arguments) if call.arguments.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"arguments 必须是 JSON 对象，实际为 {_json_type_of(data)}"
        )
    return data


def type_name_for(expected: str) -> str:
    """把 JSON Schema 类型名转成模型可读的写法（调试与轨迹使用）。"""
    return "|".join(_TYPE_ALIASES.get(expected, (expected,)))
