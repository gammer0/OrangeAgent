"""Argument validation: four gates and ``3d`` backfill.

See docs/02-data-structures/09-argument-validation.md.

| Gate | Action | Owner |
|------|--------|-------|
| ① | JSON parsing | this module |
| ② | Schema validation (required / type / enum / ignore extras) | this module |
| ③ | Type coercion and default filling | this module |
| ④ | Invoke ``fn`` | the executor; failures map to ``3c`` |

Constraints:

* J1: ``arguments`` must never be auto-repaired or completed, only parsed as-is;
* J2: the ``content`` of ``3d`` must be valid JSON containing ``stage`` and ``errors``;
* J3: extra fields may be ignored, but the upper layer must still receive
  "which fields were ignored" so it can write a ``TraceEvent``;
* J4: ``3d`` does not count toward retries (this module contains no retry logic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from orange.execution import ExecResult
from orange.message import ToolCall

__all__ = [
    "ArgIssue",
    "ArgumentValidation",
    "ValidationOutcome",
    "parse_call_arguments",
    "validate_arguments",
]

# JSON Schema type name -> model-readable spelling (fed back to the model for
# self-correction)
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
    """Return the JSON type name (``bool`` must be checked before ``int``)."""
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
    """Whether a Python value matches a JSON Schema ``type``."""
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
    return True  # an unknown type never blocks


@dataclass(frozen=True, slots=True)
class ArgIssue:
    """One self-correctable argument error (an element of J2's ``errors[]``).

    Attributes:
        field: Name of the offending field.
        kind: ``missing`` / ``type`` / ``enum``.
        expected: Expected type (for ``kind == "type"`` / ``"missing"``).
        actual: Actual type (for ``kind == "type"``).
        allowed: Allowed values (for ``kind == "enum"``).
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
    """Product of the success branch.

    Attributes:
        params: Coerced keyword arguments with defaults filled.
        canonical: The original arguments normalized to a dict (defaults not
            filled), for trace records.
        ignored: Extra fields that were ignored (J3; the upper layer writes a
            ``TraceEvent`` from this).
    """

    params: dict[str, Any]
    canonical: dict[str, Any]
    ignored: list[str] = field(default_factory=list)


ValidationOutcome = tuple[ArgumentValidation | None, ExecResult | None]
"""``(success product, 3d failure result)``; exactly one is non-``None``."""


def _fail(stage: str, errors: list[ArgIssue], **extra: Any) -> ExecResult:
    """Build the ``3d`` ``ExecResult`` (J2: ``content`` is valid JSON with stage/errors)."""
    payload: dict[str, Any] = {
        "stage": stage,
        "errors": [issue.to_dict() for issue in errors],
    }
    payload.update(extra)
    return ExecResult(
        ok=False,
        content=json.dumps(payload, ensure_ascii=False),
        error_type="3d",
        retryable=False,  # J4 / §9.3: 3d is never auto-retried
        duration_ms=0,
    )


def _parse_error_stage(raw: str, exc: json.JSONDecodeError) -> ExecResult:
    """Gate ① failure: provide the raw string and error position (never a "repaired" version)."""
    return _fail(
        "parse",
        [ArgIssue(field="<arguments>", kind="parse", expected="json")],
        raw=raw,
        position={"line": exc.lineno, "column": exc.colno, "offset": exc.pos},
        message=exc.msg,
    )


def _coerce(value: Any, expected: str) -> tuple[Any, bool]:
    """Gate ③: coerce a single value; returns ``(coerced, succeeded)``.

    Only safe literal conversions are performed; no string cleanup or other
    "auto repair" (the spirit of J1).
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
    """Whether gate ② should "hand over" this field to gate ③ (§9.1).

    Only **string literals** that need coercion (``"12"`` -> 12, ``"true"`` ->
    true) are deferred to gate ③; every other type mismatch is reported by gate
    ② with ``kind="type"``.
    """
    return isinstance(value, str) and expected in {"integer", "number", "boolean"}


def _schema_errors(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[list[ArgIssue], list[str]]:
    """Gate ②: required / type / enum checks, also collecting extra fields (J3)."""
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
            # Literals that need coercion are gate ③'s job; gate ② does not
            # report them again.
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
    """Gate ③: coercion plus default filling; returns ``(params, issues)``."""
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
    """Drop extra fields (J3: allowed and ignored, never a ``3d``)."""
    properties: dict[str, Any] = schema.get("properties") or {}
    if not properties:
        return dict(data)
    return {name: value for name, value in data.items() if name in properties}


def validate_arguments(raw: str, schema: dict[str, Any]) -> ValidationOutcome:
    """Run gates ①②③ (gate ④ is left to the executor).

    Args:
        raw: The raw JSON string of ``ToolCall.arguments``.
        schema: The tool's JSON Schema; an empty ``schema`` means "no constraints".

    Returns:
        ``(ArgumentValidation, None)`` on success, or ``(None, ExecResult)`` on
        failure; the failure is ``3d`` whose ``content`` is valid JSON with
        ``stage`` and ``errors``.

    Note:
        JSON is never auto-repaired (J1): truncation, trailing commas, single
        quotes and every other invalid input fail gate ① immediately.
    """
    schema = schema or {}

    # Gate ①: parse JSON
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

    # Gate ②: schema validation
    schema_issues, ignored = _schema_errors(data, schema)

    # Gate ③: coercion and default filling (strip extras first so they cannot
    # pollute kwargs). Both gates are merged: a coercion failure belongs to gate
    # ③ but is fed back together with gate ② errors (J2: report everything at once).
    pruned = _prune_extras(data, schema)
    params, conversion_issues = _coerce_params(pruned, schema)

    # Gates ② and ③ can both report the same type error for one field (② sees a
    # mismatch, ③ cannot convert). Duplicate entries make the model think two
    # fields are wrong, so deduplicate: keep one entry per (field, kind).
    merged: list[ArgIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in [*schema_issues, *conversion_issues]:
        key = (issue.field, issue.kind)
        if key in seen:
            continue
        seen.add(key)
        merged.append(issue)

    if merged:
        stage = "schema" if schema_issues else "coerce"
        return None, _fail(stage, merged)

    return ArgumentValidation(params=params, canonical=canonical, ignored=ignored), None


# Simplified entry point kept for the executor -------------------------------


def parse_call_arguments(call: ToolCall) -> dict[str, Any]:
    """Parse ``ToolCall.arguments`` into a dict (the executor's lightweight path).

    This path does **not** perform schema validation (the orchestration layer
    should call :func:`validate_arguments` before executing). Parse failures
    raise ``ValueError``, which the executor converges to ``3c``.
    """
    try:
        data = json.loads(call.arguments) if call.arguments.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments is not valid JSON: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"arguments must be a JSON object, got {_json_type_of(data)}"
        )
    return data


def type_name_for(expected: str) -> str:
    """Convert a JSON Schema type name into a model-readable form.

    Used for debugging and traces.
    """
    return "|".join(_TYPE_ALIASES.get(expected, (expected,)))
