"""统一异常定义。

本模块只定义异常类型，不含处置逻辑（处置见 docs/01-系统架构设计.md §4.9）。
消息不变量校验失败统一走 ``MessageInvariantError``，其 ``rule`` 字段对应
docs/02-数据结构设计/03-消息.md 中的约束编号。
"""

from __future__ import annotations


class HarnessError(Exception):
    """所有 harness 异常的基类。"""


class MessageInvariantError(HarnessError):
    """消息序列不变量被违反（A1–A14）。

    Attributes:
        rule: 约束编号，如 ``"A3"``。
        detail: 人可读的违规描述。
    """

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")
