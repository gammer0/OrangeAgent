"""消息模型（ADT-3）。

对应文档：docs/02-数据结构设计/03-消息.md

类型层次::

    Message (ABC)
    ├── SystemMsg        messages[0]，五段装配产物，会话内逐字节不变
    ├── UserMsg          用户文本，不得为空
    ├── AssistantMsg     模型回复，content 与 tool_calls 不得同时为空
    └── ToolMsg          工具结果，必须匹配某个 ToolCall.id
    ToolCall             独立值对象，不继承 Message

不变量归属：

* 单条构造期约束（A7/A8/A9）→ ``__post_init__``
* 跨消息序列约束（A1/A2/A3/A10）→ ``validate_messages``
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.exceptions import MessageInvariantError

__all__ = [
    "Message",
    "SystemMsg",
    "UserMsg",
    "AssistantMsg",
    "ToolMsg",
    "ToolCall",
    "to_provider",
    "validate_messages",
]

# 路径段之间以 / 分隔；ToolCall.name 必须是完整绝对路径（A10）
_PATH_SEGMENT = r"[A-Za-z0-9_\-.]+"
_ABSOLUTE_PATH_PATTERN = rf"^/(?:{_PATH_SEGMENT})(?:/{_PATH_SEGMENT})*$"


class _Base(BaseModel):
    """所有消息的公共基类：严格模式 + 禁止额外字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)

    # 判别字段，序列化时写入但不必要地暴露给 provider（投影时手工构造 dict）
    role: str


class SystemMsg(_Base):
    """SystemMsg：会话中仅一条，且必须是 ``messages[0]``（A1）。

    五段组成见文档 §3.3，装配由 ``ContextAssembler`` 负责，本类只承载结果。
    """

    role: Literal["system"] = "system"
    content: str


class UserMsg(_Base):
    """UserMsg：用户输入，不得为空串（A9）。"""

    role: Literal["user"] = "user"
    content: Annotated[str, Field(min_length=1)]


class ToolCall(BaseModel):
    """ToolCall：provider 返回的调用请求（独立值对象，非 Message 子类型）。

    ``arguments`` 保持原始 JSON 字符串，不提前解析（J1）；解析与校验见
    docs/02-数据结构设计/09-参数校验.md。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(pattern=_ABSOLUTE_PATH_PATTERN)]
    arguments: str


class AssistantMsg(_Base):
    """AssistantMsg：模型回复。``is_final`` 为派生只读属性（A7/A8）。"""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: 思考型模型的推理过程（DeepSeek ``reasoning_content``）。
    #: ponytail: 该字段**必须原样回传**给供应商，否则同一会话后续请求会 400
    #: （"reasoning_content in the thinking mode must be passed back"）。
    #: 它不参与历史语义，也不进入展示，只做透传。
    reasoning_content: str | None = None

    @property
    def is_final(self) -> bool:
        """是否为本轮最终答案。

        ponytail: 必须同时要求 ``content`` 非空白。只判 ``tool_calls`` 为空
        会把模型的空回复（``content=""`` + 无调用）误认为「已完成」，
        导致任务被静默截断——实测出现过写了一半就退出、理由为空的情况。
        """
        return not self.tool_calls and bool((self.content or "").strip())

    @model_validator(mode="after")
    def _check_content_or_calls(self) -> AssistantMsg:
        if self.content is None and not self.tool_calls:
            raise ValueError("A7: content 与 tool_calls 不得同时为空")
        return self


class ToolMsg(_Base):
    """ToolMsg：工具执行结果。``name`` 为内部字段（工具树路径），不投影（A4/A6）。"""

    role: Literal["tool"] = "tool"
    tool_call_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(pattern=_ABSOLUTE_PATH_PATTERN)]
    content: str


Message = Annotated[
    SystemMsg | UserMsg | AssistantMsg | ToolMsg,
    Field(discriminator="role"),
]
"""消息联合类型，用于列表声明与重新解析。"""


def validate_messages(messages: list[Message]) -> None:
    """校验跨消息的序列级约束。

    对应约束：A1（首条为 SystemMsg 且唯一）、A2（tool_call_id 可匹配）、
    A3（tool_calls 与 ToolMsg 等量配对）、A10（路径为绝对路径，构造期已保证）。
    单条构造期约束（A7/A8/A9）不在此重复检查。

    Args:
        messages: 待校验的会话消息列表。

    Raises:
        MessageInvariantError: 违反任一序列级约束，``rule`` 为对应编号。
    """
    if not messages:
        raise MessageInvariantError("A1", "messages 不得为空，且首条必须是 SystemMsg")

    if not isinstance(messages[0], SystemMsg):
        raise MessageInvariantError(
            "A1", f"messages[0] 必须是 SystemMsg，实际为 {type(messages[0]).__name__}"
        )

    for index, msg in enumerate(messages[1:], start=1):
        if isinstance(msg, SystemMsg):
            raise MessageInvariantError(
                "A1", f"index {index} 出现了第二个 SystemMsg；会话中仅允许一条"
            )

    pending: dict[str, str] = {}
    for index, msg in enumerate(messages):
        if isinstance(msg, AssistantMsg) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.id in pending:
                    raise MessageInvariantError(
                        "A2", f"index {index} 的 tool_call id 重复: {call.id}"
                    )
                pending[call.id] = call.name
            continue

        if isinstance(msg, ToolMsg):
            expected_name = pending.pop(msg.tool_call_id, None)
            if expected_name is None:
                raise MessageInvariantError(
                    "A2",
                    f"index {index} 的 tool_call_id={msg.tool_call_id!r} 无法匹配前序 ToolCall",
                )
            if expected_name != msg.name:
                raise MessageInvariantError(
                    "A6",
                    f"index {index} 的 ToolMsg.name={msg.name!r} 与 ToolCall.name="
                    f"{expected_name!r} 不一致",
                )

    if pending:
        missing = ", ".join(f"{cid}@{name}" for cid, name in pending.items())
        raise MessageInvariantError("A3", f"存在未配对的 tool_calls（缺少 ToolMsg）: {missing}")


def to_provider(
    messages: list[Message],
    discovered: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """把消息投影为 provider 请求体。

    投影规则见文档 §3.6；``discovered`` 非空时在**末尾**追加一条临时 ``system``
    消息（A11–A13），该条不写入 ``messages``，且不传 ``discovered`` 时结果逐字节
    相同（保 KV cache）。

    Args:
        messages: 会话消息列表。
        discovered: 已发现工具清单，元素形如 ``{"path": ..., "description": ...}``。

    Returns:
        可直接作为 ``chat.completions`` 的 ``messages`` 参数使用的新列表。
    """
    projected: list[dict[str, object]] = []

    for msg in messages:
        if isinstance(msg, (SystemMsg, UserMsg)):
            projected.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, AssistantMsg):
            item: dict[str, object] = {"role": "assistant", "content": msg.content}
            # ponytail: 仅在存在时回传，保证无推理内容的供应商请求体逐字节不变
            if msg.reasoning_content:
                item["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in msg.tool_calls
                ]
            projected.append(item)
        elif isinstance(msg, ToolMsg):
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
        else:  # pragma: no cover - 联合类型已封闭
            raise TypeError(f"未知消息类型: {type(msg).__name__}")

    if discovered:
        lines = ["已发现工具："]
        lines.extend(f"{item['path']} — {item['description']}" for item in discovered)
        projected.append({"role": "system", "content": "\n".join(lines)})

    return projected
