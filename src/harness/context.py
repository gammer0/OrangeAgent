"""Context assembler (the C1/C2/C3 orchestrator, plus five-section ``SystemMsg`` assembly).

See:

* docs/02-data-structures/03-messages.md §3.3 (five sections, A0a-A0e)
* docs/01-architecture.md §4.3 (watermarks and compression), §4.5.1 (section sources)

Responsibility boundaries:

* **Is** the only place that builds a ``SystemMsg`` -- section order, separators
  and byte stability all converge here.
* **Is** the caller of C1/C2/C3: judge the watermark first, compress when
  needed, then project for the provider.
* **Is not** a tool executor and **is not** session storage.

Key invariants:

* **A0a / I1**: ``SystemMsg.content`` is byte-stable within a session. It must
  therefore be cached after the first assembly; later ``assemble`` calls reuse
  it instead of rebuilding.
* **A0b**: the working-memory body never enters ``SystemMsg``; section 1 only
  describes how to read and write it.
* **A0c**: section 3 lists the long-term memory file name only, never its path
  or contents.
* **A0d**: section 5 goes last and contains no system identifiers such as
  ``session_id``.
* **I5**: each watermark fires once (tracked by ``Watermark``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from harness.compressor import compress
from harness.discovery import TreeSessionState
from harness.message import AssistantMsg, Message, SystemMsg, ToolMsg, UserMsg
from harness.watermark import JudgeResult, Watermark, estimate_messages_tokens, judge

__all__ = [
    "SEGMENT_SEPARATOR",
    "LONG_TERM_MEMORY_NAME",
    "SystemPromptParts",
    "AssembleResult",
    "ContextAssembler",
    "DEFAULT_SYSTEM_REQUIREMENTS",
    "DEFAULT_ROLE_GUIDE",
]

#: A0a: fixed separator between the five sections.
SEGMENT_SEPARATOR = "\n\n"

#: A0c: long-term memory file name (system-wide unique; no path, no contents).
LONG_TERM_MEMORY_NAME = "MEMORY.md"

#: Section 1: system requirements (in-code constant).
#: Note: this text is model-visible, so it intentionally stays in Chinese.
DEFAULT_SYSTEM_REQUIREMENTS = """你是运行在 Harness 平台上的智能体。

## 响应格式
- 需要调用工具时，通过 tool_calls 发起调用，一次可发起多个。
- 所有工具调用都必须给出完整绝对路径（以 / 开头），不得使用缩写或相对路径。
- 不再需要工具时，直接给出最终答案文本，不要输出空的 tool_calls。

## 工作记忆
- 本项目存在一个共享工作记忆文件 .agent/WORK.md，同工作目录的多个会话共享它。
- 它不是系统提示的一部分：需要回顾时用 /tool/memory/work/read 读取，
  需要沉淀时用 /tool/memory/work/append 追加，需要整理时用 /tool/memory/work/replace 重写。
- 优先用上述专用工具；它们只能操作 WORK.md，不能读写其他文件。
- 追加是安全的（不会覆盖别人写的内容），重写要谨慎。
- 读写结果都会以工具结果的形式返回给你。""".strip()

#: Section 4: role and behavioral guidelines (in-code constant).
#: Note: this text is model-visible, so it intentionally stays in Chinese.
DEFAULT_ROLE_GUIDE = """## 角色
你是一名严谨的工程助手：先理解目标，再动手；不确定时先查证，不臆测。

## 行为准则
- 只调用与当前目标相关的工具，避免无意义的试探。
- 工具失败时先读错误信息，再决定是否换工具或换参数，不要原地重复同样的调用。
- 长输出先摘要再引用，不要把大段原始内容复述给用户。
- 不隐瞒不确定性：做不到、没验证、有风险，都要明说。""".strip()

#: Summarizer signature: ``(text) -> summary``. Injected by the caller (reusing
#: the main model channel).
Summarizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class SystemPromptParts:
    """Input material for the five ``SystemMsg`` sections.

    Attributes:
        tool_index: Section 2, the output of ``ToolTree.tree_index()``; blank when
            empty.
        memory_name: Section 3, the long-term memory file name.
        working_dir: Section 5, the absolute path of the working directory.
        work_memory_relative_path: Section 5, the working-memory relative path.
        goal: Section 5, an optional goal.
        system_requirements: Section 1, defaults to the module constant.
        role_guide: Section 4, defaults to the module constant.
    """

    tool_index: str = ""
    memory_name: str = LONG_TERM_MEMORY_NAME
    working_dir: str = ""
    work_memory_relative_path: str = ".agent/WORK.md"
    goal: str | None = None
    system_requirements: str = DEFAULT_SYSTEM_REQUIREMENTS
    role_guide: str = DEFAULT_ROLE_GUIDE

    def session_context(self) -> str:
        """Section 5: session context (working directory + work-memory path + optional goal).

        A0d: no system identifiers such as ``session_id``, tenant or creation time.
        """
        lines = [
            f"工作目录: {self.working_dir or '(未指定)'}",
            f"工作记忆: {self.work_memory_relative_path}",
        ]
        if self.goal:
            lines.append(f"本次目标: {self.goal}")
        return "\n".join(lines)

    def render(self) -> str:
        """Join the five sections in A0a's fixed order with the fixed separator.

        Empty sections 2 / 3 are filled with an empty string so the **section
        count stays constant** -- section 5's position never shifts whether or
        not the tool tree is empty, preserving prefix stability.
        """
        segments = [
            self.system_requirements,
            self.tool_index,
            self.memory_name,
            self.role_guide,
            self.session_context(),
        ]
        return SEGMENT_SEPARATOR.join(segments)


@dataclass
class AssembleResult:
    """Result of one assembly.

    Attributes:
        messages: Message list ready to send to the model (compressed, watermark
            decided).
        system: The session-level ``SystemMsg`` (cached instance, reused later).
        judged: This round's watermark decision; ``None`` when not judged (e.g.
            cache hit with nothing to decide).
        compressed: Whether compression happened.
        action: Watermark action (``normal`` / ``compress_1`` / ``compress_2`` /
            ``session_end``).
    """

    messages: list[Message]
    system: SystemMsg
    judged: JudgeResult | None
    compressed: bool
    action: str


@dataclass
class ContextAssembler:
    """Flat message assembly plus C1/C2/C3 orchestration.

    Attributes:
        capacity: Model context limit (tokens).
        watermark: Session-level watermark record; held by the caller alongside
            ``Session``.
        summarizer: Summarizer used by C3; when absent compression is skipped
            (conservative degradation, I4).
        keep_records_50: Number of CallRecords kept at the 50% watermark.
        keep_records_75: Number of CallRecords kept at the 75% watermark.
        enable_auto_compress: When disabled only watermark decisions happen, no
            compression (useful for tests and degradation).
    """

    capacity: int
    watermark: Watermark | None = None
    summarizer: Summarizer | None = None
    keep_records_50: int | None = None
    keep_records_75: int | None = None
    enable_auto_compress: bool = True
    _system: SystemMsg | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.watermark is None:
            self.watermark = Watermark(capacity=self.capacity)

    # ------------------------------------------------------------ SystemMsg

    def build_system(self, parts: SystemPromptParts) -> SystemMsg:
        """Assemble the ``SystemMsg`` (A0a: assemble once, then reuse byte-for-byte).

        A cached instance is returned directly -- even if the caller invokes this
        again after the tool tree changed, the session prefix never changes (I1).
        """
        if self._system is None:
            self._system = SystemMsg(content=parts.render())
        return self._system

    @property
    def system(self) -> SystemMsg | None:
        """The cached ``SystemMsg``; ``None`` before the first assembly."""
        return self._system

    @property
    def compress_available(self) -> bool:
        """Whether C3 compression is actually available right now.

        Note: without a summarizer compression is silently skipped (conservative
        I4 degradation); operations need this signal to notice that "compression
        never happened".
        """
        return self.enable_auto_compress and self.summarizer is not None

    def reset_system(self) -> None:
        """Discard the cached ``SystemMsg``.

        Only for a **new session** (``Session`` rebuild). Calling it mid-session
        breaks I1, so it is not exposed to the orchestration layer as a normal
        interface.
        """
        self._system = None

    # ----------------------------------------------------------- assembly

    def assemble(
        self,
        messages: Sequence[Message],
        *,
        clear_trace_state: Callable[[], None] | None = None,
    ) -> AssembleResult:
        """Judge the watermark, compress if needed, and return the messages to send.

        Args:
            messages: Session history (including the ``SystemMsg``).
            clear_trace_state: Callback invoked after compression, used to clear
                ``TreeQueryState.delivered`` and ``DiscoveredTools`` (B14 / doc §4.6).

        Returns:
            :class:`AssembleResult`; when ``action == "session_end"`` no
            compression occurs and the caller should wrap up (write traces +
            settle memory + suggest a new session).
        """
        history = list(messages)
        if not history:
            raise ValueError("messages 不得为空；请先装配 SystemMsg")

        judged = judge(history, self.capacity, watermark=self.watermark)
        action = judged.action

        if action == "session_end":
            # I5: >=85% wraps up only, no compression
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        if action == "normal" or not self.enable_auto_compress or self.summarizer is None:
            # I4: a missing optimization must degrade losslessly -- without a
            # summarizer we continue with the naive path.
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        level = 50 if action == "compress_1" else 75
        # Note: both keep fields default to None; only passing None lets compress
        # fall back to default_keep_records(). Computing the default explicitly
        # would mean "keep every record, nothing compressible".
        keep = self.keep_records_50 if level == 50 else self.keep_records_75

        try:
            compressed_messages = compress(
                history,
                level,
                self.summarizer,
                keep_records=keep,
                # Note: recording happens in assemble only -- letting compress
                # record internally would count "compressed but no effect" as used,
                # blocking the remaining watermarks from trying.
                watermark=None,
            )
        except ValueError as exc:
            # Watermark already recorded, or too few records to split: degrade
            # losslessly and never block the main flow (I4).
            # Note: keep the reason so "compression silently failed" stays observable.
            self.last_compress_error = str(exc)
            return AssembleResult(
                messages=history,
                system=_system_of(history),
                judged=judged,
                compressed=False,
                action=action,
            )

        if clear_trace_state is not None:
            clear_trace_state()

        # Note: when there are too few records to split, compress() returns
        # ``list(messages)`` -- a new shallow copy. Testing ``is not history``
        # would therefore always be True, reporting compressed repeatedly and
        # consuming watermarks; we observed an infinite loop. So decide by
        # "did the length or contents actually change".
        actually_compressed = len(compressed_messages) != len(history) or any(
            before is not after
            for before, after in zip(history, compressed_messages, strict=False)
        )
        # Note: record the watermark whether or not compression had an effect.
        # Otherwise, when compress() returns the input unchanged (too few records
        # to split), judge() keeps returning the same watermark and never reaches
        # session_end (we observed it stuck spinning on compress_1).
        self.watermark.mark_compressed(level)
        return AssembleResult(
            messages=compressed_messages,
            system=_system_of(compressed_messages),
            judged=judged,
            compressed=actually_compressed,
            action=action,
        )

    # ------------------------------------------------------- convenience

    def estimate(self, messages: Sequence[Message]) -> int:
        """Estimate the token count of a message list (C2)."""
        return estimate_messages_tokens(list(messages))

    def ratio(self, messages: Sequence[Message]) -> float:
        """Current occupancy ratio, for UI display."""
        return self.estimate(messages) / self.capacity


def _system_of(messages: Sequence[Message]) -> SystemMsg:
    """Take ``messages[0]`` and verify it is a ``SystemMsg`` (A1)."""
    head = messages[0]
    if not isinstance(head, SystemMsg):
        raise ValueError(f"A1: messages[0] must be SystemMsg, got {type(head).__name__}")
    return head


def clear_after_compress(state: TreeSessionState) -> Callable[[], None]:
    """Build a callback that clears session-level tree query state after compression.

    See B14 and design doc §4.6.

    Args:
        state: Session-level ``TreeQueryState`` + ``DiscoveredTools`` facade.

    Returns:
        A no-argument callback; calling it after compression clears both caches.
    """

    def _clear() -> None:
        state.reset()

    return _clear


def has_unclosed_tool_calls(messages: Sequence[Message]) -> bool:
    """Whether any ``tool_calls`` remain unpaired (the prerequisite check for I2 / I6)."""
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMsg):
            pending.update(call.id for call in msg.tool_calls)
        elif isinstance(msg, ToolMsg):
            pending.discard(msg.tool_call_id)
    return bool(pending)


def last_user_text(messages: Sequence[Message]) -> str:
    """Return the text of the most recent ``UserMsg`` (the query source for Jev routing)."""
    for msg in reversed(messages):
        if isinstance(msg, UserMsg):
            return msg.content
    return ""
