"""C1 ResultTruncator: truncation of oversized results.

See docs/01-architecture.md §4.3 (C1).

Trigger: a single tool result longer than ``max_chars`` characters.
Action: keep the **first and last halves** and insert an omission notice in
the middle; the notice must state how many characters were omitted.

Design notes:

* The notice itself consumes budget -- the final length **never exceeds**
  ``max_chars``.
* When under the threshold the content is returned **byte-for-byte** (the very
  same ``str`` object), with no normalization applied.
* Truncation is the single entry point before writing a ``ToolMsg``, so this
  module only handles text and is unaware of message types.
"""

from __future__ import annotations

__all__ = ["DEFAULT_MAX_CHARS", "truncate_result"]

#: Default value for the design document's ``MAX_OBS_CHARS``.
DEFAULT_MAX_CHARS = 4000

#: Omission-notice template; ``{n}`` is the number of omitted characters.
#: Note: this text is model-visible, so it intentionally stays in Chinese.
_OMISSION_TEMPLATE = "\n\n[... {n} 字符已省略 ...]\n\n"


def _notice(omitted: int) -> str:
    """Render the omission notice."""
    return _OMISSION_TEMPLATE.format(n=omitted)


def truncate_result(content: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Truncate to ``max_chars`` by keeping both halves.

    Args:
        content: Original tool result text.
        max_chars: Maximum result length (including the omission notice).

    Returns:
        A new string of length ``<= max_chars``; returns ``content`` unchanged
        when ``len(content) <= max_chars``.

    Raises:
        ValueError: ``max_chars`` is smaller than the minimum notice length and
            cannot accommodate the notice.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    if len(content) <= max_chars:
        # Below the threshold: return byte-for-byte unchanged
        return content

    # The lower bound of omitted characters depends on max_chars, so first
    # estimate the notice length assuming exactly one character is omitted.
    minimal_notice = _notice(1)
    if len(minimal_notice) > max_chars:
        raise ValueError(
            f"max_chars={max_chars} cannot fit the omission notice "
            f"(needs at least {len(minimal_notice)})"
        )

    total = len(content)
    # Total budget for head+tail = limit - notice length. The notice length
    # changes with the digit count of the omitted number, so iterate to a fixed
    # point.
    omitted = total
    head_count = 0
    tail_count = 0
    for _ in range(8):
        notice_len = len(_notice(omitted))
        budget = max_chars - notice_len
        if budget < 0:  # pragma: no cover - minimal notice already verified above
            raise ValueError(f"max_chars={max_chars} is too small")
        head_count = budget // 2
        tail_count = budget - head_count
        new_omitted = total - head_count - tail_count
        if new_omitted == omitted:
            break
        omitted = new_omitted

    head = content[:head_count]
    tail = content[total - tail_count :] if tail_count else ""
    return head + _notice(omitted) + tail
