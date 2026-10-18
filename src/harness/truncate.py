"""C1 ResultTruncator：大结果截断。

对应文档：docs/01-系统架构设计.md §4.3（C1）

触发条件：单条工具结果长度 > ``max_chars`` 字符。
动作：保留**头尾各半** + 中间插入省略提示，提示中必须包含被省略的字符数。

设计要点：

* 提示本身也占额度——最终结果长度**严格不超过** ``max_chars``。
* 未超过阈值时**逐字节原样返回**（同一 ``str`` 对象），不做任何规范化。
* 截断是写 ``ToolMsg`` 前的唯一入口，因此这里只处理文本，不感知消息类型。
"""

from __future__ import annotations

__all__ = ["DEFAULT_MAX_CHARS", "truncate_result"]

#: 设计文档 ``MAX_OBS_CHARS`` 的默认值。
DEFAULT_MAX_CHARS = 4000

#: 省略提示的模板；``{n}`` 为被省略的字符数。
_OMISSION_TEMPLATE = "\n\n[... {n} 字符已省略 ...]\n\n"


def _notice(omitted: int) -> str:
    """生成省略提示。"""
    return _OMISSION_TEMPLATE.format(n=omitted)


def truncate_result(content: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """把头尾各半截断到 ``max_chars`` 以内。

    Args:
        content: 原始工具结果文本。
        max_chars: 结果长度上限（含省略提示）。

    Returns:
        长度 ``<= max_chars`` 的新字符串；``len(content) <= max_chars`` 时原样返回。

    Raises:
        ValueError: ``max_chars`` 小于省略提示的最小长度，无法容纳提示。
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars 必须为正数，实际为 {max_chars}")

    if len(content) <= max_chars:
        # 短于阈值：逐字节原样返回
        return content

    # 被省略的字符数下限受 max_chars 约束，先按最少 1 估算提示长度
    minimal_notice = _notice(1)
    if len(minimal_notice) > max_chars:
        raise ValueError(
            f"max_chars={max_chars} 无法容纳省略提示（至少需要 {len(minimal_notice)}）"
        )

    total = len(content)
    # 头尾可用的总额度 = 上限 - 提示长度；提示长度随省略数字位数变化，迭代收敛即可
    omitted = total
    head_count = 0
    tail_count = 0
    for _ in range(8):
        notice_len = len(_notice(omitted))
        budget = max_chars - notice_len
        if budget < 0:  # pragma: no cover - 上文已保证最小提示可容纳
            raise ValueError(f"max_chars={max_chars} 过小")
        head_count = budget // 2
        tail_count = budget - head_count
        new_omitted = total - head_count - tail_count
        if new_omitted == omitted:
            break
        omitted = new_omitted

    head = content[:head_count]
    tail = content[total - tail_count :] if tail_count else ""
    return head + _notice(omitted) + tail
