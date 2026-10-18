"""C1 ResultTruncator 测试。

覆盖：短内容原样返回、长内容头尾各半、结果长度上限、省略字符数提示。
"""

from __future__ import annotations

import re

import pytest

from harness.truncate import truncate_result

_OMITTED = re.compile(r"\[\.\.\. (\d+) 字符已省略 \.\.\.\]")
#: 4000 额度下省略提示的完整形态（含数字 6024）
_NOTICE_FULL = "[... 6024 字符已省略 ...]"


def _omitted_of(content: str) -> int:
    match = _OMITTED.search(content)
    assert match is not None, "缺少省略提示"
    return int(match.group(1))


class TestShortContent:
    def test_short_content_returned_verbatim(self) -> None:
        content = "hello world"
        assert truncate_result(content, 4000) == content

    def test_exact_threshold_returned_verbatim(self) -> None:
        content = "x" * 100
        assert truncate_result(content, 100) == content

    def test_short_content_keeps_objects_identity(self) -> None:
        content = "{\"a\": 1}\n\temoji: 😀"
        assert truncate_result(content, 4000) is content

    def test_empty_content(self) -> None:
        assert truncate_result("", 10) == ""

    def test_no_omission_notice_for_short_content(self) -> None:
        assert "省略" not in truncate_result("abc", 3)


class TestLongContent:
    def test_long_content_keeps_head_and_tail(self) -> None:
        content = "HEAD" + "m" * 5000 + "TAIL"
        out = truncate_result(content, 4000)
        assert out.startswith("HEAD")
        assert out.endswith("TAIL")

    def test_long_content_within_limit(self) -> None:
        for max_chars in (100, 999, 4000):
            content = "a" * (max_chars * 3)
            assert len(truncate_result(content, max_chars)) <= max_chars

    def test_head_and_tail_are_balanced(self) -> None:
        content = "".join(chr(ord("a") + i % 26) for i in range(10_000))
        out = truncate_result(content, 4000)

        head, _, rest = out.partition("\n\n[...")
        tail = rest.split("...]\n\n", 1)[1]
        # 头尾各半（奇数额度时头比尾少 1）
        assert abs(len(head) - len(tail)) <= 1
        assert head == content[: len(head)]
        assert tail == content[len(content) - len(tail) :]

    def test_omission_count_is_exact(self) -> None:
        content = "a" * 10_000
        out = truncate_result(content, 4000)
        match = _OMITTED.search(out)
        assert match is not None
        omitted = int(match.group(1))
        # 保留的原文 = 提示前后文本，各需扣掉模板自带的 2 个换行分隔符
        head = out[: match.start()].removesuffix("\n\n")
        tail = out[match.end() :].removeprefix("\n\n")
        retained = len(head) + len(tail)
        assert retained == 4000 - len(_NOTICE_FULL) - 4
        assert omitted == len(content) - retained

    def test_hint_is_present_and_counts_characters(self) -> None:
        content = "z" * 5000
        out = truncate_result(content, 1000)
        match = _OMITTED.search(out)
        assert match is not None
        head = out[: match.start()].removesuffix("\n\n")
        tail = out[match.end() :].removeprefix("\n\n")
        assert int(match.group(1)) == len(content) - len(head) - len(tail)

    def test_unicode_content_length_bounded(self) -> None:
        content = "汉字" * 3000
        out = truncate_result(content, 500)
        assert len(out) <= 500
        assert out.startswith("汉字")
        assert out.endswith("汉字")

    def test_threshold_just_below_max_still_truncates(self) -> None:
        content = "a" * 4001
        out = truncate_result(content, 4000)
        assert len(out) <= 4000
        assert "省略" in out


class TestInvalidArguments:
    def test_non_positive_max_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            truncate_result("abc", 0)

    def test_max_chars_smaller_than_notice_raises(self) -> None:
        with pytest.raises(ValueError):
            truncate_result("a" * 100, 5)
