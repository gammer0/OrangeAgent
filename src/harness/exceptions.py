"""Unified exception definitions.

This module defines exception types only; it contains no handling logic
(see docs/01-architecture.md §4.9 for handling).
Message invariant violations always go through ``MessageInvariantError``;
its ``rule`` field maps to constraint identifiers in the message design doc.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all harness exceptions."""


class MessageInvariantError(HarnessError):
    """A message-sequence invariant was violated (A1-A14).

    Attributes:
        rule: Constraint identifier, e.g. ``"A3"``.
        detail: Human-readable violation description.
    """

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")
