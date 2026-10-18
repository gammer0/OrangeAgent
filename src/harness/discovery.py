"""Discovered-tools inventory and tree-query session state (ADT-6/7).

See ``docs/02-data-structures/04-tools-and-permissions.md`` §4.6 / §4.7.

Constraint ownership:

===========  ==========================================================
B5           ``TreeSessionState.query_and_record`` returns visible children only
B7           A ``path`` already in ``delivered`` must not receive its schema again
B14          Reads and writes of ``TreeQueryState`` and ``DiscoveredTools``
             must share one lock
C3 (caps)    Context compression clears both ``delivered`` and the inventory
             via ``reset``
===========  ==========================================================

``TreeSessionState`` is the only concurrency-safe facade this module exposes:
every read and write of ``delivered`` and ``DiscoveredTools`` goes through it
and the lock never escapes to callers, which guarantees B14 structurally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from harness.permission import visible

if TYPE_CHECKING:  # pragma: no cover
    from harness.tool_tree import ToolNode, ToolTree

__all__ = [
    "DEFAULT_MAX_DISCOVERED",
    "ToolDiscovery",
    "DiscoveredTools",
    "TreeQueryState",
    "TreeSessionState",
]

# Design doc §4.2: max_entries = 50
DEFAULT_MAX_DISCOVERED = 50


@dataclass
class ToolDiscovery:
    """One record in the inventory (design doc §4.2).

    Attributes:
        path: Full absolute path.
        description: One-line description, <= 80 characters.
        kind: ``abstract`` / ``leaf``.
        first_seen_step: Step at which ``tree_query`` first returned it.
        last_used_step: Step at which it was last returned or called
            (the eviction basis, §4.7).
    """

    path: str
    description: str
    kind: Literal["abstract", "leaf"]
    first_seen_step: int
    last_used_step: int

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError(f"path must be absolute: {self.path!r}")
        if "\n" in self.description or "\r" in self.description:
            raise ValueError(f"description must be a single line: {self.path!r}")
        if len(self.description) > 80:
            self.description = self.description[:79] + "…"

    def touch(self, step: int) -> None:
        """Update ``last_used_step`` (§4.7 "use")."""
        self.last_used_step = step

    def to_row(self) -> dict[str, str]:
        """Project into an element of the temporary system message (message §3.7)."""
        return {"path": self.path, "description": self.description}


@dataclass
class DiscoveredTools:
    """Session-level discovered-tools inventory, keyed by ``path`` in ``entries``.

    Attributes:
        entries: ``path -> ToolDiscovery``.
        max_entries: Capacity limit, default 50.
    """

    entries: dict[str, ToolDiscovery] = field(default_factory=dict)
    max_entries: int = DEFAULT_MAX_DISCOVERED

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, path: object) -> bool:
        return path in self.entries

    def get(self, path: str) -> ToolDiscovery | None:
        """Look up a record by path."""
        return self.entries.get(path)

    def upsert(self, discovery: ToolDiscovery) -> None:
        """Insert or update one record, then evict by capacity (§4.7).

        * Already present: keep ``first_seen_step`` and update
          ``last_used_step`` (taking the larger value).
        * Over ``max_entries``: evict in ascending ``last_used_step`` order,
          keeping the new record (a new record's ``last_used_step`` is the
          current step, so it is normally not evicted immediately).
        """
        existing = self.entries.get(discovery.path)
        if existing is not None:
            existing.description = discovery.description
            existing.kind = discovery.kind
            existing.last_used_step = max(existing.last_used_step, discovery.last_used_step)
            return
        self.entries[discovery.path] = discovery
        self._evict()

    def _evict(self) -> None:
        """Evict in ascending ``last_used_step`` order until within ``max_entries``."""
        overflow = len(self.entries) - self.max_entries
        if overflow <= 0:
            return
        ordered = sorted(
            self.entries.values(),
            key=lambda d: (d.last_used_step, d.first_seen_step, d.path),
            reverse=True,
        )
        keep = {d.path for d in ordered[: self.max_entries]}
        self.entries = {p: d for p, d in self.entries.items() if p in keep}

    def touch(self, path: str, step: int) -> bool:
        """Update ``last_used_step`` when the model calls a path.

        Returns:
            ``True`` if the path is in the inventory.
        """
        entry = self.entries.get(path)
        if entry is None:
            return False
        entry.touch(step)
        return True

    def remove_prefix(self, path_prefix: str) -> list[str]:
        """Remove every record under a prefix on unload (B24)."""
        removed = [
            p for p in self.entries
            if p == path_prefix or p.startswith(path_prefix + "/")
        ]
        for path in removed:
            del self.entries[path]
        return removed

    def clear(self) -> None:
        """Clear the inventory (during context compression, in sync with ``delivered``)."""
        self.entries.clear()

    def rows(self) -> list[dict[str, str]]:
        """Project into temporary system-message entries, descending by ``last_used_step``."""
        ordered = sorted(
            self.entries.values(),
            key=lambda d: (d.last_used_step, d.first_seen_step, d.path),
            reverse=True,
        )
        return [d.to_row() for d in ordered]


@dataclass
class TreeQueryState:
    """Session-level record: paths whose schema has already been fully delivered (§4.2 / §4.6).

    **Never read or write this object across threads directly** -- go through
    :class:`TreeSessionState` (B14).

    Attributes:
        delivered: Paths already returned with a full schema **as a child node**
            (per design doc §4.6).
        expanded: Abstract-node paths that have been **expanded by a query**.
            Expanding implies the node's own schema is already visible (it must
            previously have been returned as some parent's child), so its schema
            can be omitted when it appears as a child again. This set is not in
            the documented fields; it exists to satisfy both B7 (schema sent
            once) and B14 (``delivered`` records children only).
    """

    delivered: set[str] = field(default_factory=set)
    expanded: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.delivered)

    def __contains__(self, path: object) -> bool:
        return path in self.delivered

    def mark(self, path: str) -> None:
        """Record that a path's schema has been delivered (B7)."""
        self.delivered.add(path)

    def mark_expanded(self, path: str) -> None:
        """Record that a node has been expanded by a query (its schema is visible)."""
        self.expanded.add(path)

    def was_exposed(self, path: str) -> bool:
        """Whether this path's schema was ever delivered (the B7 entry point)."""
        return path in self.delivered or path in self.expanded

    def clear(self) -> None:
        """Clear on context compression (design doc §4.6)."""
        self.delivered.clear()
        self.expanded.clear()

    def discard_prefix(self, path_prefix: str) -> int:
        """Remove every path under a prefix on unload (B24); returns the count removed."""
        victims = {
            p for p in self.delivered
            if p == path_prefix or p.startswith(path_prefix + "/")
        }
        self.delivered -= victims
        self.expanded -= {
            p for p in self.expanded
            if p == path_prefix or p.startswith(path_prefix + "/")
        }
        return len(victims)


class TreeSessionState:
    """Same-lock facade over ``TreeQueryState`` + ``DiscoveredTools`` (B14).

    All reads and writes happen inside one ``threading.Lock``, which lets
    ``tree_query`` be marked ``concurrency="safe"`` (design doc §4.6). The lock
    object is private and never leaks.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_DISCOVERED,
        state: TreeQueryState | None = None,
        discovered: DiscoveredTools | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        self._lock = lock or threading.Lock()
        self.state = state or TreeQueryState()
        self.discovered = discovered or DiscoveredTools(max_entries=max_entries)

    # ------------------------------------------------------------ properties

    @property
    def delivered(self) -> frozenset[str]:
        """Read-only snapshot of ``delivered`` (diagnostics)."""
        with self._lock:
            return frozenset(self.state.delivered)

    # ---------------------------------------------------------------- core

    def query_and_record(
        self,
        tree: ToolTree,
        node: ToolNode,
        step: int = 0,
        *,
        depth: int | None = None,
    ) -> dict[str, Any]:
        """Query children, write ``delivered`` and upsert the inventory -- **one lock** (B14).

        Args:
            tree: Owning tool tree (the permission filter is re-applied inside
                the lock, B5).
            node: Target abstract node.
            step: Current step number.
            depth: Reserved parameter, currently unused (children always return
                one level).

        Returns:
            ``{"path": ..., "children": [...]}``; children already in
            ``delivered`` omit ``schema`` (B7).
        """
        del depth
        with self._lock:
            children = [
                child
                for child in node.children
                if visible(child.path, tree.permissions)  # B5
            ]
            # B7: if the child was already returned by a tree query (its schema
            # was delivered), omit the schema. Both the check and the write
            # happen inside this lock (B14).
            payload: list[dict[str, Any]] = []
            for child in children:
                already_delivered = self.state.was_exposed(child.path)
                item: dict[str, Any] = {
                    "path": child.path,
                    "name": child.name,
                    "kind": child.kind,
                    "description": child.description,
                }
                if not already_delivered:
                    item["schema"] = child.schema

                self.discovered.upsert(
                    ToolDiscovery(
                        path=child.path,
                        description=child.description,
                        kind=child.kind,
                        first_seen_step=step,
                        last_used_step=step,
                    )
                )
                payload.append(item)

            # Record **each child** as soon as it is returned: next time it is
            # queried as a parent, or appears as another parent's child, its
            # schema can be omitted (B7).
            for child in children:
                self.state.mark(child.path)
            # This node has now been expanded, so its schema is never repeated (B7)
            self.state.mark_expanded(node.path)

            return {"path": node.path, "children": payload}

    # ------------------------------------------------------------------ use

    def record_use(self, path: str, step: int) -> bool:
        """Update ``last_used_step`` when the model calls a path (§4.7)."""
        with self._lock:
            return self.discovered.touch(path, step)

    # -------------------------------------------------------------- cleanup

    def reset(self) -> None:
        """Clear ``delivered`` and the inventory on compression -- under one lock (doc §4.7)."""
        with self._lock:
            self.state.clear()
            self.discovered.clear()

    def purge_prefix(self, path_prefix: str) -> None:
        """Clean every session cache under a prefix on unload (B24)."""
        with self._lock:
            self.state.discard_prefix(path_prefix)
            self.discovered.remove_prefix(path_prefix)

    # ----------------------------------------------------------- projection

    def discovered_rows(self) -> list[dict[str, str]]:
        """Projection of the discovered-tools inventory (for ``to_provider(discovered=...)``)."""
        with self._lock:
            return self.discovered.rows()
