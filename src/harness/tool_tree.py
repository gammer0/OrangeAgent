"""Tool tree and flat store (ADT-4/5).

See ``docs/02-data-structures/04-tools-and-permissions.md``.

Core model: **the tree is a view, the store is flat** (B8).

::

    FlatStore (single source of truth, key = tool_id)
        │  aggregate by path segments
        ▼
    ToolTree (derived view, never persisted)

Constraint ownership:

===========  ==========================================================
B1           ``name`` is unique among siblings
B2           Every node's ``tool_id`` must exist in FlatStore (including
             build-time synthesized entries)
B3           A ``leaf`` node's ``schema`` equals its ``ToolDef.schema``
B4           An ``abstract`` node's ``schema`` is an empty object
B5           Denied nodes never appear in the tree index or ``tree_query``
B6           The tree index defaults to L1+L2; beyond 40 L2 entries keep 40 by
             ascending description length
B7           A ``path`` already in ``delivered`` never receives its schema again
B8           FlatStore is the single source of truth; ToolTree is not persisted
B9           description is a single line
B10          L1 <= 40 characters, deeper levels <= 80
B11          concurrency is safe only when caps contains read alone
B12          idempotent defaults to false
B13          Every abstract node has concurrency=safe and idempotent=true
B17          Permission.src matches the node's resource kind
B18          side_effect_scope never participates in permission decisions
B19          The first path segment is in {tool, skill, mcp}
B20          Intermediate nodes without an explicit entry get a synthesized
             ToolDef that is not written into FlatStore
B21          kind is not stored; it is derived at build time
B22          Validate the whole tree after building; any failure rejects it
===========  ==========================================================
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from harness.permission import (
    AgentPermissions,
    CapName,
    Permission,
    check_src_matches_path,
    path_head,
    path_segments,
    visible,
)

if TYPE_CHECKING:  # pragma: no cover
    from harness.discovery import TreeSessionState

__all__ = [
    "EMPTY_SCHEMA",
    "RESOURCE_ROOTS",
    "Concurrency",
    "SideEffectScope",
    "ToolDef",
    "ToolNode",
    "ToolTree",
    "FlatStore",
    "ToolTreeError",
    "tree_query",
    "build_tree",
    "derive_concurrency",
    "truncate_description",
    "index_text_to_paths",
]

# An abstract node's schema is always an empty object (B4)
EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

# Only these three L1 resource roots are allowed (B19)
RESOURCE_ROOTS: frozenset[str] = frozenset({"tool", "skill", "mcp"})

Concurrency = Literal["safe", "exclusive"]
SideEffectScope = Literal["session", "project", "global"]

L1_DESCRIPTION_LIMIT = 40
DEEP_DESCRIPTION_LIMIT = 80
#: Default index depth: ``0`` means unlimited (fully expanded).
#:
#: It used to default to 2, but tool trees are often four levels deep
#: (``/tool/<group>/<subgroup>/<tool>``), so the model only saw directory names
#: like ``/tool/fluid`` and could not see which tools existed at all -- dropping
#: the single most important fact: what tools are available. We measured full
#: expansion costing only ~400 extra tokens, far below the cost of the model
#: probing around repeatedly.
DEFAULT_INDEX_DEPTH = 0

#: Placeholder description for synthesized (abstract) nodes.
#:
#: An earlier version wrote this straight into the index, so every intermediate
#: node dragged along the same six characters -- 95% of the index size at
#: depth=1, pure noise. It is now an internal marker only, cleared at the end of
#: tree building (unless the group has an explicit description).
#:
#: Note: this string is model-visible when a group lacks a description, so it
#: intentionally stays in Chinese.
SYNTHETIC_DESCRIPTION = "由子节点推断"


class ToolTreeError(ValueError):
    """Tool-tree construction or validation failed.

    Attributes:
        rule: The violated constraint identifier, e.g. ``"B1"``.
        detail: Human-readable description.
    """

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")


# ---------------------------------------------------------------------------
# Derivation rules (B11 / B12 / B10)
# ---------------------------------------------------------------------------


def derive_concurrency(caps: Sequence[str]) -> Concurrency:
    """Derive ``concurrency`` per §4.13 (B11).

    * ``["read"]`` / ``["read", "net"]`` -> ``safe``
    * contains ``write`` / ``exec``, or a lone ``["net"]``, or undeclared ->
      ``exclusive``
    """
    if not caps:
        return "exclusive"
    if "write" in caps or "exec" in caps:
        return "exclusive"
    if "read" in caps:
        return "safe"
    return "exclusive"


def truncate_description(description: str, limit: int) -> str:
    """Truncate a description by character count and append an ellipsis (B10)."""
    if len(description) <= limit:
        return description
    return description[: limit - 1] + "…"


def _reject_multiline(description: str) -> str:
    """B9: description must be a single line."""
    if "\n" in description or "\r" in description:
        raise ValueError(f"B9: description must be single-line, found a newline: {description!r}")
    return description


def _limit_for_path(path: str) -> int:
    """L1 (``/tool``) allows 40 characters, deeper levels 80 (B10)."""
    depth = len(path_segments(path))
    return L1_DESCRIPTION_LIMIT if depth <= 1 else DEEP_DESCRIPTION_LIMIT


# ---------------------------------------------------------------------------
# ToolDef (ADT-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDef:
    """One tool definition in the flat store (ADT-4, fields exactly per doc §4.2).

    A dataclass rather than pydantic: the doc requires the field name ``schema``,
    and pydantic's ``BaseModel`` reserves a member of that name, causing a naming
    conflict.

    Derivation and validation:

    * ``concurrency`` is derived from ``caps`` when omitted (B11); a supplied but
      inconsistent value is rejected;
    * ``idempotent`` defaults to ``False`` (B12); a ``write``/``exec`` combination
      must never be marked ``True``;
    * ``permission.src`` must match the first path segment (B17);
    * the first ``path`` segment must be in ``{tool, skill, mcp}`` (B19);
    * ``description`` must be single-line (B9).
    """

    tool_id: str
    path: str
    fn: Callable[..., Any] | None = None
    schema: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_SCHEMA))
    concurrency: Concurrency | None = None
    idempotent: bool = False
    caps: list[CapName] = field(default_factory=list)
    permission: Permission = field(
        default_factory=lambda: Permission(src="tool")
    )
    side_effect_scope: SideEffectScope = "session"
    description: str = ""
    synthetic: bool = False
    retired: bool = False

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise ValueError("tool_id must not be empty")

        # B19: the first path segment must be a resource root
        head = path_head(self.path)
        if head is None:
            raise ValueError(f"B16/B19: invalid absolute path {self.path!r}")
        if head not in RESOURCE_ROOTS:
            raise ValueError(
                f"B19: the first path segment must be in {{tool, skill, mcp}}, got {head!r}"
            )

        # B17: src matches the first path segment
        if self.permission.src != head:
            raise ValueError(
                f"B17: Permission.src={self.permission.src!r} does not match first segment {head!r}"
            )

        # B9: single line
        _reject_multiline(self.description)

        derived_concurrency = derive_concurrency(self.caps)
        if self.concurrency is None:
            object.__setattr__(self, "concurrency", derived_concurrency)
        elif self.concurrency != derived_concurrency:
            raise ValueError(
                f"B11: caps={self.caps} derives concurrency {derived_concurrency!r}, "
                f"which conflicts with the declared {self.concurrency!r}"
            )

        # B12: a ``write`` / ``exec`` combination must never be idempotent=true
        if self.idempotent and not self.synthetic:
            mutating = {"write", "exec"} & set(self.caps)
            if mutating:
                raise ValueError(
                    f"B12: caps={self.caps} contains {sorted(mutating)}, so idempotent=true is not allowed"
                )

    def with_retired(self, retired: bool = True) -> ToolDef:
        """Return a copy with the ``retired`` flag changed (B23: unload marks, never deletes)."""
        return dataclasses.replace(self, retired=retired)

    # ------------------------------------------------------- factory methods

    @classmethod
    def synthesized(cls, path: str, parent: ToolDef | None = None) -> ToolDef:
        """Synthesize a ``ToolDef`` for an intermediate node per §4.9.2 (B20).

        Synthesized entries exist only in the build-time tree and are **never
        written into FlatStore**.

        Args:
            path: Intermediate node path.
            parent: Parent entry; ``permission`` and ``side_effect_scope`` are
                inherited from it (B17).
        """
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
        head = path_head(path)
        if head not in RESOURCE_ROOTS:
            raise ToolTreeError("B19", f"cannot synthesize an entry for {path!r}: invalid first segment")
        return cls(
            tool_id=f"t_tree_{digest}",
            path=path,
            fn=None,  # the scheduler binds tree_query(node) at execution time
            schema=dict(EMPTY_SCHEMA),
            concurrency="safe",  # B13
            idempotent=True,  # B13
            caps=["read"],
            permission=parent.permission if parent is not None else Permission(src=head),
            side_effect_scope=parent.side_effect_scope if parent else "session",
            description=SYNTHETIC_DESCRIPTION,
            synthetic=True,
        )


# ---------------------------------------------------------------------------
# FlatStore (ADT-4)
# ---------------------------------------------------------------------------


class FlatStore:
    """Flat store: the single source of truth, ``entries`` keyed by ``tool_id`` (B8).

    Each entry's ``path`` must map one-to-one with its key: one path can never
    have two entries, otherwise the "last segment binds a single tool_id"
    semantics break.
    """

    def __init__(self, entries: Iterable[ToolDef] | None = None) -> None:
        self._entries: dict[str, ToolDef] = {}
        for entry in entries or ():
            self.put(entry)

    # -- queries -----------------------------------------------------

    @property
    def entries(self) -> Mapping[str, ToolDef]:
        """Read-only view: ``tool_id -> ToolDef``."""
        return _ReadOnlyMapping(self._entries)

    def get(self, tool_id: str) -> ToolDef | None:
        """Look up an entry by ``tool_id``."""
        return self._entries.get(tool_id)

    def by_path(self, path: str) -> ToolDef | None:
        """Look up an entry by full path (used by ``from_path`` resolution)."""
        for entry in self._entries.values():
            if entry.path == path:
                return entry
        return None

    def active_paths(self) -> dict[str, ToolDef]:
        """Non-retired entries indexed by path (``retired=False``)."""
        return {e.path: e for e in self._entries.values() if not e.retired}

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ToolDef]:
        return iter(list(self._entries.values()))

    # -- mounting (§4.10.1) -----------------------------------------

    def put(self, entry: ToolDef) -> ToolDef:
        """Write a single entry.

        Raises:
            ToolTreeError: Violates B19 (invalid first segment), B9/B10 (invalid
                description) or a path conflict.
        """
        head = path_head(entry.path)
        if head is None or head not in RESOURCE_ROOTS:
            raise ToolTreeError(
                "B19",
                f"refusing to mount: the first path segment must be in "
                f"{{tool, skill, mcp}}, got {entry.path!r}",
            )
        _check_description(entry.description, entry.path)
        existing = self._entries.get(entry.tool_id)
        if existing is not None:
            raise ToolTreeError("B8", f"tool_id already exists: {entry.tool_id!r}")
        for other in self._entries.values():
            if other.path == entry.path and not other.retired:
                raise ToolTreeError(
                    "B1", f"path is already taken: {entry.path!r} (tool_id={other.tool_id!r})"
                )
        self._entries[entry.tool_id] = entry
        return entry

    def mark_retired(self, path_prefix: str, *, exact: bool = False) -> list[str]:
        """Mark every entry under a prefix as ``retired=True`` (B23: never delete).

        Args:
            path_prefix: Absolute path prefix.
            exact: When ``True``, mark that path alone.

        Returns:
            The list of marked ``tool_id`` values.
        """
        touched: list[str] = []
        for tool_id, entry in list(self._entries.items()):
            if entry.retired:
                continue
            hit = entry.path == path_prefix if exact else (
                entry.path == path_prefix or entry.path.startswith(path_prefix + "/")
            )
            if hit:
                self._entries[tool_id] = entry.with_retired(True)
                touched.append(tool_id)
        return touched


class _ReadOnlyMapping(Mapping[str, ToolDef]):
    """A read-only wrapper around a dict so callers cannot pollute the source of truth."""

    def __init__(self, data: dict[str, ToolDef]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> ToolDef:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _check_description(description: str, path: str) -> None:
    """B9 single-line plus B10 length (over-long is truncated during assembly; only newlines are rejected here)."""
    del path
    _reject_multiline(description)


# ---------------------------------------------------------------------------
# ToolNode（ADT-5）
# ---------------------------------------------------------------------------


@dataclass
class ToolNode:
    """A derived-view node; never stored independently.

    ``kind`` is a **derived property**, not a constructor argument (B21): having
    ``children`` makes it ``abstract``, otherwise ``leaf``.
    """

    path: str
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""
    children: list[ToolNode] = field(default_factory=list)
    permission: Permission | None = None
    side_effect_scope: SideEffectScope = "session"

    # ------------------------------------------------- structural operations

    def child(self, name: str) -> ToolNode | None:
        """Look up a direct child by ``name`` (B1 guarantees uniqueness)."""
        for node in self.children:
            if node.name == name:
                return node
        return None

    def add_child(self, node: ToolNode) -> None:
        """Append a child; duplicate names are caught by validation before :meth:`sort_children`."""
        self.children.append(node)

    def sort_children(self) -> None:
        """Re-sort the whole subtree by ``path`` lexicographically (B6 requires an ordered index)."""
        self.children.sort(key=lambda n: n.path)
        for node in self.children:
            node.sort_children()

    def walk(self) -> Iterator[ToolNode]:
        """Pre-order traversal of the whole subtree (including self)."""
        yield self
        for node in self.children:
            yield from node.walk()

    # ---------------------------------------------------- derived properties

    @property
    def kind(self) -> Literal["abstract", "leaf"]:
        """B21: children -> abstract, otherwise leaf."""
        return "abstract" if self.children else "leaf"

    @property
    def is_leaf(self) -> bool:
        """Whether this is a concrete tool."""
        return not self.children

    @property
    def depth(self) -> int:
        """Path depth (``/`` is 0)."""
        return 0 if self.path == "/" else len(path_segments(self.path))

    def child_index(self) -> list[dict[str, Any]]:
        """Child summaries without schema (B7: used when the schema was already delivered)."""
        return [
            {
                "path": child.path,
                "name": child.name,
                "kind": child.kind,
                "description": child.description,
            }
            for child in self.children
        ]

    def full_index(self) -> list[dict[str, Any]]:
        """Full child structure including schema (§4.5 response format)."""
        return [
            {**summary, "schema": child.schema}
            for summary, child in zip(self.child_index(), self.children, strict=True)
        ]


# ---------------------------------------------------------------------------
# ToolTree (ADT-5)
# ---------------------------------------------------------------------------


@dataclass
class ToolTree:
    """A view derived from FlatStore; never persisted independently (B8).

    Attributes:
        root: Root node ``/``.
        version: Incremented on every change (B25).
        permissions: Permission table used at build time; still applied when
            indexing and querying (B5).
        flat_store: Back-reference for ``tool_id`` validation and execution-time
            resolution (B2).
        tree_query_fn: The ``tree_query`` implementation shared by every abstract
            entry (§4.5).
        synthetic_entries: Entries synthesized at build time (indexed by path,
            never written into FlatStore, B20).
    """

    root: ToolNode = field(default_factory=lambda: ToolNode(path="/", name=""))
    version: int = 1
    permissions: AgentPermissions = field(
        default_factory=lambda: AgentPermissions(agent_id="local")
    )
    flat_store: FlatStore = field(default_factory=FlatStore)
    tree_query_fn: Callable[..., Any] | None = None
    synthetic_entries: dict[str, ToolDef] = field(default_factory=dict)

    # ------------------------------------------------------------- queries

    def get_node(self, path: str) -> ToolNode | None:
        """Find a node by absolute path; ``None`` when absent or invisible (B5)."""
        if path == "/":
            node = self.root
        else:
            if not visible(path, self.permissions):
                return None
            node = self.root
            for segment in path_segments(path):
                found = node.child(segment)
                if found is None:
                    return None
                node = found
            # If an intermediate node is denied, the whole subtree is invisible
        return node if visible(node.path, self.permissions) else None

    def visible_nodes(self, depth: int | None = None) -> list[ToolNode]:
        """List visible nodes lexicographically by ``path``, optionally depth-limited."""
        return [
            node
            for node in self.root.walk()
            if node.path != "/" and visible(node.path, self.permissions)
            and (depth is None or node.depth <= depth)
        ]

    def tree_index(self, depth: int = DEFAULT_INDEX_DEPTH) -> str:
        """Tree index projection (B6): expand to ``depth``, ordered by path.

        Visibility rules (**a group must justify its own value**):

        * **Leaves** (tools/skills that can actually be called) always appear.
        * **Intermediate nodes** appear only when they have an **explicit
          description** -- only a description carries information beyond the leaf
          paths (what the group is for). An undescribed intermediate node merely
          repeats the common prefix of its children, which is pure noise, so it is
          omitted.
        * Exception: the L1 resource roots (``/tool``, ``/skill``, ``/mcp``) always
          appear; they mark the boundary between the three capability kinds.

        Other rules:

        * Denied nodes never appear (B5).
        * ``depth`` means **how many levels to expand** (1 = L1 only; ``0`` =
          unlimited).
        * The result depends only on tree structure and permissions, so it is
          byte-identical within a session.

        No entry cap: tool groups are few (tens of tools, three or four levels
        deep), so a complete index is already within a reasonable token budget.
        A low cap would hide some tools and force the model to go hunting -- a net
        loss.
        """
        ordered = self.visible_nodes()
        if depth and depth > 0:
            shown = [n for n in ordered if n.depth <= depth]
        else:
            shown = list(ordered)

        # Groups are kept only when described; leaves and L1 roots are always kept
        shown = [
            n
            for n in shown
            if n.is_leaf or n.depth == 1 or (n.description or "").strip()
        ]

        lines = []
        for node in shown:
            description = (node.description or "").strip()
            if description:
                limit = _limit_for_path(node.path)
                lines.append(f"{node.path} — {truncate_description(description, limit)}")
            else:
                lines.append(node.path)

        # Nodes hidden purely by the depth limit -- tell the model to expand from
        # the parent path
        depth_hidden = [
            n for n in ordered if depth and depth > 0 and n.depth > depth
        ]
        if depth_hidden:
            lines.append(
                f"（另有 {len(depth_hidden)} 个更深节点，用 tree_query 按上级路径展开）"
            )
        return "\n".join(lines)

    @property
    def paths(self) -> list[str]:
        """Every visible node path (lexicographic)."""
        return [n.path for n in self.visible_nodes()]

    # -------------------------------------------------- execution resolution

    def resolve(self, path: str) -> ToolDef:
        """``path -> ToolNode -> tool_id -> ToolDef`` (the §4.5 call chain).

        Raises:
            PermissionDenied: The path is denied (B5; returns ``3a``).
            KeyError: The path does not exist.
        """
        from harness.permission import PermissionDenied

        if not visible(path, self.permissions):
            raise PermissionDenied(path)
        node = self.get_node(path)
        if node is None:
            raise KeyError(f"tool path does not exist: {path!r}")
        entry = self.flat_store.get(node.tool_id) or self.synthetic_entries.get(node.path)
        if entry is None:
            raise KeyError(f"tool_id={node.tool_id!r} is not in FlatStore (B2)")
        return entry

    # ----------------------------------------------------- mutation (B25)

    def bump(self) -> None:
        """``version += 1`` (B25)."""
        self.version += 1


# ---------------------------------------------------------------------------
# tree_query (§4.5 / §4.6 / §4.7)
# ---------------------------------------------------------------------------


def tree_query(
    node: ToolNode | str,
    *,
    tree: ToolTree,
    state: TreeSessionState | None = None,
    step: int = 0,
) -> dict[str, Any]:
    """Query the immediate children of an abstract node (§4.5).

    * Returns ``{"path": ..., "children": [...]}``, with visible children only (B5).
    * When ``state`` is given: "check delivered + write delivered + upsert
      DiscoveredTools" all happen inside one lock (B7 / B14).
    * Children already in ``delivered`` omit ``schema`` (B7).
    * This function keeps no state of its own; thread safety is provided by the
      :class:`TreeSessionState` lock.

    Args:
        node: A node object or an absolute path.
        tree: The owning tool tree (for permission filtering and state binding
            validation).
        state: Session-level query state (optional; when omitted the full schema
            is delivered every time).
        step: Current step number, written into ``first_seen_step`` /
            ``last_used_step``.
    """
    target: ToolNode
    if isinstance(node, str):
        found = tree.get_node(node)
        if found is None:
            from harness.permission import PermissionDenied

            # The node is not in the tree: it may be denied or simply absent.
            # An explicitly denied path returns 3a per B5 instead of leaking the
            # "exists or not" difference.
            if node in tree.permissions.grants and tree.permissions.grants[node] is False:
                raise PermissionDenied(node)
            raise KeyError(f"abstract node does not exist or is invisible: {node!r}")
        target = found
    else:
        target = node
        # Prevent callers from smuggling in an arbitrary node object to bypass
        # permissions (B5)
        checked = tree.get_node(target.path)
        if checked is None:
            from harness.permission import PermissionDenied

            raise PermissionDenied(target.path)
        target = checked

    if state is None:
        # Without session state the full schema is sent every time; permission
        # filtering happens before full_index.
        visible_children = [
            child for child in target.children if visible(child.path, tree.permissions)
        ]
        return {
            "path": target.path,
            "children": [
                {
                    "path": child.path,
                    "name": child.name,
                    "kind": child.kind,
                    "description": child.description,
                    "schema": child.schema,
                }
                for child in visible_children
            ],
        }

    return state.query_and_record(tree, target, step)


# ---------------------------------------------------------------------------
# Build (§4.9)
# ---------------------------------------------------------------------------


def build_tree(
    flat_store: FlatStore,
    permissions: AgentPermissions | None = None,
    *,
    group_descriptions: Mapping[str, str] | None = None,
) -> ToolTree:
    """Build the tool tree from the flat store per §4.9.1.

    Algorithm:

    1. Iterate non-retired (``retired=False``) entries lexicographically by ``path``;
    2. Permission filtering (B5) -- denied entries and their subtrees never enter
       the tree;
    3. Walk the path segments with ``get_or_create`` on intermediate nodes;
    4. Bind ``tool_id`` / ``schema`` / ``permission`` on the final segment;
    5. Synthesize a ``ToolDef`` for intermediate nodes with no explicit entry
       (B20; never written into FlatStore);
    6. Derive ``kind`` (B21) and validate the whole tree (B22).

    Args:
        flat_store: The flat store (single source of truth).
        permissions: Permission table; an empty table when omitted (all allowed).
        group_descriptions: ``group_id -> description``, e.g.
            ``{"tool/os": "local files"}``. Intermediate nodes prefer this
            description; when absent it stays empty (the index shows the path only
            and costs no tokens).

    Raises:
        ToolTreeError: Any build-time validation (B1/B2/B3/B4/B9/B10/B19) failed;
            the tree is rejected.
    """
    perms = permissions or AgentPermissions(agent_id="local")
    tree = ToolTree(
        root=ToolNode(path="/", name=""),
        version=1,
        permissions=perms,
        flat_store=flat_store,
        tree_query_fn=tree_query,
    )

    active = sorted(flat_store.active_paths().values(), key=lambda e: e.path)

    # ---- 1) Pre-validation: path format and description (B19 / B9 / B10 / B16) ----
    for entry in active:
        _validate_entry_metadata(entry)

    # ---- 2) Segment aggregation -----------------------------------------------
    bound: dict[str, ToolDef] = {}
    for entry in active:
        if not perms.is_allowed(entry.path):
            continue  # B5: a denied entry never enters the tree (exact check, B16)
        node = tree.root
        segments = path_segments(entry.path)
        for i, seg in enumerate(segments):
            child_path = "/" + "/".join(segments[: i + 1])
            existing = node.child(seg)
            if existing is None:
                existing = ToolNode(path=child_path, name=seg)
                node.add_child(existing)
            node = existing
        if node.path in bound and bound[node.path].tool_id != entry.tool_id:
            raise ToolTreeError(
                "B1",
                f"path {node.path!r} is bound to multiple tool_ids: "
                f"{bound[node.path].tool_id!r} and {entry.tool_id!r}",
            )
        node.schema = entry.schema
        node.tool_id = entry.tool_id
        node.description = entry.description
        node.permission = entry.permission
        node.side_effect_scope = entry.side_effect_scope
        bound[node.path] = entry

    tree.root.sort_children()

    # ---- 3) Synthesize intermediate nodes (B20) -------------------------------
    synthetic = _synthesize_prefixes(
        active,
        bound,
        perms,
        existing_paths=[node.path for node in tree.root.walk()],
    )

    # ---- 4) Normalize abstract schemas (B4) and truncate descriptions (B10) ---
    #        Intermediate nodes with no explicit entry receive a synthetic
    #        tool_id here (B2 / B20).
    for node in tree.root.walk():
        if node.path == "/":
            node.permission = Permission(src="tool", caps=["read"])
            node.schema = dict(EMPTY_SCHEMA)
            node.description = ""
            continue
        if node.path not in bound and node.path in synthetic:
            synth = synthetic[node.path]
            node.tool_id = synth.tool_id
            # B17: a synthesized node inherits its parent's Permission
            node.permission = synth.permission
            node.side_effect_scope = synth.side_effect_scope
            node.description = synth.description
        if node.kind == "abstract":
            node.schema = dict(EMPTY_SCHEMA)
            # Group descriptions prefer the explicit declaration; when absent it
            # stays empty and the index shows the path only
            declared = (group_descriptions or {}).get(node.path.lstrip("/"))
            if declared:
                node.description = declared
            elif node.description == SYNTHETIC_DESCRIPTION:
                node.description = ""
        node.description = truncate_description(node.description, _limit_for_path(node.path))

    # Synthesized entries live only in the tree's private view (never in
    # FlatStore, B20), used by resolve and validation.
    tree.synthetic_entries = synthetic

    _validate_tree(tree, flat_store, synthetic)
    return tree


def _validate_entry_metadata(entry: ToolDef) -> None:
    """B19 / B9 / B10 / B16: validate one entry's metadata."""
    head = path_head(entry.path)
    if head is None:
        raise ToolTreeError("B16", f"invalid absolute path: {entry.path!r}")
    if head not in RESOURCE_ROOTS:
        raise ToolTreeError(
            "B19",
            f"the first path segment must be in {{tool, skill, mcp}}, got {head!r}: {entry.path!r}",
        )
    if "\n" in entry.description or "\r" in entry.description:
        raise ToolTreeError("B9", f"description must be single-line: {entry.path!r}")
    if len(entry.description) > _limit_for_path(entry.path):
        # B10: over-long descriptions are truncated during assembly; not rejected here
        return


def _synthesize_prefixes(
    active: list[ToolDef],
    bound: dict[str, ToolDef],
    permissions: AgentPermissions,
    *,
    existing_paths: Iterable[str] = (),
) -> dict[str, ToolDef]:
    """Synthesize a ``ToolDef`` for intermediate nodes with no explicit entry (B20); returns path -> entry.

    Synthesized entries inherit ``permission`` / ``side_effect_scope`` from their
    parent (B17); an L1 node with no parent entry uses the resource-root default.
    Invisible prefixes are never synthesized (B5): denying an intermediate node
    hides the whole subtree.

    Args:
        active: Every non-retired entry.
        bound: path -> entry for paths bound to an explicit entry (already
            permission-filtered).
        permissions: The permission decider.
        existing_paths: Every path present in the tree (used to fill in L1 roots, B2).
    """
    synthetic: dict[str, ToolDef] = {}
    # Synthesis is driven by **nodes actually present in the tree**, not by bound
    # (denied entries also leave parent-chain nodes behind).
    targets = sorted({p for p in existing_paths if p != "/"})
    for path in targets:
        segments = path_segments(path)
        parent_entry: ToolDef | None = None
        for i in range(1, len(segments) + 1):
            prefix = "/" + "/".join(segments[:i])
            if prefix in bound:
                parent_entry = bound[prefix]
                continue
            if prefix not in synthetic:
                base = parent_entry or _resource_root_entry("/" + segments[0])
                synthetic[prefix] = ToolDef.synthesized(prefix, base)
            parent_entry = synthetic[prefix]
    return synthetic


def _resource_root_entry(path: str) -> ToolDef:
    """Synthesis base for an L1 intermediate node with no parent entry."""
    head = path_head(path)
    if head not in RESOURCE_ROOTS:
        raise ToolTreeError("B19", f"cannot synthesize an L1 node: {path!r}")
    return ToolDef(
        tool_id=f"t_root_{head}",
        path=path,
        schema=dict(EMPTY_SCHEMA),
        concurrency="safe",
        idempotent=True,
        caps=["read"],
        permission=Permission(src=head, caps=["read"]),  # type: ignore[arg-type]
        description={
            "tool": "操作系统能力",
            "skill": "可加载的技能包",
            "mcp": "外部 MCP 服务",
        }[head],
        synthetic=True,
    )


def _validate_tree(
    tree: ToolTree,
    store: FlatStore,
    synthetic: dict[str, ToolDef],
) -> None:
    """Whole-tree build-time validation (B22). Any failure raises ``ToolTreeError``."""
    # Callers may omit synthetic (e.g. structural-only validation); fall back to
    # the tree's own synthesized entries.
    if not synthetic:
        synthetic = tree.synthetic_entries

    for node in tree.root.walk():
        if node.path == "/":
            continue

        # B1: names are unique among siblings (checked first so B2 cannot mask it)
        names = [child.name for child in node.children]
        if len(set(names)) != len(names):
            duplicated = sorted({n for n in names if names.count(n) > 1})
            raise ToolTreeError("B1", f"duplicate child names under {node.path!r}: {duplicated}")

        # B2: tool_id must exist in FlatStore or the synthetic table
        if not node.tool_id:
            raise ToolTreeError("B2", f"node is missing tool_id: {node.path!r}")
        entry = store.get(node.tool_id) or synthetic.get(node.path)
        if entry is None or entry.tool_id != node.tool_id:
            raise ToolTreeError(
                "B2",
                f"node {node.path!r} has tool_id={node.tool_id!r} which is not in FlatStore",
            )

        # B4: an abstract schema is an empty object
        if node.kind == "abstract":
            if node.schema != EMPTY_SCHEMA:
                raise ToolTreeError(
                    "B4", f"abstract node {node.path!r} must have an empty schema object"
                )
        # B3: a leaf schema matches ToolDef.schema
        else:
            if node.schema != entry.schema:
                raise ToolTreeError(
                    "B3", f"leaf node {node.path!r} schema does not match ToolDef.schema"
                )

        # B17: permission.src matches the first path segment
        if node.permission is None:
            raise ToolTreeError("B17", f"node {node.path!r} is missing permission")
        try:
            check_src_matches_path(node.permission.src, node.path)
        except ValueError as exc:
            raise ToolTreeError("B17", str(exc)) from exc

        # B9 / B10
        if "\n" in node.description or "\r" in node.description:
            raise ToolTreeError("B9", f"description must be single-line: {node.path!r}")

    # Synthesized entries must never reach FlatStore (B20); each must also carry
    # synthetic=True.
    for path, entry in synthetic.items():
        if any(e.path == path and not e.retired for e in store):
            # A path with an explicit entry is never synthesized; reaching here
            # means the synthesis logic is wrong.
            raise ToolTreeError("B20", f"synthesized entry conflicts with FlatStore: {path!r}")
        if not entry.synthetic:
            raise ToolTreeError("B20", f"synthesized entry must set synthetic=True: {path!r}")


# ---------------------------------------------------------------------------
# Index parsing helpers
# ---------------------------------------------------------------------------


def index_text_to_paths(text: str) -> list[str]:
    """Extract paths from ``tree_index`` text (for tests and replay).

    Handles both line shapes: ``path — description`` and a bare ``path`` (for
    nodes without a description). Hint lines starting with ``（`` contain no path
    and are skipped.

    Historical bug: an earlier version parsed only lines containing `` — ``, so
    paths without a description (L1 resource roots, undescribed groups) were
    silently dropped and assertions built on it reached wrong conclusions.
    """
    paths: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("（"):
            continue
        path = stripped.split(" — ", 1)[0].strip()
        if path.startswith("/"):
            paths.append(path)
    return paths
