"""Permission model (DAC).

See ``docs/02-data-structures/04-tools-and-permissions.md`` §4.3.

Model highlights:

* A single principal (the agent) plus three resource kinds (tool / skill / mcp),
  analogous to Linux owner bits.
* Grants are at **instance** granularity: keys in ``grants`` are absolute
  resource paths.
* ``grants`` stores exceptions only; **no record means allowed** (B15).
* Decisions depend on the path only; ``side_effect_scope`` is not consulted (B18).

Constraint ownership:

===========  ==========================================================
B5           The visibility check is shared by ``tree_index`` and ``tree_query``
B15          ``is_allowed`` returns True for paths with no record
B16          ``grants`` keys must be full absolute paths matching ``ToolNode.path`` exactly
B17          ``Permission.src`` must match the node's actual resource kind
B18          Permission decisions ignore ``side_effect_scope``
===========  ==========================================================
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CapName",
    "ResourceKind",
    "Permission",
    "AgentIdentity",
    "AgentPermissions",
    "is_absolute_path",
    "path_segments",
    "path_head",
    "check_src_matches_path",
    "PermissionDenied",
]

# Four capability bits: read / write / exec / net
CapName = Literal["read", "write", "exec", "net"]

# Resource kind: determines which permission bit applies (B17)
ResourceKind = Literal["tool", "skill", "mcp"]

RESOURCE_KINDS: frozenset[str] = frozenset({"tool", "skill", "mcp"})

# Path segment charset shared with message.py so the two definitions cannot drift
_PATH_SEGMENT = r"[A-Za-z0-9_\-.]+"
_ABSOLUTE_PATH_RE = re.compile(rf"^/(?:{_PATH_SEGMENT})(?:/{_PATH_SEGMENT})*$")
_HEAD_RE = re.compile(rf"^/({_PATH_SEGMENT})")


def is_absolute_path(path: str) -> bool:
    """Whether this is a valid absolute path (leading ``/``, at least one segment, no empties).

    B16 requires ``grants`` keys to be full absolute paths matching
    ``ToolNode.path`` exactly. This function only validates the format;
    character-exact consistency is guaranteed by how lookups are done (a dict
    lookup is itself an exact match).
    """
    return isinstance(path, str) and bool(_ABSOLUTE_PATH_RE.match(path))


def path_segments(path: str) -> list[str]:
    """Split an absolute path into segments; ``"/"`` yields an empty list.

    Raises:
        ValueError: The path format is invalid.
    """
    if path == "/":
        return []
    if not is_absolute_path(path):
        raise ValueError(f"invalid absolute path: {path!r}")
    return path.strip("/").split("/")


def path_head(path: str) -> str | None:
    """Return the first segment (the L1 resource root); ``None`` if not absolute."""
    match = _HEAD_RE.match(path)
    return match.group(1) if match else None


def check_src_matches_path(src: str, path: str) -> None:
    """Verify ``Permission.src`` matches the first segment of the node path (B17).

    Raises:
        ValueError: Mismatch or invalid path.
    """
    head = path_head(path)
    if head is None:
        raise ValueError(f"B16: invalid absolute path {path!r}")
    if head != src:
        raise ValueError(
            f"B17: Permission.src={src!r} does not match path head {head!r}: {path!r}"
        )


class Permission(BaseModel):
    """Resource kind and capability bits declared by a node.

    Attributes:
        src: Resource kind; determines the permission bit (tool->call,
            skill->read, mcp->request).
        caps: Capability bits, used for concurrency inference (B11), idempotency
            tagging and risk hints.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    src: ResourceKind
    caps: list[CapName] = Field(default_factory=list)

    @field_validator("caps")
    @classmethod
    def _caps_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"caps contains duplicates: {value}")
        return value

    @property
    def permission_bit(self) -> str:
        """Permission bit name for this resource kind (design doc §4.3)."""
        return {"tool": "call", "skill": "read", "mcp": "request"}[self.src]


class AgentIdentity(BaseModel):
    """Multi-agent placeholder: single instance for now, ``role`` is always ``owner``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    role: Literal["owner"] = "owner"


class AgentPermissions(BaseModel):
    """Per-agent permission table, kept separate from the shared tree.

    Keys in ``grants`` are absolute resource paths and values are whether the
    path is allowed; **exceptions only**, no record means allowed (B15). Every
    key is validated as an absolute path (B16).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    grants: dict[str, bool] = Field(default_factory=dict)

    @field_validator("grants")
    @classmethod
    def _keys_must_be_absolute(cls, value: dict[str, bool]) -> dict[str, bool]:
        for key in value:
            if not is_absolute_path(key):
                raise ValueError(f"B16: grants keys must be full absolute paths, got {key!r}")
        return value

    # ------------------------------------------------------------ decisions

    def is_allowed(self, path: str) -> bool:
        """Whether this absolute path is allowed.

        Performs an **exact-path** decision only (B16: no prefix matching, no
        ancestor inheritance). Subtree hiding is handled by :func:`visible`,
        shared by ``build_tree`` / ``tree_index`` / ``tree_query`` (B5).

        Args:
            path: Node absolute path (must match ``ToolNode.path`` exactly, B16).

        Returns:
            ``True`` if allowed: no record defaults to allowed (B15), otherwise
            the recorded value.

        Raises:
            ValueError: ``path`` is not a valid absolute path (B16: relative
                paths and prefix matching are forbidden).
        """
        if not is_absolute_path(path):
            raise ValueError(
                f"B16: permission checks accept full absolute paths only, got {path!r}"
            )
        return self.grants.get(path, True)

    def is_denied(self, path: str) -> bool:
        """The inverse of ``is_allowed``, for readability (B5)."""
        return not self.is_allowed(path)

    def grant(self, path: str, allowed: bool = True) -> AgentPermissions:
        """Return a new table recording one exception (immutable, snapshot-friendly)."""
        if not is_absolute_path(path):
            raise ValueError(f"B16: grants keys must be full absolute paths, got {path!r}")
        return AgentPermissions(
            agent_id=self.agent_id, grants={**self.grants, path: allowed}
        )

    def revoke(self, path: str) -> AgentPermissions:
        """Return a new table with that exception record removed."""
        return AgentPermissions(
            agent_id=self.agent_id,
            grants={k: v for k, v in self.grants.items() if k != path},
        )


def visible(path: str, permissions: AgentPermissions) -> bool:
    """Visibility filter shared by ``tree_index``, ``tree_query`` and ``build_tree``.

    Denied nodes must not appear in the tree index or query results (B5). This
    function looks at the path and ``grants`` only, never at
    ``side_effect_scope`` (B18).

    Ancestor inheritance: **if any ancestor is explicitly denied, the whole
    subtree is invisible** (B5: "denying an intermediate node hides the entire
    subtree"). This differs from :meth:`AgentPermissions.is_allowed`, which
    answers only "does this exact path have a record".
    """
    segments = [seg for seg in path.split("/") if seg]
    for index in range(1, len(segments)):
        ancestor = "/" + "/".join(segments[:index])
        if permissions.grants.get(ancestor) is False:
            return False
    return permissions.is_allowed(path)


class PermissionDenied(Exception):
    """``3a PermissionDenied``: raised when calling a denied node directly.

    Per design doc §4.3: filtered out at query time, ``3a`` when called
    directly (B5).
    """

    error_type = "3a"

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"3a PermissionDenied: {path}")
