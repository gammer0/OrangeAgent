"""Capability manifest: tool and skill definitions, mount state and disk persistence.

This is the configuration surface of ADT-4/ADT-5.

See ``docs/02-data-structures/04-tools-and-permissions.md``.

This module solves **one problem**: which ``ToolDef`` values the agent gets.

::

    CapabilityManifest (in memory, the single configuration surface)
        ├── tools   : list[ToolDef]    -- first path segment tool / mcp
        └── skills  : list[SkillDef]   -- first path segment skill; SKILL.md read on demand

    <workdir>/.agent/capabilities.json    -- persisted (idempotent, Git-diffable)

Design conventions:

* **Standard library only**: parsing JSON and assembling schemas never uses pydantic.
* **No arbitrary code execution**: tools are referenced by a ``builtin`` whitelist
  (``os/file/read`` etc.); an unknown ``builtin`` in the manifest is refused, so
  the config file never becomes a code entry point.
* **Describe, never adjudicate**: concurrency / idempotency / permissions are
  still derived and validated by ``ToolDef`` (B11/B12/B17).
* **Skills take no tool slots**: a skill entry carries only a path and a
  description; its content is returned by ``skill/read``, keeping the tool-tree
  index from bloating.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness.permission import Permission
from harness.tool_tree import ToolDef, ToolTreeError, _reject_multiline
from harness.tools_builtin import TOOL_SPECS
from harness.tools_builtin import build_tool as _build_tool

__all__ = [
    "CAPABILITY_DIR_NAME",
    "CAPABILITY_FILENAME",
    "MANIFEST_VERSION",
    "BUILTIN_TOOL_PATHS",
    "BUILTIN_SKILLS",
    "CapabilityError",
    "GroupDef",
    "SourceSpec",
    "SkillDef",
    "CapabilityManifest",
    "manifest_path",
    "skills_dir",
    "load_manifest",
    "save_manifest",
    "write_skill_files",
    "group_of",
    "normalize_group",
    "load_ext_callable",
]

#: Directory holding the capability manifest (same ``.agent`` as ``memory.py``).
CAPABILITY_DIR_NAME = ".agent"

#: Manifest file name.
CAPABILITY_FILENAME = "capabilities.json"

#: Manifest format version; bumped on structural changes so stale files are rejected.
#: Since v2 it supports ``groups``, external-tool ``source`` and MCP server declarations.
MANIFEST_VERSION = 2

#: Old versions that can be auto-upgraded (missing fields are filled on read).
SUPPORTED_OLD_VERSIONS: frozenset[int] = frozenset({1})


BUILTIN_TOOL_PATHS: tuple[str, ...] = tuple(sorted(TOOL_SPECS))


class CapabilityError(ValueError):
    """The capability manifest is invalid (unknown builtin, bad path, version mismatch)."""


# ---------------------------------------------------------------------------
# Groups and sources
# ---------------------------------------------------------------------------

#: L1 resource roots (consistent with ``tool_tree.RESOURCE_ROOTS``, B19).
RESOURCE_ROOTS: tuple[str, ...] = ("tool", "skill", "mcp")


def normalize_group(group: str) -> str:
    """Normalize a group identifier: ``/tool/os`` and ``tool/os`` are equivalent, both become the latter.

    Raises:
        CapabilityError: The identifier is invalid (empty, first segment not a
            resource root, or containing an illegal character).
    """
    text = (group or "").strip().strip("/")
    if not text:
        return ""
    segments = text.split("/")
    if any(not seg or seg in {".", ".."} for seg in segments):
        raise CapabilityError(f"group identifier contains an illegal segment: {group!r}")
    if segments[0] not in RESOURCE_ROOTS:
        raise CapabilityError(
            f"the first group segment must be one of {RESOURCE_ROOTS}, "
            f"got {segments[0]!r}: {group!r}"
        )
    return "/".join(segments)


def group_of(path: str) -> str:
    """Derive a tool's group from its path (dropping the last segment).

    ``/tool/os/file/read`` -> ``tool/os/file``; ``/tool/os`` -> ``tool``.
    A first-level path (``/tool``) yields an empty group -- it is itself a
    resource root, not a group.
    """
    segments = [seg for seg in path.split("/") if seg]
    if len(segments) <= 1:
        return ""
    return "/".join(segments[:-1])


@dataclass(frozen=True)
class GroupDef:
    """One piece of group metadata.

    A group is **not** an independently stored object: its existence is derived
    from the tool paths mounted within it. This structure carries only metadata
    that cannot be derived from the path, such as a description, so an empty
    group does not have to fall back to placeholder text.

    Attributes:
        group_id: Normalized group identifier, e.g. ``tool/os``, ``mcp/filesystem``.
        description: Single-line description; when empty the index shows the path
            only and costs no tokens.
    """

    group_id: str
    description: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_group(self.group_id)
        if not normalized:
            raise CapabilityError("group identifier must not be empty")
        if len(normalized.split("/")) < 2:
            raise CapabilityError(
                f"a group needs at least two segments (resource root + group name), "
                f"got {self.group_id!r}"
            )
        object.__setattr__(self, "group_id", normalized)
        _reject_multiline(self.description)

    @property
    def path(self) -> str:
        """Absolute path in the tool tree."""
        return "/" + self.group_id


@dataclass(frozen=True)
class SourceSpec:
    """Source declaration of an external tool.

    Position: **running the tool's own code is allowed**; source safety is
    guaranteed by the supply chain. So there is no whitelist here -- only three
    "typo guards" (valid path, single-line description, target exists).

    Attributes:
        kind: Source kind; currently ``python`` or ``mcp``.
        target: For ``python``, a script path (relative to workdir) or module
            name; for ``mcp``, the server name.
        callable: Target function name, default ``run``.
    """

    kind: str = "python"
    target: str = ""
    callable: str = "run"

    def __post_init__(self) -> None:
        if self.kind not in {"python", "mcp"}:
            raise CapabilityError(f"unknown source kind: {self.kind!r}")
        if not self.target:
            raise CapabilityError("source is missing target")
        if self.kind == "python" and ("\n" in self.target or "\r" in self.target):
            raise CapabilityError(f"source target must not contain newlines: {self.target!r}")

    def to_dict(self) -> dict[str, Any]:
        """Export as a JSON-friendly structure."""
        return {"kind": self.kind, "target": self.target, "callable": self.callable}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        """Restore from a JSON structure."""
        return cls(
            kind=data.get("kind", "python"),
            target=data.get("target", ""),
            callable=data.get("callable", "run"),
        )


# ---------------------------------------------------------------------------
# SkillDef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillDef:
    """A single skill directory declaration.

    Attributes:
        name: Skill name (unique; used as the final path segment).
        description: Single-line description written into the tool-tree index
            (subject to B9/B10).
        source: Source marker, e.g. ``builtin`` / ``github:anthropics/skills``.
        dir: Directory path relative to the working directory; ``SKILL.md``
            lives inside it.
        extra_files: Accompanying reference files in the same directory
            (relative to that directory).
    """

    name: str
    description: str
    source: str = "builtin"
    dir: str = ""
    extra_files: tuple[str, ...] = ()
    group: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise CapabilityError(f"invalid skill name: {self.name!r}")
        _reject_multiline(self.description)
        group = normalize_group(self.group)
        if group and not group.startswith("skill"):
            raise CapabilityError(f"a skill group must start with 'skill': {self.group!r}")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "dir", self.dir or f".agent/skills/{self.name}")

    @property
    def path(self) -> str:
        """Resource path in the tool tree; mounted under the group when one is set."""
        if self.group:
            return f"/{self.group}/{self.name}"
        return f"/skill/{self.name}"

    @property
    def skill_md(self) -> str:
        """Relative path of ``SKILL.md``."""
        return f"{self.dir}/SKILL.md"

    def to_tooldef(self, workdir: str) -> ToolDef:
        """Convert into a tool-tree entry: a leaf whose argument is a relative file name."""
        return ToolDef(
            tool_id=f"s_{self.name}",
            path=self.path,
            fn=_skill_reader(workdir, self.dir),
            schema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "相对文件名，默认 SKILL.md",
                        "default": "SKILL.md",
                    }
                },
                "required": [],
            },
            caps=["read"],
            permission=Permission(src="skill"),
            description=self.description,
        )


def _skill_reader(workdir: str, rel_dir: str):
    """Build a skill reader: only files inside that skill's directory may be read."""

    base = (Path(workdir) / rel_dir).resolve()

    def read(file: str = "SKILL.md") -> str:
        target = (base / file).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"path escapes the skill directory: {file}")
        if not target.is_file():
            raise FileNotFoundError(f"skill file does not exist: {rel_dir}/{file}")
        return target.read_text(encoding="utf-8", errors="replace")

    return read


# ---------------------------------------------------------------------------
# CapabilityManifest
# ---------------------------------------------------------------------------


@dataclass
class CapabilityManifest:
    """The complete declaration of tools + skills.

    Attributes:
        tools: Mounted tool entries (including the mcp namespace).
        skills: Mounted skill declarations.
        groups: Group metadata (descriptions). A group's **existence** is still
            derived from paths; only indescribable-by-path metadata is stored
            here, which is why an empty group may still have a description.
        sources: ``path -> SourceSpec``, source declarations for external tools.
            Builtin tools are absent because the ``builtin`` whitelist rebuilds them.
        mcp_servers: MCP server declarations (the server name is the group name).
        notes: Human-readable remarks (e.g. data provenance); not used at runtime.
    """

    tools: list[ToolDef] = field(default_factory=list)
    skills: list[SkillDef] = field(default_factory=list)
    groups: list[GroupDef] = field(default_factory=list)
    sources: dict[str, SourceSpec] = field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # --------------------------------------------------------------- groups

    def tool_paths(self) -> list[str]:
        """Mounted tool paths (sorted)."""
        return sorted(t.path for t in self.tools)

    def skill_names(self) -> list[str]:
        """Mounted skill names (sorted)."""
        return sorted(s.name for s in self.skills)

    def has_path(self, path: str) -> bool:
        """Whether the path is already mounted (tool or skill)."""
        if any(t.path == path for t in self.tools):
            return True
        return any(s.path == path for s in self.skills)

    def get_tool(self, path: str) -> ToolDef | None:
        """Look up a tool by path."""
        for tool in self.tools:
            if tool.path == path:
                return tool
        return None

    def get_skill(self, name: str) -> SkillDef | None:
        """Look up a skill by name."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    # -------------------------------------------------------------- mutation

    def add_tool(self, tool: ToolDef) -> None:
        """Append a tool; a duplicate path raises ``ToolTreeError`` (B1)."""
        if any(t.path == tool.path for t in self.tools) or any(
            s.path == tool.path for s in self.skills
        ):
            raise ToolTreeError("B1", f"path is already taken: {tool.path!r}")
        self.tools.append(tool)

    def add_skill(self, skill: SkillDef) -> None:
        """Append a skill; a duplicate name raises ``CapabilityError``."""
        if any(s.name == skill.name for s in self.skills):
            raise CapabilityError(f"skill already exists: {skill.name!r}")
        if self.has_path(skill.path):
            raise ToolTreeError("B1", f"path is already taken: {skill.path!r}")
        self.skills.append(skill)

    def remove(self, path: str) -> bool:
        """Remove a tool or skill by path; returns whether anything matched."""
        before = len(self.tools) + len(self.skills)
        self.tools = [t for t in self.tools if t.path != path]
        self.skills = [s for s in self.skills if s.path != path]
        return len(self.tools) + len(self.skills) != before

    def to_tooldefs(self, workdir: str) -> list[ToolDef]:
        """Export every tool entry (skills are converted into leaf tools here)."""
        return list(self.tools) + [s.to_tooldef(workdir) for s in self.skills]

    # --------------------------------------------------------------- groups

    def get_group(self, group_id: str) -> GroupDef | None:
        """Look up group metadata by identifier."""
        normalized = normalize_group(group_id)
        for group in self.groups:
            if group.group_id == normalized:
                return group
        return None

    def add_group(self, group: GroupDef) -> None:
        """Append group metadata; duplicates raise ``CapabilityError``."""
        if self.get_group(group.group_id) is not None:
            raise CapabilityError(f"group already exists: {group.group_id!r}")
        self.groups.append(group)

    def remove_group(self, group_id: str, *, force: bool = False) -> bool:
        """Remove group metadata.

        Groups are not stored independently, so deleting the description never
        deletes tools. But when members remain and ``force`` is not set we refuse,
        so users do not mistake this for removing the members too.
        """
        normalized = normalize_group(group_id)
        prefix = f"/{normalized}/"
        members = [t.path for t in self.tools if t.path.startswith(prefix)]
        members += [s.path for s in self.skills if s.path.startswith(prefix)]
        if members and not force:
            raise CapabilityError(
                f"group {normalized!r} still has {len(members)} members; "
                "remove the members first or use force"
            )
        before = len(self.groups)
        self.groups = [g for g in self.groups if g.group_id != normalized]
        return len(self.groups) != before

    def group_ids(self) -> list[str]:
        """All group identifiers: declared ones plus those derived from paths (sorted, deduped)."""
        found = {g.group_id for g in self.groups}
        for tool in self.tools:
            parent = group_of(tool.path)
            while parent and parent.count("/") >= 1:
                found.add(parent)
                parent = group_of("/" + parent)
        for skill in self.skills:
            parent = group_of(skill.path)
            while parent and parent.count("/") >= 1:
                found.add(parent)
                parent = group_of("/" + parent)
        return sorted(found)

    def groups_by_root(self) -> dict[str, list[str]]:
        """Bucket group identifiers by their L1 resource root."""
        buckets: dict[str, list[str]] = {root: [] for root in RESOURCE_ROOTS}
        for group_id in self.group_ids():
            buckets.setdefault(group_id.split("/")[0], []).append(group_id)
        return buckets

    # ---------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Export as a JSON-friendly structure (tools record path and description only, never functions)."""
        tools = []
        for tool in sorted(self.tools, key=lambda x: x.path):
            entry: dict[str, Any] = {
                "path": tool.path,
                "description": tool.description,
                "caps": list(tool.caps),
                "idempotent": tool.idempotent,
                "builtin": self._builtin_key(tool.path),
            }
            source = self.sources.get(tool.path)
            if source is not None:
                entry["source"] = source.to_dict()
            if tool.schema and tool.schema.get("properties"):
                entry["schema"] = tool.schema
            tools.append(entry)
        return {
            "version": MANIFEST_VERSION,
            "notes": list(self.notes),
            "groups": [
                {"group_id": g.group_id, "description": g.description}
                for g in sorted(self.groups, key=lambda x: x.group_id)
            ],
            "tools": tools,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "source": s.source,
                    "dir": s.dir,
                    "extra_files": list(s.extra_files),
                    "group": s.group,
                }
                for s in sorted(self.skills, key=lambda x: x.name)
            ],
            "mcp_servers": list(self.mcp_servers),
        }

    @staticmethod
    def _builtin_key(path: str) -> str:
        """Builtin tool path -> whitelist key (``/tool/os/echo`` -> ``os/echo``)."""
        if path in BUILTIN_TOOL_PATHS:
            return path.lstrip("/").split("/", 1)[-1]
        return ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def manifest_path(workdir: str | Path) -> Path:
    """Manifest path: ``<workdir>/.agent/capabilities.json``."""
    return Path(workdir).resolve() / CAPABILITY_DIR_NAME / CAPABILITY_FILENAME


def save_manifest(manifest: CapabilityManifest, workdir: str | Path) -> Path:
    """Atomically write the manifest (temp file then replace; Windows-friendly)."""
    target = manifest_path(workdir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def load_manifest(
    workdir: str | Path,
    *,
    factory: Any = None,
    problems: list[str] | None = None,
) -> CapabilityManifest:
    """Load the manifest and rebuild the tool entries.

    Three kinds of entry coexist:

    * **Builtin tools**: the ``builtin`` field points to a whitelist key
      (``os/file/read``).
    * **External tools**: the ``source`` field describes the origin, and running
      the tool's own code is allowed; when the target is missing other tools are
      unaffected -- the failure is recorded in ``problems`` and that entry is
      skipped.
    * **MCP tools**: ``source.kind == "mcp"``, forwarded over JSON-RPC when called.

    Args:
        workdir: Working directory.
        factory: ``path -> ToolDef`` factory; defaults to
            :func:`build_builtin_tool`.
        problems: Optional list; non-fatal errors (e.g. a missing external tool
            target) are appended here.

    Raises:
        CapabilityError: The file is missing, the version is unsupported, or a
            builtin path is unknown.
    """
    target = manifest_path(workdir)
    if not target.is_file():
        raise CapabilityError(f"capability manifest does not exist: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CapabilityError(f"capability manifest is not valid JSON: {exc}") from exc

    version = data.get("version")
    if version != MANIFEST_VERSION and version not in SUPPORTED_OLD_VERSIONS:
        raise CapabilityError(
            f"unsupported capability manifest version: {version!r} "
            f"(current {MANIFEST_VERSION})"
        )

    make = factory or build_builtin_tool
    manifest = CapabilityManifest(notes=list(data.get("notes") or []))

    for entry in data.get("groups") or []:
        try:
            manifest.add_group(
                GroupDef(
                    group_id=entry.get("group_id", ""),
                    description=entry.get("description", ""),
                )
            )
        except CapabilityError as exc:
            if problems is not None:
                problems.append(f"group {entry.get('group_id')!r}: {exc}")

    for entry in data.get("tools") or []:
        path = entry.get("path", "")
        source_data = entry.get("source")
        try:
            if source_data:
                source = SourceSpec.from_dict(source_data)
                tool = _build_external_tool(path, entry, source, str(workdir))
                manifest.sources[path] = source
            elif path in BUILTIN_TOOL_PATHS:
                tool = make(path, str(workdir))
            else:
                raise CapabilityError(
                    f"tool {path!r} is neither a builtin whitelist path nor declares a source"
                )
            tool = _apply_entry_overrides(tool, entry)
            manifest.add_tool(tool)
        except Exception as exc:  # noqa: BLE001 - one bad entry must not break the manifest
            if problems is None:
                raise
            problems.append(f"tool {path!r}: {exc}")

    for entry in data.get("skills") or []:
        manifest.add_skill(
            SkillDef(
                name=entry["name"],
                description=entry.get("description", ""),
                source=entry.get("source", "builtin"),
                dir=entry.get("dir", ""),
                extra_files=tuple(entry.get("extra_files") or ()),
                group=entry.get("group", ""),
            )
        )

    manifest.mcp_servers = list(data.get("mcp_servers") or [])
    return manifest


def _apply_entry_overrides(tool: ToolDef, entry: dict[str, Any]) -> ToolDef:
    """Override factory metadata with explicit manifest declarations (``ToolDef`` is immutable; returns a new object)."""
    changes: dict[str, Any] = {}
    description = entry.get("description")
    if description:
        changes["description"] = description
    caps = entry.get("caps")
    if caps:
        changes["caps"] = list(caps)
        changes["permission"] = Permission(src=tool.permission.src, caps=list(caps))
    if "idempotent" in entry:
        changes["idempotent"] = bool(entry["idempotent"])
    schema = entry.get("schema")
    if isinstance(schema, dict) and schema.get("properties") is not None:
        changes["schema"] = schema
    if not changes:
        return tool
    changes.setdefault("concurrency", None)  # re-derive from the new caps (B11)
    return dataclasses.replace(tool, **changes)


# ---------------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------------


def load_ext_callable(
    source: SourceSpec, workdir: str | Path
) -> Callable[..., Any]:
    """Load an external tool function (lazily; imported on first call).

    Position: **running the tool's own code is allowed**; source safety is
    guaranteed by the supply chain. So there is no sandbox and no whitelist -- we
    only require a valid path and an existing target.

    ``target`` supports two forms:

    * A script path: ``./ext/hello.py`` or ``ext/hello.py`` (relative to workdir);
    * A module name: ``mypkg.tools.hello`` (imported from ``sys.path``).

    Raises:
        CapabilityError: The target is missing or no callable can be resolved.
    """
    target = source.target
    callable_name = source.callable or "run"

    if target.endswith(".py") or "/" in target or "\\" in target:
        script = Path(target)
        if not script.is_absolute():
            script = Path(workdir).resolve() / script
        if not script.is_file():
            raise CapabilityError(f"external tool script does not exist: {script}")
        module_name = f"_harness_ext_{abs(hash(str(script))):x}"
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise CapabilityError(f"cannot load external tool script: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise CapabilityError(f"cannot import module {target!r}: {exc}") from exc

    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise CapabilityError(
            f"no callable {callable_name!r} found in {target!r}"
        )
    return fn


def _build_external_tool(
    path: str,
    entry: dict[str, Any],
    source: SourceSpec,
    workdir: str,
) -> ToolDef:
    """Build an external tool entry.

    Two kinds of check happen at two different times:

    * **Load time** (this function): confirm the script exists and a callable can
      be resolved -- so ``load_manifest`` finds the problem immediately and
      records it in ``problems`` instead of blowing up only when the model finally
      calls the tool.
    * **Execution time** (``invoke``): actually import and call; if the signature
      declares ``workdir``, the current working directory is injected by name
      (consistent with the builtin hidden-parameter injection).
    """
    if source.kind == "mcp":
        return _build_mcp_tool(path, entry, source, workdir)

    fn = load_ext_callable(source, workdir)  # load-time check: target must exist

    description = entry.get("description") or f"external tool {path}"
    schema = entry.get("schema") or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    inject_workdir = _accepts_workdir(fn)

    def invoke(**kwargs: Any) -> Any:
        if inject_workdir:
            kwargs.setdefault("workdir", workdir)
        return fn(**kwargs)

    caps = list(entry.get("caps") or ["exec"])
    return ToolDef(
        tool_id="t_ext_" + path.strip("/").replace("/", "_"),
        path=path,
        fn=invoke,
        schema=schema,
        caps=caps,
        idempotent=bool(entry.get("idempotent", False)),
        permission=Permission(src="tool", caps=caps),
        description=description,
    )


def _accepts_workdir(fn: Callable[..., Any]) -> bool:
    """Whether the signature accepts a ``workdir`` parameter (including ``**kwargs``)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.name == "workdir":
            return True
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _build_mcp_tool(
    path: str,
    entry: dict[str, Any],
    source: SourceSpec,
    workdir: str,
) -> ToolDef:
    """Build an MCP tool entry: ``source.target`` is the server name."""
    from harness.mcp_client import call_tool as _mcp_call
    from harness.mcp_client import load_mcp_servers

    server = source.target
    tool_name = entry.get("remote_name") or path.rsplit("/", 1)[-1]

    def invoke(**kwargs: Any) -> Any:
        servers = load_mcp_servers(workdir)
        conf = servers.get(server)
        if conf is None:
            raise CapabilityError(f"MCP server is not declared: {server!r}")
        return _mcp_call(conf, tool_name, kwargs)

    return ToolDef(
        tool_id="t_mcp_" + path.strip("/").replace("/", "_"),
        path=path,
        fn=invoke,
        schema=entry.get("schema")
        or {"type": "object", "properties": {}, "required": []},
        caps=list(entry.get("caps") or ["exec"]),
        idempotent=bool(entry.get("idempotent", False)),
        permission=Permission(src="mcp", caps=list(entry.get("caps") or ["exec"])),
        description=entry.get("description") or f"MCP {server}/{tool_name}",
    )


# ---------------------------------------------------------------------------
# Builtin tool factory
# ---------------------------------------------------------------------------

#: Builtin skill templates: name -> (description, body). The body is written to
#: disk by ``skills add --from <source>``.
BUILTIN_SKILLS: dict[str, tuple[str, str]] = {}


def build_builtin_tool(path: str, workdir: str) -> ToolDef:
    """Build a builtin tool by path (raises ``CapabilityError`` when unimplemented)."""
    if path not in TOOL_SPECS:
        raise CapabilityError(f"unknown tool path: {path!r}")
    return _build_tool(path, workdir)


# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------


def skills_dir(workdir: str | Path) -> Path:
    """Skill root directory: ``<workdir>/.agent/skills``."""
    return Path(workdir).resolve() / CAPABILITY_DIR_NAME / "skills"


def write_skill_files(workdir: str | Path, name: str, files: dict[str, str]) -> Path:
    """Write ``filename -> content`` into ``.agent/skills/<name>`` and return the directory.

    Raises:
        CapabilityError: A filename contains a path separator (escaping is forbidden).
    """
    base = skills_dir(workdir) / name
    for filename, content in files.items():
        if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
            raise CapabilityError(f"invalid file name: {filename!r}")
        base.mkdir(parents=True, exist_ok=True)
        (base / filename).write_text(content, encoding="utf-8")
    return base
