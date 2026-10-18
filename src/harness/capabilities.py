"""能力清单：工具与 skill 的定义、挂载状态与磁盘落盘（ADT-4/ADT-5 的配置面）。

对应文档：``docs/02-数据结构设计/04-工具与权限.md``。

本模块只解决**一个问题**：agent 能拿到哪些 ``ToolDef``。

::

    CapabilityManifest（内存，唯一配置面）
        ├── tools   : list[ToolDef]    —— path 首段 tool / mcp
        └── skills  : list[SkillDef]   —— path 首段 skill，按需读取 SKILL.md

    <workdir>/.agent/capabilities.json    —— 持久化（幂等，可 diff 进 Git）

设计约定：

* **只用标准库**：解析 JSON、拼装 schema 都不依赖 pydantic。
* **不执行任意代码**：工具由 ``builtin`` 白名单引用（``os/file/read`` 等），
  清单里出现未知 ``builtin`` 时拒绝装载，避免把配置文件变成代码入口。
* **只描述、不裁决**：并发/幂等/权限仍由 ``ToolDef`` 推导与校验（B11/B12/B17）。
* **skill 不占工具位**：skill 条目只提供路径与描述，内容在 ``skill/read`` 时返回，
  保证工具树索引不变胖。
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

#: 能力清单所在目录（与 ``memory.py`` 的 ``.agent`` 保持一致）。
CAPABILITY_DIR_NAME = ".agent"

#: 清单文件名。
CAPABILITY_FILENAME = "capabilities.json"

#: 清单格式版本；结构变更时递增，便于拒绝过旧文件。
#: v2 起支持 ``groups``、外来工具 ``source`` 与 MCP 服务器声明。
MANIFEST_VERSION = 2

#: 可自动升级的旧版本（读取时补齐缺失字段）。
SUPPORTED_OLD_VERSIONS: frozenset[int] = frozenset({1})


BUILTIN_TOOL_PATHS: tuple[str, ...] = tuple(sorted(TOOL_SPECS))


class CapabilityError(ValueError):
    """能力清单非法（未知 builtin、路径不合规、版本不符）。"""


# ---------------------------------------------------------------------------
# 分组与来源
# ---------------------------------------------------------------------------

#: 一级资源根（与 ``tool_tree.RESOURCE_ROOTS`` 一致，B19）。
RESOURCE_ROOTS: tuple[str, ...] = ("tool", "skill", "mcp")


def normalize_group(group: str) -> str:
    """规范化分组标识：``/tool/os`` 与 ``tool/os`` 等价，统一为后者。

    Raises:
        CapabilityError: 分组标识不合法（空、首段不在资源根内、含非法字符）。
    """
    text = (group or "").strip().strip("/")
    if not text:
        return ""
    segments = text.split("/")
    if any(not seg or seg in {".", ".."} for seg in segments):
        raise CapabilityError(f"分组标识含非法段: {group!r}")
    if segments[0] not in RESOURCE_ROOTS:
        raise CapabilityError(
            f"分组首段必须属于 {RESOURCE_ROOTS}，实际为 {segments[0]!r}: {group!r}"
        )
    return "/".join(segments)


def group_of(path: str) -> str:
    """从工具路径推导其所属分组（去掉末段）。

    ``/tool/os/file/read`` → ``tool/os/file``；``/tool/os`` → ``tool``。
    一级路径（``/tool``）的分组为空串——它本身就是资源根，不是分组。
    """
    segments = [seg for seg in path.split("/") if seg]
    if len(segments) <= 1:
        return ""
    return "/".join(segments[:-1])


@dataclass(frozen=True)
class GroupDef:
    """一条分组元信息。

    分组**不是**独立存储对象：它的存在性由挂载在其中的工具路径派生。
    本结构只承载「描述」这类无法从路径推导的元信息，避免让空分组
    只能显示默认的占位文字。

    Attributes:
        group_id: 规范化分组标识，如 ``tool/os``、``mcp/filesystem``。
        description: 单行描述；为空时索引只显示路径，不占 token。
    """

    group_id: str
    description: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_group(self.group_id)
        if not normalized:
            raise CapabilityError("分组标识不能为空")
        if len(normalized.split("/")) < 2:
            raise CapabilityError(
                f"分组至少需要两段（资源根 + 组名），实际为 {self.group_id!r}"
            )
        object.__setattr__(self, "group_id", normalized)
        _reject_multiline(self.description)

    @property
    def path(self) -> str:
        """工具树中的绝对路径。"""
        return "/" + self.group_id


@dataclass(frozen=True)
class SourceSpec:
    """外来工具的来源声明。

    立场：**允许执行工具本体代码**，来源安全由供应链保证。因此这里不做
    白名单限制，只做三项「防手误」校验（路径合法、描述单行、目标存在）。

    Attributes:
        kind: 来源类型，目前支持 ``python`` 与 ``mcp``。
        target: ``python`` 时为脚本路径（相对 workdir）或模块名；
            ``mcp`` 时为服务器名。
        callable: 目标函数名，缺省 ``run``。
    """

    kind: str = "python"
    target: str = ""
    callable: str = "run"

    def __post_init__(self) -> None:
        if self.kind not in {"python", "mcp"}:
            raise CapabilityError(f"未知来源类型: {self.kind!r}")
        if not self.target:
            raise CapabilityError("来源缺少 target")
        if self.kind == "python" and ("\n" in self.target or "\r" in self.target):
            raise CapabilityError(f"来源 target 不得含换行: {self.target!r}")

    def to_dict(self) -> dict[str, Any]:
        """导出为 JSON 友好结构。"""
        return {"kind": self.kind, "target": self.target, "callable": self.callable}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        """从 JSON 结构还原。"""
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
    """一条 skill 目录声明。

    Attributes:
        name: skill 名（唯一，用作路径末段）。
        description: 单行描述，写入工具树索引（受 B9/B10 约束）。
        source: 来源标记，如 ``builtin`` / ``github:anthropics/skills``。
        dir: 相对工作目录的目录路径；``SKILL.md`` 位于其下。
        extra_files: 同目录下随附的参考文件（相对该目录）。
    """

    name: str
    description: str
    source: str = "builtin"
    dir: str = ""
    extra_files: tuple[str, ...] = ()
    group: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise CapabilityError(f"skill 名非法: {self.name!r}")
        _reject_multiline(self.description)
        group = normalize_group(self.group)
        if group and not group.startswith("skill"):
            raise CapabilityError(f"skill 分组必须以 skill 开头: {self.group!r}")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "dir", self.dir or f".agent/skills/{self.name}")

    @property
    def path(self) -> str:
        """工具树中的资源路径；指定分组时挂到分组之下。"""
        if self.group:
            return f"/{self.group}/{self.name}"
        return f"/skill/{self.name}"

    @property
    def skill_md(self) -> str:
        """``SKILL.md`` 的相对路径。"""
        return f"{self.dir}/SKILL.md"

    def to_tooldef(self, workdir: str) -> ToolDef:
        """转成工具树条目：叶子，参数为相对文件名。"""
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
    """生成 skill 读取函数：只允许读取该 skill 目录内的文件。"""

    base = (Path(workdir) / rel_dir).resolve()

    def read(file: str = "SKILL.md") -> str:
        target = (base / file).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"路径越界: {file}")
        if not target.is_file():
            raise FileNotFoundError(f"skill 文件不存在: {rel_dir}/{file}")
        return target.read_text(encoding="utf-8", errors="replace")

    return read


# ---------------------------------------------------------------------------
# CapabilityManifest
# ---------------------------------------------------------------------------


@dataclass
class CapabilityManifest:
    """工具 + skill 的完整声明。

    Attributes:
        tools: 已挂载的工具条目（含 mcp 命名空间）。
        skills: 已挂载的 skill 声明。
        groups: 分组元信息（描述）。分组的**存在性**仍由路径派生，
            这里只存无法从路径推导的描述，因此允许出现空分组的描述。
        sources: ``path -> SourceSpec``，外来工具的来源声明。
            内置工具不在此表（它们由 ``builtin`` 白名单重建）。
        mcp_servers: MCP 服务器声明（服务器名即分组名）。
        notes: 人类可读备注（例如数据来源），不参与运行。
    """

    tools: list[ToolDef] = field(default_factory=list)
    skills: list[SkillDef] = field(default_factory=list)
    groups: list[GroupDef] = field(default_factory=list)
    sources: dict[str, SourceSpec] = field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- 分组

    def tool_paths(self) -> list[str]:
        """已挂载工具路径（排序）。"""
        return sorted(t.path for t in self.tools)

    def skill_names(self) -> list[str]:
        """已挂载 skill 名（排序）。"""
        return sorted(s.name for s in self.skills)

    def has_path(self, path: str) -> bool:
        """路径是否已挂载（工具或 skill）。"""
        if any(t.path == path for t in self.tools):
            return True
        return any(s.path == path for s in self.skills)

    def get_tool(self, path: str) -> ToolDef | None:
        """按路径取工具。"""
        for tool in self.tools:
            if tool.path == path:
                return tool
        return None

    def get_skill(self, name: str) -> SkillDef | None:
        """按名字取 skill。"""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    # ---------------------------------------------------------------- 变更

    def add_tool(self, tool: ToolDef) -> None:
        """追加工具；路径重复时抛 ``ToolTreeError``（B1）。"""
        if any(t.path == tool.path for t in self.tools) or any(
            s.path == tool.path for s in self.skills
        ):
            raise ToolTreeError("B1", f"路径已被占用: {tool.path!r}")
        self.tools.append(tool)

    def add_skill(self, skill: SkillDef) -> None:
        """追加 skill；同名时抛 ``CapabilityError``。"""
        if any(s.name == skill.name for s in self.skills):
            raise CapabilityError(f"skill 已存在: {skill.name!r}")
        if self.has_path(skill.path):
            raise ToolTreeError("B1", f"路径已被占用: {skill.path!r}")
        self.skills.append(skill)

    def remove(self, path: str) -> bool:
        """按路径移除工具或 skill，返回是否命中。"""
        before = len(self.tools) + len(self.skills)
        self.tools = [t for t in self.tools if t.path != path]
        self.skills = [s for s in self.skills if s.path != path]
        return len(self.tools) + len(self.skills) != before

    def to_tooldefs(self, workdir: str) -> list[ToolDef]:
        """导出全部工具条目（skill 在此转换成叶子工具）。"""
        return list(self.tools) + [s.to_tooldef(workdir) for s in self.skills]

    # ---------------------------------------------------------------- 分组

    def get_group(self, group_id: str) -> GroupDef | None:
        """按标识取分组元信息。"""
        normalized = normalize_group(group_id)
        for group in self.groups:
            if group.group_id == normalized:
                return group
        return None

    def add_group(self, group: GroupDef) -> None:
        """追加分组元信息；重复时抛 ``CapabilityError``。"""
        if self.get_group(group.group_id) is not None:
            raise CapabilityError(f"分组已存在: {group.group_id!r}")
        self.groups.append(group)

    def remove_group(self, group_id: str, *, force: bool = False) -> bool:
        """删除分组元信息。

        分组不独立存储，因此删描述不会删工具；但有成员且未 ``force`` 时
        拒绝，避免用户误以为成员也一并移除。
        """
        normalized = normalize_group(group_id)
        prefix = f"/{normalized}/"
        members = [t.path for t in self.tools if t.path.startswith(prefix)]
        members += [s.path for s in self.skills if s.path.startswith(prefix)]
        if members and not force:
            raise CapabilityError(
                f"分组 {normalized!r} 下仍有 {len(members)} 个成员，"
                "先移除成员或使用 force"
            )
        before = len(self.groups)
        self.groups = [g for g in self.groups if g.group_id != normalized]
        return len(self.groups) != before

    def group_ids(self) -> list[str]:
        """所有分组标识：已声明的 + 从路径派生的（排序、去重）。"""
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
        """按一级资源根聚合分组标识。"""
        buckets: dict[str, list[str]] = {root: [] for root in RESOURCE_ROOTS}
        for group_id in self.group_ids():
            buckets.setdefault(group_id.split("/")[0], []).append(group_id)
        return buckets

    # ---------------------------------------------------------------- 序列化

    def to_dict(self) -> dict[str, Any]:
        """导出为 JSON 友好结构（工具只记路径与描述，不记函数）。"""
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
        """内置工具路径 → 白名单键（``/tool/os/echo`` → ``os/echo``）。"""
        if path in BUILTIN_TOOL_PATHS:
            return path.lstrip("/").split("/", 1)[-1]
        return ""


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------


def manifest_path(workdir: str | Path) -> Path:
    """清单路径：``<workdir>/.agent/capabilities.json``。"""
    return Path(workdir).resolve() / CAPABILITY_DIR_NAME / CAPABILITY_FILENAME


def save_manifest(manifest: CapabilityManifest, workdir: str | Path) -> Path:
    """原子写入清单（先写临时文件再替换，Windows 友好）。"""
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
    """读取清单并重建工具条目。

    三种条目共存：

    * **内置工具**：``builtin`` 字段指向白名单键（``os/file/read``）。
    * **外来工具**：``source`` 字段描述来源，允许执行本体代码；
      目标不存在时不阻塞其他工具，记入 ``problems`` 并跳过该条。
    * **MCP 工具**：``source.kind == "mcp"``，调用时经 JSON-RPC 转发。

    Args:
        workdir: 工作目录。
        factory: ``path -> ToolDef`` 工厂；省略时使用 :func:`build_builtin_tool`。
        problems: 可选列表；非致命错误（如外来工具目标缺失）追加到此。

    Raises:
        CapabilityError: 文件缺失、版本不受支持或内置路径未知。
    """
    target = manifest_path(workdir)
    if not target.is_file():
        raise CapabilityError(f"能力清单不存在: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CapabilityError(f"能力清单不是合法 JSON: {exc}") from exc

    version = data.get("version")
    if version != MANIFEST_VERSION and version not in SUPPORTED_OLD_VERSIONS:
        raise CapabilityError(
            f"能力清单版本不符: {version!r} 不支持（当前 {MANIFEST_VERSION}）"
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
                    f"工具 {path!r} 既非内置白名单路径，也未声明 source"
                )
            tool = _apply_entry_overrides(tool, entry)
            manifest.add_tool(tool)
        except Exception as exc:  # noqa: BLE001 - 单条失败不拖垮整张清单
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
    """用清单里的显式声明覆盖工厂产物的元数据（``ToolDef`` 不可变，返回新对象）。"""
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
    changes.setdefault("concurrency", None)  # 由新 caps 重新推导（B11）
    return dataclasses.replace(tool, **changes)


# ---------------------------------------------------------------------------
# 外来工具
# ---------------------------------------------------------------------------


def load_ext_callable(
    source: SourceSpec, workdir: str | Path
) -> Callable[..., Any]:
    """加载外来工具函数（惰性，首次调用时才 import）。

    立场：**允许执行工具本体代码**，来源安全由供应链保证。因此不做
    沙箱与白名单限制，只保证路径合法且目标存在。

    ``target`` 支持两种形式：

    * 脚本路径：``./ext/hello.py`` 或 ``ext/hello.py``（相对 workdir）；
    * 模块名：``mypkg.tools.hello``（从 ``sys.path`` 导入）。

    Raises:
        CapabilityError: 目标不存在或无法解析出可调用对象。
    """
    target = source.target
    callable_name = source.callable or "run"

    if target.endswith(".py") or "/" in target or "\\" in target:
        script = Path(target)
        if not script.is_absolute():
            script = Path(workdir).resolve() / script
        if not script.is_file():
            raise CapabilityError(f"外来工具脚本不存在: {script}")
        module_name = f"_harness_ext_{abs(hash(str(script))):x}"
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise CapabilityError(f"无法加载外来工具脚本: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise CapabilityError(f"无法导入模块 {target!r}: {exc}") from exc

    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise CapabilityError(
            f"{target!r} 中不存在可调用对象 {callable_name!r}"
        )
    return fn


def _build_external_tool(
    path: str,
    entry: dict[str, Any],
    source: SourceSpec,
    workdir: str,
) -> ToolDef:
    """构造外来工具条目。

    两类检查分别在两个时点：

    * **加载期**（本函数）：确认脚本存在、能解析出可调用对象——这样
      ``load_manifest`` 能立即发现问题并记入 ``problems``，而不是等到
      模型真的调用时才炸。
    * **执行期**（``invoke``）：真正 import 并调用；若函数签名声明了
      ``workdir``，按名注入当前工作目录（与内置工具的 hidden 注入一致）。
    """
    if source.kind == "mcp":
        return _build_mcp_tool(path, entry, source, workdir)

    fn = load_ext_callable(source, workdir)  # 加载期校验：目标必须存在

    description = entry.get("description") or f"外来工具 {path}"
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
    """函数签名是否接受 ``workdir`` 参数（含 ``**kwargs``）。"""
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
    """构造 MCP 工具条目：``source.target`` 为服务器名。"""
    from harness.mcp_client import call_tool as _mcp_call
    from harness.mcp_client import load_mcp_servers

    server = source.target
    tool_name = entry.get("remote_name") or path.rsplit("/", 1)[-1]

    def invoke(**kwargs: Any) -> Any:
        servers = load_mcp_servers(workdir)
        conf = servers.get(server)
        if conf is None:
            raise CapabilityError(f"MCP 服务器未声明: {server!r}")
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
# 内置工具工厂
# ---------------------------------------------------------------------------

#: 内置 skill 模板：名称 → （描述, 正文）。正文在 ``skills add --from <来源>`` 时落盘。
BUILTIN_SKILLS: dict[str, tuple[str, str]] = {}


def build_builtin_tool(path: str, workdir: str) -> ToolDef:
    """按路径构造内置工具（未实现时抛 ``CapabilityError``）。"""
    if path not in TOOL_SPECS:
        raise CapabilityError(f"未知工具路径: {path!r}")
    return _build_tool(path, workdir)


# ---------------------------------------------------------------------------
# skill 目录
# ---------------------------------------------------------------------------


def skills_dir(workdir: str | Path) -> Path:
    """skill 根目录：``<workdir>/.agent/skills``。"""
    return Path(workdir).resolve() / CAPABILITY_DIR_NAME / "skills"


def write_skill_files(workdir: str | Path, name: str, files: dict[str, str]) -> Path:
    """把 ``文件名 -> 内容`` 写入 ``.agent/skills/<name>``，返回目录。

    Raises:
        CapabilityError: 文件名含路径分隔符（禁止逃逸）。
    """
    base = skills_dir(workdir) / name
    for filename, content in files.items():
        if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
            raise CapabilityError(f"非法文件名: {filename!r}")
        base.mkdir(parents=True, exist_ok=True)
        (base / filename).write_text(content, encoding="utf-8")
    return base
