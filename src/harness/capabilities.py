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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "SkillDef",
    "CapabilityManifest",
    "manifest_path",
    "skills_dir",
    "load_manifest",
    "save_manifest",
    "write_skill_files",
]

#: 能力清单所在目录（与 ``memory.py`` 的 ``.agent`` 保持一致）。
CAPABILITY_DIR_NAME = ".agent"

#: 清单文件名。
CAPABILITY_FILENAME = "capabilities.json"

#: 清单格式版本；结构变更时递增，便于拒绝过旧文件。
MANIFEST_VERSION = 1


BUILTIN_TOOL_PATHS: tuple[str, ...] = tuple(sorted(TOOL_SPECS))


class CapabilityError(ValueError):
    """能力清单非法（未知 builtin、路径不合规、版本不符）。"""


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

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise CapabilityError(f"skill 名非法: {self.name!r}")
        _reject_multiline(self.description)
        object.__setattr__(self, "dir", self.dir or f".agent/skills/{self.name}")

    @property
    def path(self) -> str:
        """工具树中的资源路径。"""
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
        notes: 人类可读备注（例如数据来源），不参与运行。
    """

    tools: list[ToolDef] = field(default_factory=list)
    skills: list[SkillDef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- 查询

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

    # ---------------------------------------------------------------- 序列化

    def to_dict(self) -> dict[str, Any]:
        """导出为 JSON 友好结构（工具只记路径与描述，不记函数）。"""
        return {
            "version": MANIFEST_VERSION,
            "notes": list(self.notes),
            "tools": [
                {
                    "path": t.path,
                    "description": t.description,
                    "caps": list(t.caps),
                    "idempotent": t.idempotent,
                }
                for t in sorted(self.tools, key=lambda x: x.path)
            ],
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "source": s.source,
                    "dir": s.dir,
                    "extra_files": list(s.extra_files),
                }
                for s in sorted(self.skills, key=lambda x: x.name)
            ],
        }


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
) -> CapabilityManifest:
    """读取清单并按 ``builtin`` 白名单重建步骤。

    Args:
        workdir: 工作目录。
        factory: ``path -> ToolDef`` 工厂；省略时使用 :func:`build_builtin_tool`。

    Raises:
        CapabilityError: 文件缺失、版本不符或含未知路径。
    """
    target = manifest_path(workdir)
    if not target.is_file():
        raise CapabilityError(f"能力清单不存在: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CapabilityError(f"能力清单不是合法 JSON: {exc}") from exc
    if data.get("version") != MANIFEST_VERSION:
        raise CapabilityError(
            f"能力清单版本不符: {data.get('version')!r} != {MANIFEST_VERSION}"
        )

    make = factory or build_builtin_tool
    manifest = CapabilityManifest(notes=list(data.get("notes") or []))
    for entry in data.get("tools") or []:
        path = entry.get("path", "")
        if path not in BUILTIN_TOOL_PATHS:
            raise CapabilityError(f"未知工具路径: {path!r}")
        manifest.add_tool(make(path, str(workdir)))
    for entry in data.get("skills") or []:
        manifest.add_skill(
            SkillDef(
                name=entry["name"],
                description=entry.get("description", ""),
                source=entry.get("source", "builtin"),
                dir=entry.get("dir", ""),
                extra_files=tuple(entry.get("extra_files") or ()),
            )
        )
    return manifest


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
