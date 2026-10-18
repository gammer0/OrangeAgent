"""权限模型（DAC）。

对应文档：``docs/02-数据结构设计/04-工具与权限.md`` §4.3。

模型要点：

* 单主体（agent）+ 三类资源（tool / skill / mcp），类比 Linux owner 位。
* 授予粒度为**实例**：``grants`` 的键是资源绝对路径。
* ``grants`` 只记例外项，**无记录视为允许**（B15）。
* 判定只依赖 path，``side_effect_scope`` 不参与（B18）。

约束归属：

===========  ==========================================================
B5           可见性判定供 ``tree_index`` 与 ``tree_query`` 共用
B15          ``is_allowed`` 对无记录路径返回 True
B16          ``grants`` 的键必须是完整绝对路径，与 ``ToolNode.path`` 逐字符一致
B17          ``Permission.src`` 必须与节点实际资源类型一致
B18          权限判定不看 ``side_effect_scope``
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

# 四类能力位：读 / 写 / 执行 / 网络
CapName = Literal["read", "write", "exec", "net"]

# 资源类型：决定对应哪个权限位（B17）
ResourceKind = Literal["tool", "skill", "mcp"]

RESOURCE_KINDS: frozenset[str] = frozenset({"tool", "skill", "mcp"})

# 路径段：与 message.py 保持同一套字符集，避免两处定义漂移
_PATH_SEGMENT = r"[A-Za-z0-9_\-.]+"
_ABSOLUTE_PATH_RE = re.compile(rf"^/(?:{_PATH_SEGMENT})(?:/{_PATH_SEGMENT})*$")
_HEAD_RE = re.compile(rf"^/({_PATH_SEGMENT})")


def is_absolute_path(path: str) -> bool:
    """判断是否为合法绝对路径（以 ``/`` 开头、至少一段、无空段）。

    B16 要求 ``grants`` 的键是完整绝对路径，且与 ``ToolNode.path`` 逐字符一致；
    本函数只做格式校验，逐字符一致性由 :meth:`check_src_matches_path` 之外的
    查表方式保证（查表本身即逐字符匹配）。
    """
    return isinstance(path, str) and bool(_ABSOLUTE_PATH_RE.match(path))


def path_segments(path: str) -> list[str]:
    """把绝对路径切分为段列表；``"/"`` 视为空列表。

    Raises:
        ValueError: 路径格式非法。
    """
    if path == "/":
        return []
    if not is_absolute_path(path):
        raise ValueError(f"非法绝对路径: {path!r}")
    return path.strip("/").split("/")


def path_head(path: str) -> str | None:
    """返回路径首段（L1 资源层），非绝对路径返回 ``None``。"""
    match = _HEAD_RE.match(path)
    return match.group(1) if match else None


def check_src_matches_path(src: str, path: str) -> None:
    """校验 ``Permission.src`` 与节点路径首段一致（B17）。

    Raises:
        ValueError: 不一致或路径非法。
    """
    head = path_head(path)
    if head is None:
        raise ValueError(f"B16: 非法绝对路径 {path!r}")
    if head != src:
        raise ValueError(f"B17: Permission.src={src!r} 与路径首段 {head!r} 不一致: {path!r}")


class Permission(BaseModel):
    """节点声明的资源类型与能力位。

    Attributes:
        src: 资源类型，决定对应权限位（tool→call、skill→read、mcp→request）。
        caps: 能力位列表，用于并发推导（B11）、幂等标注与风险提示。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    src: ResourceKind
    caps: list[CapName] = Field(default_factory=list)

    @field_validator("caps")
    @classmethod
    def _caps_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"caps 含重复项: {value}")
        return value

    @property
    def permission_bit(self) -> str:
        """本资源类型对应的权限位名称（文档 §4.3）。"""
        return {"tool": "call", "skill": "read", "mcp": "request"}[self.src]


class AgentIdentity(BaseModel):
    """多 agent 预留：当前单实例，``role`` 恒为 ``owner``。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    role: Literal["owner"] = "owner"


class AgentPermissions(BaseModel):
    """按 agent 存的权限表，与共享树分离。

    ``grants`` 的键是资源绝对路径，值是是否允许；**只记例外项**，
    无记录视为允许（B15）。键每次查询都经过绝对路径校验（B16）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    grants: dict[str, bool] = Field(default_factory=dict)

    @field_validator("grants")
    @classmethod
    def _keys_must_be_absolute(cls, value: dict[str, bool]) -> dict[str, bool]:
        for key in value:
            if not is_absolute_path(key):
                raise ValueError(f"B16: grants 的键必须是完整绝对路径，实际为 {key!r}")
        return value

    # ---------------------------------------------------------------- 判定

    def is_allowed(self, path: str) -> bool:
        """判定某绝对路径是否允许。

        只做**精确路径**判定（B16：前缀不匹配、不做祖先继承）。子树隐藏语义
        由 :func:`visible` 承担，供 ``build_tree`` / ``tree_index`` / ``tree_query``
        共用（B5）。

        Args:
            path: 节点绝对路径（须与 ``ToolNode.path`` 逐字符一致，B16）。

        Returns:
            ``True`` 表示允许：无记录默认允许（B15），有记录取记录值。

        Raises:
            ValueError: ``path`` 不是合法绝对路径（B16：禁止相对路径或前缀匹配）。

        """
        if not is_absolute_path(path):
            raise ValueError(f"B16: 权限判定只接受完整绝对路径，实际为 {path!r}")
        return self.grants.get(path, True)

    def is_denied(self, path: str) -> bool:
        """``is_allowed`` 的反面，便于阅读（B5）。"""
        return not self.is_allowed(path)

    def grant(self, path: str, allowed: bool = True) -> AgentPermissions:
        """返回记录了一条例外的新权限表（不可变对象，便于快照）。"""
        if not is_absolute_path(path):
            raise ValueError(f"B16: grants 的键必须是完整绝对路径，实际为 {path!r}")
        return AgentPermissions(
            agent_id=self.agent_id, grants={**self.grants, path: allowed}
        )

    def revoke(self, path: str) -> AgentPermissions:
        """返回移除了该例外记录的新权限表。"""
        return AgentPermissions(
            agent_id=self.agent_id,
            grants={k: v for k, v in self.grants.items() if k != path},
        )


def visible(path: str, permissions: AgentPermissions) -> bool:
    """可见性过滤器：``tree_index``、``tree_query``、``build_tree`` 共用。

    被拒绝的节点不得出现在树索引与查询结果中（B5）；本函数只看 path
    与 ``grants``，不看 ``side_effect_scope``（B18）。

    祖先继承：**任一祖先被显式拒绝，则整棵子树不可见**（B5「拒绝中间节点
    即隐藏整个子树」）。这与 :meth:`AgentPermissions.is_allowed` 的精确判定
    分工不同：``is_allowed`` 只回答"这个 path 本身有没有记录"。
    """
    segments = [seg for seg in path.split("/") if seg]
    for index in range(1, len(segments)):
        ancestor = "/" + "/".join(segments[:index])
        if permissions.grants.get(ancestor) is False:
            return False
    return permissions.is_allowed(path)


class PermissionDenied(Exception):
    """``3a PermissionDenied``：直接调用被拒绝节点的错误。

    对应文档 §4.3 表格右侧：查询时过滤、直接调用时返回 ``3a``（B5）。
    """

    error_type = "3a"

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"3a PermissionDenied: {path}")
