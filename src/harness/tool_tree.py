"""工具树与扁平存储（ADT-4/5）。

对应文档：``docs/02-数据结构设计/04-工具与权限.md``。

核心模型：**树是视图，存储是扁平的**（B8）。

::

    FlatStore（唯一真源，key = tool_id）
        │  按 path 分段聚合
        ▼
    ToolTree（派生视图，不持久化）

约束归属：

===========  ==========================================================
B1           同一父节点下 ``name`` 唯一
B2           每个节点的 ``tool_id`` 必须存在于 FlatStore（含构建期合成条目）
B3           ``leaf`` 节点的 ``schema`` 与对应 ``ToolDef.schema`` 完全一致
B4           ``abstract`` 节点的 ``schema`` 为空对象
B5           被拒绝的节点不出现在树索引与 ``tree_query`` 结果中
B6           树索引默认 L1+L2，L2 超过 40 条按 description 长度升序保留 40
B7           同一 ``path`` 已在 ``delivered`` 中时不得重复下发 schema
B8           FlatStore 唯一真源，ToolTree 不持久化
B9           description 单行
B10          L1 ≤ 40 字符，深层 ≤ 80 字符
B11          concurrency 仅当 caps 只含 read 时为 safe
B12          idempotent 默认 false
B13          所有 abstract 节点 concurrency=safe、idempotent=true
B17          Permission.src 与节点资源类型一致
B18          side_effect_scope 不参与权限判定
B19          path 第一段 ∈ {tool, skill, mcp}
B20          中间节点无显式条目时自动合成 ToolDef，且不写入 FlatStore
B21          kind 不存储，由构建期派生
B22          构建完成后整体校验，任一失败拒绝建树
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

# abstract 节点的 schema 恒为空对象（B4）
EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

# L1 资源层只允许这三种（B19）
RESOURCE_ROOTS: frozenset[str] = frozenset({"tool", "skill", "mcp"})

Concurrency = Literal["safe", "exclusive"]
SideEffectScope = Literal["session", "project", "global"]

L1_DESCRIPTION_LIMIT = 40
DEEP_DESCRIPTION_LIMIT = 80
#: 默认索引深度：``0`` 表示不限制（全展开）。
#:
#: 曾经默认 2，但工具树往往是 ``/tool/<组>/<子组>/<工具>`` 四层深，
#: 结果模型只看到 ``/tool/fluid`` 这种目录名，完全看不到有哪些工具可用——
#: 等于把「有哪些工具」这个最关键的信息省掉了。实测全展开仅增加约 400 token，
#: 远低于模型因此反复试探的代价。
DEFAULT_INDEX_DEPTH = 0
DEFAULT_L2_LIMIT = 120

#: 合成（abstract）节点的占位描述。
#:
#: 早期版本把它直接写进索引，导致每个中间节点都拖一句同样的 6 个字——
#: 在 depth=1 时占索引体积 95%，纯属噪音。现在仅作为内部标记，
#: 建树末尾会被清空（除非分组有显式描述）。
SYNTHETIC_DESCRIPTION = "由子节点推断"


class ToolTreeError(ValueError):
    """工具树构建/校验失败。

    Attributes:
        rule: 违反的约束编号，如 ``"B1"``。
        detail: 人可读描述。
    """

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")


# ---------------------------------------------------------------------------
# 派生规则（B11 / B12 / B10）
# ---------------------------------------------------------------------------


def derive_concurrency(caps: Sequence[str]) -> Concurrency:
    """按 §4.13 推导 ``concurrency``（B11）。

    * ``["read"]`` / ``["read", "net"]`` → ``safe``
    * 含 ``write`` / ``exec``，或 ``["net"]`` 单写，或未声明 → ``exclusive``
    """
    if not caps:
        return "exclusive"
    if "write" in caps or "exec" in caps:
        return "exclusive"
    if "read" in caps:
        return "safe"
    return "exclusive"


def truncate_description(description: str, limit: int) -> str:
    """按字符数截断描述并附省略号（B10）。"""
    if len(description) <= limit:
        return description
    return description[: limit - 1] + "…"


def _reject_multiline(description: str) -> str:
    """B9：description 必须单行。"""
    if "\n" in description or "\r" in description:
        raise ValueError(f"B9: description 必须单行，实际含换行: {description!r}")
    return description


def _limit_for_path(path: str) -> int:
    """L1（``/tool``）40 字符，深层 80 字符（B10）。"""
    depth = len(path_segments(path))
    return L1_DESCRIPTION_LIMIT if depth <= 1 else DEEP_DESCRIPTION_LIMIT


# ---------------------------------------------------------------------------
# ToolDef（ADT-4）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDef:
    """扁平存储中的一条工具定义（ADT-4，字段严格按文档 §4.2）。

    用 dataclass 而非 pydantic：文档要求字段名为 ``schema``，而 pydantic 的
    ``BaseModel`` 保留同名成员，会产生命名冲突。

    推导与校验：

    * ``concurrency`` 未显式提供时由 ``caps`` 推导（B11），提供但不一致则拒绝；
    * ``idempotent`` 默认 ``False``（B12），``write``/``exec`` 组合不得标 ``True``；
    * ``permission.src`` 必须与 path 首段一致（B17）；
    * ``path`` 首段必须∈ ``{tool, skill, mcp}``（B19）；
    * ``description`` 必须单行（B9）。
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
            raise ValueError("tool_id 不得为空")

        # B19：path 第一段必须是资源层
        head = path_head(self.path)
        if head is None:
            raise ValueError(f"B16/B19: 非法绝对路径 {self.path!r}")
        if head not in RESOURCE_ROOTS:
            raise ValueError(f"B19: path 第一段必须∈ {{tool, skill, mcp}}，实际为 {head!r}")

        # B17：src 与路径首段一致
        if self.permission.src != head:
            raise ValueError(
                f"B17: Permission.src={self.permission.src!r} 与路径首段 {head!r} 不一致"
            )

        # B9：单行
        _reject_multiline(self.description)

        derived_concurrency = derive_concurrency(self.caps)
        if self.concurrency is None:
            object.__setattr__(self, "concurrency", derived_concurrency)
        elif self.concurrency != derived_concurrency:
            raise ValueError(
                f"B11: caps={self.caps} 推导出的 concurrency 为 {derived_concurrency!r}，"
                f"与声明的 {self.concurrency!r} 不符"
            )

        # B12：``write`` / ``exec`` 组合不得标记 idempotent=true
        if self.idempotent and not self.synthetic:
            mutating = {"write", "exec"} & set(self.caps)
            if mutating:
                raise ValueError(
                    f"B12: caps={self.caps} 含 {sorted(mutating)}，不得标记 idempotent=true"
                )

    def with_retired(self, retired: bool = True) -> ToolDef:
        """返回 ``retired`` 标记变更后的副本（B23：卸载只标记不删除）。"""
        return dataclasses.replace(self, retired=retired)

    # ------------------------------------------------------------ 工厂方法

    @classmethod
    def synthesized(cls, path: str, parent: ToolDef | None = None) -> ToolDef:
        """按 §4.9.2 合成中间节点的 ``ToolDef``（B20）。

        合成条目只存在于构建期的树中，**不写入 FlatStore**。

        Args:
            path: 中间节点路径。
            parent: 父节点条目；``permission`` 与 ``side_effect_scope`` 自此继承（B17）。
        """
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
        head = path_head(path)
        if head not in RESOURCE_ROOTS:
            raise ToolTreeError("B19", f"无法为 {path!r} 合成条目：首段非法")
        return cls(
            tool_id=f"t_tree_{digest}",
            path=path,
            fn=None,  # 执行期由调度层绑定 tree_query(node)
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
# FlatStore（ADT-4）
# ---------------------------------------------------------------------------


class FlatStore:
    """扁平存储：唯一真源，``entries`` 以 ``tool_id`` 为键（B8）。

    每条 entry 的 ``path`` 必须与键唯一对应：同一路径不允许两条条目，
    否则会破坏"末段绑定单一 tool_id"的语义。
    """

    def __init__(self, entries: Iterable[ToolDef] | None = None) -> None:
        self._entries: dict[str, ToolDef] = {}
        for entry in entries or ():
            self.put(entry)

    # -- 查询 --------------------------------------------------------

    @property
    def entries(self) -> Mapping[str, ToolDef]:
        """只读视图：``tool_id -> ToolDef``。"""
        return _ReadOnlyMapping(self._entries)

    def get(self, tool_id: str) -> ToolDef | None:
        """按 ``tool_id`` 取条目。"""
        return self._entries.get(tool_id)

    def by_path(self, path: str) -> ToolDef | None:
        """按完整路径取条目（供 ``from_path`` 解析使用）。"""
        for entry in self._entries.values():
            if entry.path == path:
                return entry
        return None

    def active_paths(self) -> dict[str, ToolDef]:
        """按 path 索引的未卸载条目（``retired=False``）。"""
        return {e.path: e for e in self._entries.values() if not e.retired}

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ToolDef]:
        return iter(list(self._entries.values()))

    # -- 挂载（§4.10.1）---------------------------------------------

    def put(self, entry: ToolDef) -> ToolDef:
        """写入单条目。

        Raises:
            ToolTreeError: 违反 B19（首段非法）、B9/B10（描述非法）或路径冲突。
        """
        head = path_head(entry.path)
        if head is None or head not in RESOURCE_ROOTS:
            raise ToolTreeError(
                "B19", f"拒绝挂载：path 首段必须∈ {{tool, skill, mcp}}，{entry.path!r}"
            )
        _check_description(entry.description, entry.path)
        existing = self._entries.get(entry.tool_id)
        if existing is not None:
            raise ToolTreeError("B8", f"tool_id 已存在: {entry.tool_id!r}")
        for other in self._entries.values():
            if other.path == entry.path and not other.retired:
                raise ToolTreeError(
                    "B1", f"路径已被占用: {entry.path!r}（tool_id={other.tool_id!r}）"
                )
        self._entries[entry.tool_id] = entry
        return entry

    def mark_retired(self, path_prefix: str, *, exact: bool = False) -> list[str]:
        """把前缀下全部条目标记 ``retired=True``（B23：不删除）。

        Args:
            path_prefix: 绝对路径前缀。
            exact: 为 ``True`` 时只标记该路径本身。

        Returns:
            被标记的 ``tool_id`` 列表。
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
    """包裹 dict 的只读视图，避免外部直接污染真源。"""

    def __init__(self, data: dict[str, ToolDef]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> ToolDef:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _check_description(description: str, path: str) -> None:
    """B9 单行 + B10 长度（超长在装配时截断，此处只拒绝换行）。"""
    del path
    _reject_multiline(description)


# ---------------------------------------------------------------------------
# ToolNode（ADT-5）
# ---------------------------------------------------------------------------


@dataclass
class ToolNode:
    """派生视图节点，不独立存储。

    ``kind`` 是**派生属性**，不是构造参数（B21）：有 ``children`` 即
    ``abstract``，否则 ``leaf``。
    """

    path: str
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""
    children: list[ToolNode] = field(default_factory=list)
    permission: Permission | None = None
    side_effect_scope: SideEffectScope = "session"

    # ---------------------------------------------------------- 结构操作

    def child(self, name: str) -> ToolNode | None:
        """按 ``name`` 查找直接子节点（B1 保证唯一）。"""
        for node in self.children:
            if node.name == name:
                return node
        return None

    def add_child(self, node: ToolNode) -> None:
        """追加子节点；同名冲突由 :meth:`sort_children` 之前的校验拦截。"""
        self.children.append(node)

    def sort_children(self) -> None:
        """按 ``path`` 字典序重排整棵子树（B6 要求索引有序）。"""
        self.children.sort(key=lambda n: n.path)
        for node in self.children:
            node.sort_children()

    def walk(self) -> Iterator[ToolNode]:
        """先序遍历整棵子树（含自身）。"""
        yield self
        for node in self.children:
            yield from node.walk()

    # ---------------------------------------------------------- 派生属性

    @property
    def kind(self) -> Literal["abstract", "leaf"]:
        """B21：有 children → abstract，否则 leaf。"""
        return "abstract" if self.children else "leaf"

    @property
    def is_leaf(self) -> bool:
        """是否为具体工具。"""
        return not self.children

    @property
    def depth(self) -> int:
        """路径深度（``/`` 为 0）。"""
        return 0 if self.path == "/" else len(path_segments(self.path))

    def child_index(self) -> list[dict[str, Any]]:
        """不含 schema 的子节点摘要（B7：已下发过 schema 时使用）。"""
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
        """含 schema 的子节点完整结构（§4.5 返回格式）。"""
        return [
            {**summary, "schema": child.schema}
            for summary, child in zip(self.child_index(), self.children, strict=True)
        ]


# ---------------------------------------------------------------------------
# ToolTree（ADT-5）
# ---------------------------------------------------------------------------


@dataclass
class ToolTree:
    """从 FlatStore 派生的视图，不独立持久化（B8）。

    Attributes:
        root: 根节点 ``/``。
        version: 每次变更递增（B25）。
        permissions: 构建时使用的权限表，索引与查询时继续用于过滤（B5）。
        flat_store: 反向引用，供 ``tool_id`` 校验与执行期解析（B2）。
        tree_query_fn: 所有 abstract 条目共享的 ``tree_query`` 实现（§4.5）。
        synthetic_entries: 构建期合成的条目（按 path 索引，不写入 FlatStore，B20）。
    """

    root: ToolNode = field(default_factory=lambda: ToolNode(path="/", name=""))
    version: int = 1
    permissions: AgentPermissions = field(
        default_factory=lambda: AgentPermissions(agent_id="local")
    )
    flat_store: FlatStore = field(default_factory=FlatStore)
    tree_query_fn: Callable[..., Any] | None = None
    synthetic_entries: dict[str, ToolDef] = field(default_factory=dict)

    # ---------------------------------------------------------- 查询

    def get_node(self, path: str) -> ToolNode | None:
        """按绝对路径查找节点；不存在或不可见返回 ``None``（B5）。"""
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
            # 中间节点若被拒绝，子树整体不可见
        return node if visible(node.path, self.permissions) else None

    def visible_nodes(self, depth: int | None = None) -> list[ToolNode]:
        """按 ``path`` 字典序列出可见节点，可限制最大深度。"""
        return [
            node
            for node in self.root.walk()
            if node.path != "/" and visible(node.path, self.permissions)
            and (depth is None or node.depth <= depth)
        ]

    def tree_index(self, depth: int = DEFAULT_INDEX_DEPTH) -> str:
        """树索引投影（B6）：按 ``depth`` 展开层级，按 path 字典序。

        可见性规则（**分组必须自证价值**）：

        * **叶子**（真正可调用的工具/skill）一律出现。
        * **中间节点**只在其有**显式描述**时出现——因为只有描述能提供叶子
          路径之外的信息（用途说明）。无描述的中间节点只是把子路径的公共
          前缀重复一遍，属纯噪音，一律省略。
        * 例外：L1 资源根（``/tool``、``/skill``、``/mcp``）恒定出现，
          它们标出三类能力的边界。

        其余规则：

        * 被拒绝的节点不出现（B5）。
        * ``depth`` 表示**展开到第几层**（1 = 只列 L1；``0`` = 不限制）。
        * 条目总数超过 ``DEFAULT_L2_LIMIT`` 时按描述长度升序保留前若干条。
        * 结果只由树结构与权限决定，同一会话内逐字节一致。
        """
        ordered = self.visible_nodes()
        if depth and depth > 0:
            shown = [n for n in ordered if n.depth <= depth]
        else:
            shown = list(ordered)

        # 分组只在有描述时保留；叶子与 L1 资源根恒定保留
        shown = [
            n
            for n in shown
            if n.is_leaf or n.depth == 1 or (n.description or "").strip()
        ]

        if len(shown) > DEFAULT_L2_LIMIT:
            # 保留所有 L1 与有描述的分组，剩余名额按描述长度（短优先）挑选
            keep = [
                n for n in shown if n.depth == 1 or not n.is_leaf
            ]
            rest = sorted(
                (n for n in shown if n.depth > 1 and n.is_leaf),
                key=lambda n: (len(n.description), n.path),
            )
            budget = max(DEFAULT_L2_LIMIT - len(keep), 0)
            shown = sorted(keep + rest[:budget], key=lambda n: n.path)

        lines = []
        for node in shown:
            description = (node.description or "").strip()
            if description:
                limit = _limit_for_path(node.path)
                lines.append(f"{node.path} — {truncate_description(description, limit)}")
            else:
                lines.append(node.path)

        # 折叠掉的叶子不能凭空消失——否则模型永远不知道它们存在，也没法
        # tree_query 出来。按所属分组聚合提示，但只列前几个，避免提示本身
        # 变成新的噪音（工具数多时分组可能上百）。
        #
        # 注意：深度限制与折叠可能同时发生，两者都是「去上级路径找」，
        # 所以合并成一条提示，不重复表达同一件事。
        shown_set = {n.path for n in shown}
        folded = [n for n in ordered if n.is_leaf and n.path not in shown_set]
        depth_hidden = [
            n for n in ordered if depth and depth > 0 and n.depth > depth
        ]

        if folded:
            buckets: dict[str, int] = {}
            for node in folded:
                parent = node.path.rsplit("/", 1)[0] or "/"
                buckets[parent] = buckets.get(parent, 0) + 1
            listed = sorted(buckets.items())[:5]
            detail = "、".join(
                f"{path}" + (f"（{count} 个）" if count > 1 else "")
                for path, count in listed
            )
            if len(buckets) > len(listed):
                detail += f" 等 {len(buckets)} 个位置"
            lines.append(
                f"（另有 {len(folded)} 个工具未展开，可用 tree_query 展开：{detail}）"
            )
        elif depth_hidden:
            lines.append(
                f"（另有 {len(depth_hidden)} 个更深节点，用 tree_query 按上级路径展开）"
            )
        return "\n".join(lines)

    @property
    def paths(self) -> list[str]:
        """全部可见节点路径（按字典序）。"""
        return [n.path for n in self.visible_nodes()]

    # ---------------------------------------------------------- 执行解析

    def resolve(self, path: str) -> ToolDef:
        """``path → ToolNode → tool_id → ToolDef``（§4.5 调用链）。

        Raises:
            PermissionDenied: 路径被权限拒绝（B5，返回 ``3a``）。
            KeyError: 路径不存在。
        """
        from harness.permission import PermissionDenied

        if not visible(path, self.permissions):
            raise PermissionDenied(path)
        node = self.get_node(path)
        if node is None:
            raise KeyError(f"工具路径不存在: {path!r}")
        entry = self.flat_store.get(node.tool_id) or self.synthetic_entries.get(node.path)
        if entry is None:
            raise KeyError(f"tool_id={node.tool_id!r} 不在 FlatStore 中（B2）")
        return entry

    # ---------------------------------------------------------- 变更（B25）

    def bump(self) -> None:
        """``version += 1``（B25）。"""
        self.version += 1


# ---------------------------------------------------------------------------
# tree_query（§4.5 / §4.6 / §4.7）
# ---------------------------------------------------------------------------


def tree_query(
    node: ToolNode | str,
    *,
    tree: ToolTree,
    state: TreeSessionState | None = None,
    step: int = 0,
) -> dict[str, Any]:
    """查询 abstract 节点的下一级子节点（§4.5）。

    * 返回 ``{"path": ..., "children": [...]}``，子节点仅含可见项（B5）。
    * 若 ``state`` 提供：在同一把锁内完成「判定 delivered + 写入 delivered +
      upsert DiscoveredTools」（B7 / B14）。
    * 已在 ``delivered`` 中的子节点省略 ``schema``（B7）。
    * 本函数不维护自身状态，线程安全性由 :class:`TreeSessionState` 的锁保证。

    Args:
        node: 节点对象或绝对路径。
        tree: 所属工具树（用于权限过滤与 state 绑定校验）。
        state: 会话级查询状态（可选；不传则每次都下发完整 schema）。
        step: 当前步号，写入 ``first_seen_step`` / ``last_used_step``。
    """
    target: ToolNode
    if isinstance(node, str):
        found = tree.get_node(node)
        if found is None:
            from harness.permission import PermissionDenied

            # 节点不在树中：可能是权限拒绝，也可能是路径不存在。
            # 被显式拒绝的路径按 B5 返回 3a，而不是暴露"存在与否"的差异。
            if node in tree.permissions.grants and tree.permissions.grants[node] is False:
                raise PermissionDenied(node)
            raise KeyError(f"abstract 节点不存在或不可见: {node!r}")
        target = found
    else:
        target = node
        # 防止把任意节点对象塞进来绕过权限（B5）
        checked = tree.get_node(target.path)
        if checked is None:
            from harness.permission import PermissionDenied

            raise PermissionDenied(target.path)
        target = checked

    if state is None:
        # 无会话状态时每次都下发完整 schema；权限过滤在 full_index 前完成
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
# 构建（§4.9）
# ---------------------------------------------------------------------------


def build_tree(
    flat_store: FlatStore,
    permissions: AgentPermissions | None = None,
    *,
    group_descriptions: Mapping[str, str] | None = None,
) -> ToolTree:
    """按 §4.9.1 从扁平存储构建工具树。

    算法：

    1. 按 ``path`` 字典序遍历未卸载（``retired=False``）条目；
    2. 权限过滤（B5）——被拒绝的条目及其子树不进入树；
    3. 沿路径分段 ``get_or_create`` 中间节点；
    4. 末段绑定 ``tool_id`` / ``schema`` / ``permission``；
    5. 中间节点无显式条目时自动合成 ``ToolDef``（B20，不写入 FlatStore）；
    6. 派生 ``kind``（B21）并整体校验（B22）。

    Args:
        flat_store: 扁平存储（唯一真源）。
        permissions: 权限表；省略时为空表（全部允许）。
        group_descriptions: ``group_id -> 描述``，如 ``{"tool/os": "本机文件"}``。
            中间节点优先用这里的描述；未提供时留空（索引只显示路径，不占 token）。

    Raises:
        ToolTreeError: 任一构建期校验（B1/B2/B3/B4/B9/B10/B19）失败，拒绝建树。
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

    # ---- 1) 前置校验：路径格式与描述（B19 / B9 / B10 / B16）------------
    for entry in active:
        _validate_entry_metadata(entry)

    # ---- 2) 分段聚合 ---------------------------------------------------
    bound: dict[str, ToolDef] = {}
    for entry in active:
        if not perms.is_allowed(entry.path):
            continue  # B5：被拒绝的条目本身不入树（精确判定，B16）
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
                f"路径 {node.path!r} 绑定了多个 tool_id："
                f"{bound[node.path].tool_id!r} 与 {entry.tool_id!r}",
            )
        node.schema = entry.schema
        node.tool_id = entry.tool_id
        node.description = entry.description
        node.permission = entry.permission
        node.side_effect_scope = entry.side_effect_scope
        bound[node.path] = entry

    tree.root.sort_children()

    # ---- 3) 中间节点合成（B20）------------------------------------------
    synthetic = _synthesize_prefixes(
        active,
        bound,
        perms,
        existing_paths=[node.path for node in tree.root.walk()],
    )

    # ---- 4) abstract 节点的 schema 归一（B4）与描述截断（B10）------------
    #        未绑定显式条目的中间节点在此拿到合成 tool_id（B2 / B20）
    for node in tree.root.walk():
        if node.path == "/":
            node.permission = Permission(src="tool", caps=["read"])
            node.schema = dict(EMPTY_SCHEMA)
            node.description = ""
            continue
        if node.path not in bound and node.path in synthetic:
            synth = synthetic[node.path]
            node.tool_id = synth.tool_id
            # B17：合成节点继承父节点的 Permission
            node.permission = synth.permission
            node.side_effect_scope = synth.side_effect_scope
            node.description = synth.description
        if node.kind == "abstract":
            node.schema = dict(EMPTY_SCHEMA)
            # 分组描述优先用显式声明；未声明则留空，索引只显示路径
            declared = (group_descriptions or {}).get(node.path.lstrip("/"))
            if declared:
                node.description = declared
            elif node.description == SYNTHETIC_DESCRIPTION:
                node.description = ""
        node.description = truncate_description(node.description, _limit_for_path(node.path))

    # 合成条目只存在于树的私有视图（不进 FlatStore，B20），供 resolve 与校验使用
    tree.synthetic_entries = synthetic

    _validate_tree(tree, flat_store, synthetic)
    return tree


def _validate_entry_metadata(entry: ToolDef) -> None:
    """B19 / B9 / B10 / B16：单条目元数据校验。"""
    head = path_head(entry.path)
    if head is None:
        raise ToolTreeError("B16", f"非法绝对路径：{entry.path!r}")
    if head not in RESOURCE_ROOTS:
        raise ToolTreeError(
            "B19", f"path 首段必须∈ {{tool, skill, mcp}}，实际为 {head!r}：{entry.path!r}"
        )
    if "\n" in entry.description or "\r" in entry.description:
        raise ToolTreeError("B9", f"description 必须单行：{entry.path!r}")
    if len(entry.description) > _limit_for_path(entry.path):
        # B10：超长由装配期截断，这里不拒绝
        return


def _synthesize_prefixes(
    active: list[ToolDef],
    bound: dict[str, ToolDef],
    permissions: AgentPermissions,
    *,
    existing_paths: Iterable[str] = (),
) -> dict[str, ToolDef]:
    """为无显式条目的中间节点合成 ``ToolDef``（B20），返回 path→条目。

    合成条目的 ``permission`` / ``side_effect_scope`` 继承父节点（B17）；
    L1 节点无父条目时使用资源层默认条目。不可见的前缀不合成（B5）：
    拒绝中间节点即隐藏整个子树。

    Args:
        active: 全部未卸载条目。
        bound: 已绑定显式条目的 path → 条目（已过权限过滤）。
        permissions: 权限判定器。
        existing_paths: 树中已存在的全部路径（用于补齐 L1 根节点，B2）。
    """
    synthetic: dict[str, ToolDef] = {}
    # 合成依据是**树中实际存在的节点**，而非 bound（被拒条目也会留下父链节点）
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
    """L1 中间节点无父条目时的合成基底。"""
    head = path_head(path)
    if head not in RESOURCE_ROOTS:
        raise ToolTreeError("B19", f"无法合成 L1 节点：{path!r}")
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
    """构建期整体校验（B22）。任一失败抛 ``ToolTreeError``。"""
    # 允许调用方省略 synthetic（例如仅校验树结构），此时回退到树自带的合成条目
    if not synthetic:
        synthetic = tree.synthetic_entries

    for node in tree.root.walk():
        if node.path == "/":
            continue

        # B1：同一父节点下 name 唯一（最先检查，避免被 B2 掩盖）
        names = [child.name for child in node.children]
        if len(set(names)) != len(names):
            duplicated = sorted({n for n in names if names.count(n) > 1})
            raise ToolTreeError("B1", f"{node.path!r} 下存在重名子节点：{duplicated}")

        # B2：tool_id 必须存在于 FlatStore 或合成表
        if not node.tool_id:
            raise ToolTreeError("B2", f"节点缺少 tool_id：{node.path!r}")
        entry = store.get(node.tool_id) or synthetic.get(node.path)
        if entry is None or entry.tool_id != node.tool_id:
            raise ToolTreeError(
                "B2", f"节点 {node.path!r} 的 tool_id={node.tool_id!r} 不在 FlatStore 中"
            )

        # B4：abstract schema 为空对象
        if node.kind == "abstract":
            if node.schema != EMPTY_SCHEMA:
                raise ToolTreeError(
                    "B4", f"abstract 节点 {node.path!r} 的 schema 必须为空对象"
                )
        # B3：leaf schema 与 ToolDef.schema 一致
        else:
            if node.schema != entry.schema:
                raise ToolTreeError(
                    "B3", f"leaf 节点 {node.path!r} 的 schema 与 ToolDef.schema 不一致"
                )

        # B17：permission.src 与路径首段一致
        if node.permission is None:
            raise ToolTreeError("B17", f"节点 {node.path!r} 缺少 permission")
        try:
            check_src_matches_path(node.permission.src, node.path)
        except ValueError as exc:
            raise ToolTreeError("B17", str(exc)) from exc

        # B9 / B10
        if "\n" in node.description or "\r" in node.description:
            raise ToolTreeError("B9", f"description 必须单行：{node.path!r}")

    # 合成条目不得写入 FlatStore（B20）；同时每个合成条目必须带 synthetic=True
    for path, entry in synthetic.items():
        if any(e.path == path and not e.retired for e in store):
            # 有显式条目的路径不会合成；落入此分支说明合成逻辑错误
            raise ToolTreeError("B20", f"合成条目与 FlatStore 冲突：{path!r}")
        if not entry.synthetic:
            raise ToolTreeError("B20", f"合成条目必须标记 synthetic=True：{path!r}")


# ---------------------------------------------------------------------------
# 索引解析辅助
# ---------------------------------------------------------------------------


def index_text_to_paths(text: str) -> list[str]:
    """从 ``tree_index`` 文本中提取路径（测试与回放用）。

    同时处理两种行：``路径 — 描述`` 与只有 ``路径`` 的行（无描述的节点）。
    以 ``（`` 开头的提示行不含路径，跳过。

    历史 bug：早期只解析含 `` — `` 的行，导致无描述的路径（L1 资源根、
    无描述分组）被静默丢弃，用它做的断言会得出错误结论。
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
