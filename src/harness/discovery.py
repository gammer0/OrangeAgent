"""已发现工具清单与树查询会话状态（ADT-6/7）。

对应文档：``docs/02-数据结构设计/04-工具与权限.md`` §4.6 / §4.7。

约束归属：

===========  ==========================================================
B5           ``TreeSessionState.query_and_record`` 只返回可见子节点
B7           同一 ``path`` 已在 ``delivered`` 中时不得重复下发 schema
B14          ``TreeQueryState`` 与 ``DiscoveredTools`` 的读写必须在同一把锁内
C3（能力）   上下文压缩时 ``reset`` 同步清空 ``delivered`` 与清单
===========  ==========================================================

``TreeSessionState`` 是本模块唯一对外的并发安全门面：所有对
``delivered`` 与 ``DiscoveredTools`` 的读写都经由它，锁不暴露给调用方，
从而在结构上保证 B14。
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

# 文档 §4.2：max_entries = 50
DEFAULT_MAX_DISCOVERED = 50


@dataclass
class ToolDiscovery:
    """清单中的一条记录（文档 §4.2）。

    Attributes:
        path: 完整绝对路径。
        description: 一行描述，≤ 80 字符。
        kind: ``abstract`` / ``leaf``。
        first_seen_step: 首次被 ``tree_query`` 返回的步号。
        last_used_step: 最近一次被返回或被调用的步号（淘汰依据，§4.7）。
    """

    path: str
    description: str
    kind: Literal["abstract", "leaf"]
    first_seen_step: int
    last_used_step: int

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError(f"path 必须是绝对路径：{self.path!r}")
        if "\n" in self.description or "\r" in self.description:
            raise ValueError(f"description 必须单行：{self.path!r}")
        if len(self.description) > 80:
            self.description = self.description[:79] + "…"

    def touch(self, step: int) -> None:
        """更新 ``last_used_step``（§4.7「使用」）。"""
        self.last_used_step = step

    def to_row(self) -> dict[str, str]:
        """投影为临时 system 消息的元素（消息 §3.7）。"""
        return {"path": self.path, "description": self.description}


@dataclass
class DiscoveredTools:
    """会话级已发现工具清单，``entries`` 以 ``path`` 为键。

    Attributes:
        entries: ``path -> ToolDiscovery``。
        max_entries: 容量上限，默认 50。
    """

    entries: dict[str, ToolDiscovery] = field(default_factory=dict)
    max_entries: int = DEFAULT_MAX_DISCOVERED

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, path: object) -> bool:
        return path in self.entries

    def get(self, path: str) -> ToolDiscovery | None:
        """按 path 取记录。"""
        return self.entries.get(path)

    def upsert(self, discovery: ToolDiscovery) -> None:
        """插入或更新一条记录，然后按容量淘汰（§4.7）。

        * 已存在：保留 ``first_seen_step``，更新 ``last_used_step``（取较大值）。
        * 超过 ``max_entries``：按 ``last_used_step`` 升序淘汰，保留新记录
          （新记录的 ``last_used_step`` 等于当前步，通常不会被立刻淘汰）。
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
        """按 ``last_used_step`` 升序淘汰至 ``max_entries`` 以内。"""
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
        """模型调用某 path 时更新 ``last_used_step``。

        Returns:
            ``True`` 表示该 path 在清单中。
        """
        entry = self.entries.get(path)
        if entry is None:
            return False
        entry.touch(step)
        return True

    def remove_prefix(self, path_prefix: str) -> list[str]:
        """卸载时移除前缀下全部记录（B24）。"""
        removed = [
            p for p in self.entries
            if p == path_prefix or p.startswith(path_prefix + "/")
        ]
        for path in removed:
            del self.entries[path]
        return removed

    def clear(self) -> None:
        """清空清单（上下文压缩时，与 ``delivered`` 同步）。"""
        self.entries.clear()

    def rows(self) -> list[dict[str, str]]:
        """投影为临时 system 消息条目，按 ``last_used_step`` 降序（最近的在前）。"""
        ordered = sorted(
            self.entries.values(),
            key=lambda d: (d.last_used_step, d.first_seen_step, d.path),
            reverse=True,
        )
        return [d.to_row() for d in ordered]


@dataclass
class TreeQueryState:
    """会话级记录：已完整下发过 schema 的节点路径集合（§4.2 / §4.6）。

    **不要直接跨线程读写本对象**——请通过 :class:`TreeSessionState`（B14）。

    Attributes:
        delivered: 已作为**子节点**返回过完整 schema 的 path（文档 §4.6 语义）。
        expanded: 已被**查询展开**过的 abstract 节点 path。展开时该节点自身的
            schema 必然已经对外可见（它此前一定作为某个父节点的子节点返回过），
            因此它再次作为子节点出现时可以省略 schema。该集合不在文档字段中，
            是为了同时满足 B7（schema 只下发一次）与 B14（``delivered`` 只记
            子节点）而引入的审计辅助。
    """

    delivered: set[str] = field(default_factory=set)
    expanded: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.delivered)

    def __contains__(self, path: object) -> bool:
        return path in self.delivered

    def mark(self, path: str) -> None:
        """登记某 path 的 schema 已下发（B7）。"""
        self.delivered.add(path)

    def mark_expanded(self, path: str) -> None:
        """登记某节点已被展开查询（其 schema 已可见）。"""
        self.expanded.add(path)

    def was_exposed(self, path: str) -> bool:
        """该 path 的 schema 是否曾经下发过（B7 判定入口）。"""
        return path in self.delivered or path in self.expanded

    def clear(self) -> None:
        """上下文压缩时清空（文档 §4.6）。"""
        self.delivered.clear()
        self.expanded.clear()

    def discard_prefix(self, path_prefix: str) -> int:
        """卸载时移除前缀下全部 path（B24），返回移除数量。"""
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
    """``TreeQueryState`` + ``DiscoveredTools`` 的同锁门面（B14）。

    所有读写都在同一把 ``threading.Lock`` 内完成，因此 ``tree_query`` 可以
    标记为 ``concurrency="safe"``（文档 §4.6）。锁对象私有，不外泄。
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

    # ------------------------------------------------------------ 属性

    @property
    def delivered(self) -> frozenset[str]:
        """``delivered`` 的只读快照（诊断用）。"""
        with self._lock:
            return frozenset(self.state.delivered)

    # ------------------------------------------------------------ 核心

    def query_and_record(
        self,
        tree: ToolTree,
        node: ToolNode,
        step: int = 0,
        *,
        depth: int | None = None,
    ) -> dict[str, Any]:
        """查询子节点、判定并写入 ``delivered``、upsert 清单——**同一把锁内**（B14）。

        Args:
            tree: 所属工具树（权限过滤在锁内重做一次，B5）。
            node: 目标 abstract 节点。
            step: 当前步号。
            depth: 保留参数，暂未使用（子节点总是返回一层）。

        Returns:
            ``{"path": ..., "children": [...]}``，已在 ``delivered`` 中的子节点
            省略 ``schema``（B7）。
        """
        del depth
        with self._lock:
            children = [
                child
                for child in node.children
                if visible(child.path, tree.permissions)  # B5
            ]
            # B7：该子节点此前已被树查询返回过（其 schema 已下发）时省略 schema。
            # 判定与写入都在本锁内完成（B14）。
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

            # 命中后立即登记**各子节点**：下次它作为父节点被查询，或作为
            # 其它父节点的子节点出现时，均可省去 schema（B7）
            for child in children:
                self.state.mark(child.path)
            # 本节点已被展开，其 schema 此后不再重复下发（B7）
            self.state.mark_expanded(node.path)

            return {"path": node.path, "children": payload}

    # ------------------------------------------------------------ 使用

    def record_use(self, path: str, step: int) -> bool:
        """模型调用某 path 时更新 ``last_used_step``（§4.7）。"""
        with self._lock:
            return self.discovered.touch(path, step)

    # ------------------------------------------------------------ 清理

    def reset(self) -> None:
        """上下文压缩时清空 ``delivered`` 与清单——同一把锁（文档 §4.7 清空）。"""
        with self._lock:
            self.state.clear()
            self.discovered.clear()

    def purge_prefix(self, path_prefix: str) -> None:
        """卸载时清理前缀下全部会话缓存（B24）。"""
        with self._lock:
            self.state.discard_prefix(path_prefix)
            self.discovered.remove_prefix(path_prefix)

    # ------------------------------------------------------------ 投影

    def discovered_rows(self) -> list[dict[str, str]]:
        """已发现工具清单投影（供 ``to_provider(discovered=...)`，消息 §3.7）。"""
        with self._lock:
            return self.discovered.rows()
