"""服务管理：把分散的运行时组件组装为一个可查询、可启停的服务实例。

对应职责（不是新的业务概念，而是**组合根**）：

* 统一持有 ``FlatStore`` / ``ToolTree`` / ``SessionRegistry`` / ``SessionStore`` /
  ``ReverseProxy`` / ``ContextAssembler`` 等组件的生命周期；
* 提供只读状态快照（`ServiceStatus`），供 CLI 与健康检查使用；
* 提供受限的运维动作（驱逐空闲会话、关闭持久化连接）。

设计约束：

* **不引入新不变量**：所有约束仍由各模块守护，本模块只做组装与观测。
* **状态快照必须无副作用**：``status()`` 不触发 LLM 调用、不写库、不清缓存。
* **失败即数据**：组件初始化失败不抛到调用方，而是记录在快照的 ``problems`` 中，
  让 CLI 能报告"哪一层没起来"而不是整个进程崩掉。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.capabilities import (
    CapabilityError,
    CapabilityManifest,
    load_manifest,
)
from harness.config import ConfigError, LLMConfig, build_proxy, load_llm_config
from harness.context import ContextAssembler
from harness.defaults import (
    DEFAULT_CONTEXT,
    DEFAULT_EXECUTION,
    DEFAULT_SESSIONS,
    DEFAULT_TOOLS,
)
from harness.discovery import DiscoveredTools, TreeQueryState, TreeSessionState
from harness.memory import ensure_working_memory
from harness.model import ReverseProxy
from harness.permission import AgentPermissions
from harness.permission_store import LOCAL_AGENT, load_permissions
from harness.session import EvictionPolicy, SessionRegistry
from harness.session_store import SessionStore
from harness.tool_tree import FlatStore, ToolTree, build_tree
from harness.trace import TraceLogger

__all__ = [
    "ServiceState",
    "ServiceStatus",
    "HarnessService",
    "build_service",
]

ServiceState = str  # "not_loaded" | "ready" | "degraded" | "failed"


@dataclass(frozen=True, slots=True)
class LayerStatus:
    """单个组件的状态。

    Attributes:
        name: 组件名（与模块对应）。
        state: ``ready`` / ``degraded`` / ``failed``。
        detail: 人可读说明；失败时为错误原因（不含密钥）。
        metrics: 该组件的关键计数。
    """

    name: str
    state: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """服务整体快照（只读、无副作用）。

    Attributes:
        workdir: 工作目录。
        state: 整体状态，取各层最差结果。
        layers: 各组件状态列表。
        problems: 初始化阶段收集的问题。
        captured_at: 快照时间戳。
    """

    workdir: str
    state: str
    layers: list[LayerStatus]
    problems: list[str] = field(default_factory=list)
    captured_at: float = 0.0

    @property
    def ok(self) -> bool:
        """是否全部组件就绪。"""
        return self.state == "ready"

    def layer(self, name: str) -> LayerStatus | None:
        """按名称取某层状态。"""
        for item in self.layers:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """导出为可 JSON 序列化的字典（CLI ``--json`` 使用）。"""
        return {
            "workdir": self.workdir,
            "state": self.state,
            "ok": self.ok,
            "captured_at": self.captured_at,
            "problems": list(self.problems),
            "layers": [
                {
                    "name": layer.name,
                    "state": layer.state,
                    "detail": layer.detail,
                    "metrics": dict(layer.metrics),
                }
                for layer in self.layers
            ],
        }


class HarnessService:
    """服务组合根。

    构造时**不抛异常**：各组件的初始化失败会记录在 ``self.problems``，
    对应层状态置为 ``failed`` 或 ``degraded``，便于 CLI 逐层报告。

    Attributes:
        workdir: 工作目录（``.env``、``.agent/`` 均相对它）。
        config: LLM 配置；加载失败时为 ``None``。
        flat_store / tree: 工具能力层。
        registry: 会话注册表。
        store: SQLite 持久化。
        proxy: 模型反向代理。
        assembler: 上下文装配器。
    """

    def __init__(
        self,
        workdir: str | Path = ".",
        *,
        capacity: int = DEFAULT_CONTEXT.capacity,
        load_model: bool = True,
        global_dir: str | Path | None = None,
    ) -> None:
        """组装服务。

        Args:
            workdir: 工作目录；``.env`` 与 ``.agent/`` 都相对它解析。
            capacity: 上下文容量（token）。
            load_model: 是否加载模型通道。``False`` 时只做本地状态检查，
                不校验 API Key，也不建立 HTTP 客户端。
            global_dir: 全局目录（长期记忆所在）；省略时取工作目录。
        """
        self.workdir = Path(workdir).resolve()
        self.global_dir = Path(global_dir).resolve() if global_dir else self.workdir
        self.capacity = capacity
        self.problems: list[str] = []
        self.started_at = time.time()

        self.config: LLMConfig | None = None
        self.flat_store: FlatStore | None = None
        self.tree: ToolTree | None = None
        #: 按 ``.agent/capabilities.json`` 装载的能力清单；未装载时为 None。
        self.capabilities: CapabilityManifest | None = None
        self.permissions: AgentPermissions = AgentPermissions(agent_id=LOCAL_AGENT)
        self.registry: SessionRegistry | None = None
        self.store: SessionStore | None = None
        self.proxy: ReverseProxy | None = None
        self.assembler: ContextAssembler | None = None
        #: 工作记忆文件路径与本次是否新建（诊断用）。
        self.work_memory: Path | None = None
        self.work_memory_created = False

        #: 是否尝试加载模型通道；``False`` 时该层状态为 ``not_loaded`` 而非降级
        self.model_requested = load_model

        self._ensure_work_memory()
        self._load_capability()
        self._load_session()
        self._load_context()
        if load_model:
            self._load_model()

    # ------------------------------------------------------------ 装配

    def _ensure_work_memory(self) -> None:
        """确保 ``<workdir>/.agent/WORK.md`` 存在（缺失时用模板创建）。

        系统提示会告诉模型「本项目存在 ``.agent/WORK.md``」，但之前从未真的
        创建它——模型第一次去读会拿到「文件不存在」，与提示词自相矛盾。

        **失败不阻止启动**：工作记忆是辅助手段，没有它 agent 仍可工作，
        因此只记入 ``problems`` 而不抛异常。
        """
        try:
            path, created = ensure_working_memory(self.workdir)
            self.work_memory = path
            self.work_memory_created = created
        except Exception as exc:  # noqa: BLE001 - 降级而非崩溃
            self.problems.append(f"work_memory: {exc}")


    def _load_capability(self) -> None:
        """装配工具能力层（扁平存储 + 派生树）。

        ponytail: 必须读 ``.agent/capabilities.json``，否则 CLI 看到的是一棵空树，
        而 Web 服务按清单装载了实际工具——两边观测到的能力面不一致。
        清单不存在或非法时退回空树（保守降级，与 I4 一致）。

        权限从 ``.agent/permissions.json`` 读取（agent_id=``local``），
        使 CLI 与运行时看到同一套屏蔽结果；文件缺失时为空表（全部允许）。
        """
        try:
            self.flat_store = FlatStore()
            manifest_problems: list[str] = []
            try:
                manifest = load_manifest(self.workdir, problems=manifest_problems)
            except CapabilityError as exc:
                self.problems.append(f"capability: {exc}")
                manifest = None
            self.problems.extend(manifest_problems)
            if manifest is not None:
                for entry in manifest.to_tooldefs(str(self.workdir)):
                    self.flat_store.put(entry)
                self.capabilities = manifest
            self.permissions = load_permissions(
                self.workdir, LOCAL_AGENT, problems=self.problems
            )
            group_descriptions = (
                {g.group_id: g.description for g in self.capabilities.groups}
                if self.capabilities is not None
                else {}
            )
            self.tree = build_tree(
                self.flat_store,
                self.permissions,
                group_descriptions=group_descriptions,
            )
        except Exception as exc:  # noqa: BLE001 - 组合根必须降级而非崩溃
            self.problems.append(f"capability: {exc}")
            self.flat_store = None
            self.tree = None

    def _load_session(self) -> None:
        """装配会话注册表与 SQLite 持久化。"""
        try:
            self.registry = SessionRegistry(
                EvictionPolicy(
                    max_resident=DEFAULT_SESSIONS.max_resident,
                    idle_timeout_s=DEFAULT_SESSIONS.idle_timeout_s,
                    sweep_interval_s=DEFAULT_SESSIONS.sweep_interval_s,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"session-registry: {exc}")
            self.registry = None

        try:
            self.store = SessionStore(self.workdir)
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"session-store: {exc}")
            self.store = None

    def _load_context(self) -> None:
        """装配上下文装配器（不注入 summarizer，压缩按 I4 无损跳过）。"""
        try:
            self.assembler = ContextAssembler(capacity=self.capacity)
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"context: {exc}")
            self.assembler = None

    def _load_model(self) -> None:
        """装配模型通道（密钥从环境变量读取）。"""
        try:
            self.config = load_llm_config(dotenv_path=self.workdir / ".env")
            self.proxy = build_proxy(self.config)
        except ConfigError as exc:
            # 配置缺失是可预期状态：降级而不是失败
            self.problems.append(f"model: {exc}")
            self.config = None
            self.proxy = None
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"model: {exc}")
            self.config = None
            self.proxy = None

    # ------------------------------------------------------------ 观测

    def status(self) -> ServiceStatus:
        """返回只读状态快照（无副作用）。"""
        layers = [
            self._capability_layer(),
            self._session_layer(),
            self._store_layer(),
            self._context_layer(),
            self._model_layer(),
        ]
        return ServiceStatus(
            workdir=str(self.workdir),
            state=_worst_state(layers),
            layers=layers,
            problems=list(self.problems),
            captured_at=time.time(),
        )

    def _capability_layer(self) -> LayerStatus:
        if self.flat_store is None or self.tree is None:
            return LayerStatus("capability", "failed", self._problem_of("capability"))
        active = self.flat_store.active_paths()
        return LayerStatus(
            "capability",
            "ready",
            metrics={
                "tools": len(active),
                "nodes": len(self.tree.paths),
                "tree_version": self.tree.version,
                "tree_enabled": len(active) >= DEFAULT_TOOLS.tree_enable_threshold,
            },
        )

    def _session_layer(self) -> LayerStatus:
        if self.registry is None:
            return LayerStatus("session", "failed", self._problem_of("session-registry"))
        return LayerStatus(
            "session",
            "ready",
            metrics={
                "resident": len(self.registry),
                "max_resident": DEFAULT_SESSIONS.max_resident,
                "session_ids": self.registry.session_ids(),
            },
        )

    def _store_layer(self) -> LayerStatus:
        if self.store is None:
            return LayerStatus("store", "failed", self._problem_of("session-store"))
        try:
            mode = self.store.journal_mode
            records = self.store.list_sessions()
        except Exception as exc:  # noqa: BLE001
            return LayerStatus("store", "degraded", f"读取失败: {exc}")
        return LayerStatus(
            "store",
            "ready",
            metrics={
                "db": str(self.store.db_path),
                "journal_mode": mode,
                "sessions": len(records),
            },
        )

    def _context_layer(self) -> LayerStatus:
        if self.assembler is None:
            return LayerStatus("context", "failed", self._problem_of("context"))
        return LayerStatus(
            "context",
            "ready",
            metrics={
                "capacity": self.assembler.capacity,
                "compressed_at": sorted(self.assembler.watermark.compressed_at)
                if self.assembler.watermark
                else [],
                "auto_compress": self.assembler.summarizer is not None,
            },
        )

    def _model_layer(self) -> LayerStatus:
        if not self.model_requested:
            # 主动跳过模型加载：不算降级，避免 --no-model 时整体状态变 WARN
            return LayerStatus("model", "not_loaded", "按参数跳过模型通道")
        if self.proxy is None or self.config is None:
            return LayerStatus("model", "degraded", self._problem_of("model"))
        return LayerStatus(
            "model",
            "ready",
            metrics={
                "base_url": self.config.base_url,
                "model": self.config.model,
                "api_key": self.config.masked_key(),
                "max_retries": self.config.max_retries,
                "rate_window_s": self.proxy.rate_window.window_s,
            },
        )

    def _problem_of(self, prefix: str) -> str:
        for item in self.problems:
            if item.startswith(prefix):
                return item
        return "未初始化"

    # ------------------------------------------------------------ 运维

    def new_tree_session_state(self) -> TreeSessionState:
        """为一个新会话创建树查询状态（``delivered`` + 已发现清单）。

        B14：两个缓存必须在同一把锁内读写，因此不单独暴露。
        """
        return TreeSessionState(DiscoveredTools(), TreeQueryState())

    def new_trace_logger(self, session_id: str) -> TraceLogger:
        """为会话创建轨迹写入器（一会话一 JSONL，E3）。"""
        return TraceLogger(self.workdir / ".agent" / "traces", session_id)

    def sweep(self) -> list[str]:
        """驱逐可卸载的空闲会话，返回被卸载的会话 id。"""
        if self.registry is None:
            return []
        return self.registry.sweep()

    def close(self) -> None:
        """释放持久化连接（幂等）。"""
        if self.store is not None:
            try:
                self.store.close()
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"close: {exc}")


def _worst_state(layers: list[LayerStatus]) -> str:
    """按严重程度归并各层状态（``not_loaded`` 视为可用）。"""
    states = {layer.state for layer in layers}
    if "failed" in states:
        return "failed"
    if "degraded" in states:
        return "degraded"
    return "ready"


def build_service(
    workdir: str | Path = ".",
    *,
    capacity: int | None = None,
    load_model: bool = True,
) -> HarnessService:
    """便捷构造函数。

    Args:
        workdir: 工作目录。
        capacity: 上下文容量；省略时取默认值。
        load_model: 是否加载模型通道。
    """
    return HarnessService(
        workdir,
        capacity=capacity or DEFAULT_CONTEXT.capacity,
        load_model=load_model,
    )


def default_execution_limits() -> dict[str, int]:
    """执行层默认限额（供 CLI 展示）。"""
    return {
        "max_workers": DEFAULT_EXECUTION.max_workers,
        "chat_capacity": DEFAULT_EXECUTION.chat_capacity,
    }
