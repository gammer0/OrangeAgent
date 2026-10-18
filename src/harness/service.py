"""Service management: assemble scattered runtime components into one queryable,
startable/stoppable service instance.

Responsibilities (this is a **composition root**, not a new business concept):

* Own the lifecycle of ``FlatStore`` / ``ToolTree`` / ``SessionRegistry`` /
  ``SessionStore`` / ``ReverseProxy`` / ``ContextAssembler`` and friends;
* Provide a read-only status snapshot (``ServiceStatus``) for the CLI and health
  checks;
* Provide constrained operational actions (evict idle sessions, close the
  persistence connection).

Design constraints:

* **Introduce no new invariants**: every constraint is still guarded by its own
  module; this module only assembles and observes.
* **The status snapshot must be side-effect free**: ``status()`` triggers no LLM
  call, writes no database row and clears no cache.
* **Failure is data**: component initialization failures are not raised to the
  caller but recorded in the snapshot's ``problems``, so the CLI can report
  "which layer did not come up" instead of crashing the whole process.
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
    """Status of a single component.

    Attributes:
        name: Component name (maps to a module).
        state: ``ready`` / ``degraded`` / ``failed``.
        detail: Human-readable description; the error reason on failure (never
            containing secrets).
        metrics: Key counters for this component.
    """

    name: str
    state: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Whole-service snapshot (read-only, side-effect free).

    Attributes:
        workdir: Working directory.
        state: Overall state, the worst result across layers.
        layers: Per-component status list.
        problems: Problems collected during initialization.
        captured_at: Snapshot timestamp.
    """

    workdir: str
    state: str
    layers: list[LayerStatus]
    problems: list[str] = field(default_factory=list)
    captured_at: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether every component is ready."""
        return self.state == "ready"

    def layer(self, name: str) -> LayerStatus | None:
        """Look up one layer's status by name."""
        for item in self.layers:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Export as a JSON-serializable dict (for CLI ``--json``)."""
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
    """Service composition root.

    Construction **never raises**: component initialization failures are recorded
    in ``self.problems`` and the corresponding layer becomes ``failed`` or
    ``degraded``, so the CLI can report layer by layer.

    Attributes:
        workdir: Working directory (``.env`` and ``.agent/`` are relative to it).
        config: LLM configuration; ``None`` when loading failed.
        flat_store / tree: Tool capability layer.
        registry: Session registry.
        store: SQLite persistence.
        proxy: Model reverse proxy.
        assembler: Context assembler.
    """

    def __init__(
        self,
        workdir: str | Path = ".",
        *,
        capacity: int = DEFAULT_CONTEXT.capacity,
        load_model: bool = True,
        global_dir: str | Path | None = None,
    ) -> None:
        """Assemble the service.

        Args:
            workdir: Working directory; both ``.env`` and ``.agent/`` are resolved
                relative to it.
            capacity: Context capacity (tokens).
            load_model: Whether to load the model channel. When ``False`` only
                local state is checked: no API key validation and no HTTP client.
            global_dir: Global directory (holding long-term memory); defaults to
                the working directory.
        """
        self.workdir = Path(workdir).resolve()
        self.global_dir = Path(global_dir).resolve() if global_dir else self.workdir
        self.capacity = capacity
        self.problems: list[str] = []
        self.started_at = time.time()

        self.config: LLMConfig | None = None
        self.flat_store: FlatStore | None = None
        self.tree: ToolTree | None = None
        #: Capability manifest loaded from ``.agent/capabilities.json``; None if not loaded.
        self.capabilities: CapabilityManifest | None = None
        self.permissions: AgentPermissions = AgentPermissions(agent_id=LOCAL_AGENT)
        self.registry: SessionRegistry | None = None
        self.store: SessionStore | None = None
        self.proxy: ReverseProxy | None = None
        self.assembler: ContextAssembler | None = None
        #: Path of the working-memory file and whether this run created it (diagnostics).
        self.work_memory: Path | None = None
        self.work_memory_created = False

        #: Whether the model channel was requested; when ``False`` the layer is
        #: ``not_loaded`` rather than degraded.
        self.model_requested = load_model

        self._ensure_work_memory()
        self._load_capability()
        self._load_session()
        self._load_context()
        if load_model:
            self._load_model()

    # --------------------------------------------------------- assembly

    def _ensure_work_memory(self) -> None:
        """Ensure ``<workdir>/.agent/WORK.md`` exists (created from the template if missing).

        The system prompt tells the model "this project has ``.agent/WORK.md``",
        but nothing ever created it -- the model's first read returned "file not
        found", contradicting the prompt.

        **Failure never blocks startup**: working memory is an aid, and the agent
        still works without it, so failures are only recorded in ``problems``.
        """
        try:
            path, created = ensure_working_memory(self.workdir)
            self.work_memory = path
            self.work_memory_created = created
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
            self.problems.append(f"work_memory: {exc}")

    def _load_capability(self) -> None:
        """Assemble the tool capability layer (flat store + derived tree).

        Note: ``.agent/capabilities.json`` must be read here, otherwise the CLI
        sees an empty tree while the web service loads the real tools from the
        manifest -- the two sides would observe different capability surfaces.
        A missing or invalid manifest falls back to an empty tree (conservative
        degradation, consistent with I4).

        Permissions are read from ``.agent/permissions.json`` (agent_id=``local``)
        so the CLI and the runtime see the same blocking results; a missing file
        yields an empty table (everything allowed).
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
        except Exception as exc:  # noqa: BLE001 - the composition root must degrade, not crash
            self.problems.append(f"capability: {exc}")
            self.flat_store = None
            self.tree = None

    def _load_session(self) -> None:
        """Assemble the session registry and SQLite persistence."""
        try:
            self.registry = SessionRegistry(
                EvictionPolicy(
                    max_resident=DEFAULT_SESSIONS.max_resident,
                    idle_timeout_s=DEFAULT_SESSIONS.idle_timeout_s,
                    sweep_interval_s=DEFAULT_SESSIONS.sweep_interval_s,
                )
            )
            # S2: start the background periodic sweep, otherwise idle sessions
            # are never reclaimed
            self.registry.start_sweeper()
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"session-registry: {exc}")
            self.registry = None

        try:
            self.store = SessionStore(self.workdir)
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"session-store: {exc}")
            self.store = None

    def _load_context(self) -> None:
        """Assemble the context assembler (no summarizer injected; compression degrades losslessly per I4)."""
        try:
            self.assembler = ContextAssembler(capacity=self.capacity)
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"context: {exc}")
            self.assembler = None

    def _load_model(self) -> None:
        """Assemble the model channel (secret read from the environment)."""
        try:
            self.config = load_llm_config(dotenv_path=self.workdir / ".env")
            self.proxy = build_proxy(self.config)
        except ConfigError as exc:
            # Missing configuration is an expected state: degrade, do not fail
            self.problems.append(f"model: {exc}")
            self.config = None
            self.proxy = None
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"model: {exc}")
            self.config = None
            self.proxy = None

    # ---------------------------------------------------------- observation

    def status(self) -> ServiceStatus:
        """Return a read-only status snapshot (side-effect free)."""
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
            return LayerStatus("store", "degraded", f"read failed: {exc}")
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
            # Model loading was deliberately skipped: not a degradation, so
            # ``--no-model`` does not turn the overall state into WARN.
            return LayerStatus("model", "not_loaded", "model channel skipped by request")
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
        return "not initialized"

    # -------------------------------------------------------- operations

    def new_tree_session_state(self) -> TreeSessionState:
        """Create tree-query state for a new session (``delivered`` + discovered inventory).

        B14: the two caches must be read and written under the same lock, so they
        are never exposed separately.
        """
        return TreeSessionState(DiscoveredTools(), TreeQueryState())

    def new_trace_logger(self, session_id: str) -> TraceLogger:
        """Create a trace writer for the session (one JSONL per session, E3)."""
        return TraceLogger(self.workdir / ".agent" / "traces", session_id)

    def sweep(self) -> list[str]:
        """Evict unloadable idle sessions; returns the ids that were unloaded."""
        if self.registry is None:
            return []
        return self.registry.sweep()

    def close(self) -> None:
        """Release the persistence connection (idempotent)."""
        if self.registry is not None:
            try:
                self.registry.stop_sweeper()
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"close-sweeper: {exc}")
        if self.store is not None:
            try:
                self.store.close()
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"close: {exc}")


def _worst_state(layers: list[LayerStatus]) -> str:
    """Merge layer states by severity (``not_loaded`` counts as usable)."""
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
    """Convenience constructor.

    Args:
        workdir: Working directory.
        capacity: Context capacity; defaults when omitted.
        load_model: Whether to load the model channel.
    """
    return HarnessService(
        workdir,
        capacity=capacity or DEFAULT_CONTEXT.capacity,
        load_model=load_model,
    )


def default_execution_limits() -> dict[str, int]:
    """Execution-layer default limits (for CLI display)."""
    return {
        "max_workers": DEFAULT_EXECUTION.max_workers,
        "chat_capacity": DEFAULT_EXECUTION.chat_capacity,
    }
