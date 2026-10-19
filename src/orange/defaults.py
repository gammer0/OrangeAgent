"""Default configuration inventory (single source of truth).

This module collects the hyperparameters from the initial-configuration doc
into importable constants so that:

* docs and code can validate each other (``tests/test_defaults.py`` asserts
  both stay in sync);
* a new session can read every default in one place instead of picking
  values up from scattered modules.

**Precedence**: environment variables (``.env`` / process env) > the defaults
in this module. Secrets never live here; :mod:`orange.config` reads them
from the environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "ContextDefaults",
    "LoopDefaults",
    "ExecutionDefaults",
    "ToolDefaults",
    "SessionDefaults",
    "RateDefaults",
    "MemoryDefaults",
    "DEFAULT_CONTEXT",
    "DEFAULT_LOOP",
    "DEFAULT_EXECUTION",
    "DEFAULT_TOOLS",
    "DEFAULT_SESSIONS",
    "DEFAULT_RATE",
    "DEFAULT_MEMORY",
    "as_dict",
]


@dataclass(frozen=True, slots=True)
class ContextDefaults:
    """Context assembly and compression defaults (C1/C2/C3)."""

    capacity: int = 32_768
    auto_1: float = 0.50
    auto_2: float = 0.75
    session_end: float = 0.85
    chars_per_token: float = 2.0
    max_obs_chars: int = 4_000
    keep_records_50: int = 4
    keep_records_75: int = 2


@dataclass(frozen=True, slots=True)
class LoopDefaults:
    """Loop and termination defaults."""

    max_steps: int = 40
    max_injects: int = 10
    deadlock_threshold: int = 3


@dataclass(frozen=True, slots=True)
class ExecutionDefaults:
    """Execution and concurrency defaults."""

    max_workers: int = 4
    chat_capacity: int = 100


@dataclass(frozen=True, slots=True)
class ToolDefaults:
    """Tool-tree and routing defaults."""

    tree_enable_threshold: int = 15
    route_tool_count: int = 30
    route_branch_count: int = 6
    route_history_tokens: int = 4_000
    route_confidence: float = 0.7
    index_depth: int = 2
    l2_limit: int = 40
    max_discovered: int = 50
    description_l1: int = 40
    description_deep: int = 80


@dataclass(frozen=True, slots=True)
class SessionDefaults:
    """Session and persistence defaults."""

    max_resident: int = 20
    idle_timeout_s: int = 1_800
    sweep_interval_s: int = 60
    sqlite_timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class RateDefaults:
    """Model-layer rate limiting and circuit-breaker defaults."""

    window_s: int = 60
    max_requests: int = 60
    max_tokens: int = 200_000
    failure_threshold: int = 3
    health_backoff_s: float = 30.0
    request_backoff_s: float = 1.0


@dataclass(frozen=True, slots=True)
class MemoryDefaults:
    """Memory and trace defaults."""

    agent_dir: str = ".agent"
    work_memory: str = "WORK.md"
    long_term_memory: str = "MEMORY.md"
    extract_principle: str = "extract.md"
    trace_dir: str = "traces"
    replace_retries: int = 40
    replace_backoff_s: float = 0.002


#: Global default instances (read-only).
DEFAULT_CONTEXT = ContextDefaults()
DEFAULT_LOOP = LoopDefaults()
DEFAULT_EXECUTION = ExecutionDefaults()
DEFAULT_TOOLS = ToolDefaults()
DEFAULT_SESSIONS = SessionDefaults()
DEFAULT_RATE = RateDefaults()
DEFAULT_MEMORY = MemoryDefaults()


def as_dict() -> dict[str, dict[str, object]]:
    """Export all defaults as a nested dict (for doc generation and snapshot tests)."""
    return {
        "context": asdict(DEFAULT_CONTEXT),
        "loop": asdict(DEFAULT_LOOP),
        "execution": asdict(DEFAULT_EXECUTION),
        "tools": asdict(DEFAULT_TOOLS),
        "sessions": asdict(DEFAULT_SESSIONS),
        "rate": asdict(DEFAULT_RATE),
        "memory": asdict(DEFAULT_MEMORY),
    }
