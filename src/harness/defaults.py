"""默认配置清单（单一真源）。

本模块把 `docs/04-初始配置清单.md` 中的超参数集中为可导入常量，
便于：
* 文档与代码互相校验（`tests/test_defaults.py` 断言两者一致）；
* 新会话一次性读取全部默认值，而不是散落在各模块。

**优先级**：环境变量（`.env` / 进程环境）> 本模块默认值。
密钥永远不在此处，由 :mod:`harness.config` 从环境变量读取。
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
    """上下文与压缩默认值（C1/C2/C3）。"""

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
    """循环与终止默认值。"""

    max_steps: int = 40
    max_injects: int = 10
    deadlock_threshold: int = 3


@dataclass(frozen=True, slots=True)
class ExecutionDefaults:
    """执行与并发默认值。"""

    max_workers: int = 4
    chat_capacity: int = 100


@dataclass(frozen=True, slots=True)
class ToolDefaults:
    """工具树与路由默认值。"""

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
    """会话与持久化默认值。"""

    max_resident: int = 20
    idle_timeout_s: int = 1_800
    sweep_interval_s: int = 60
    sqlite_timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class RateDefaults:
    """模型层限流与熔断默认值。"""

    window_s: int = 60
    max_requests: int = 60
    max_tokens: int = 200_000
    failure_threshold: int = 3
    health_backoff_s: float = 30.0
    request_backoff_s: float = 1.0


@dataclass(frozen=True, slots=True)
class MemoryDefaults:
    """记忆与轨迹默认值。"""

    agent_dir: str = ".agent"
    work_memory: str = "WORK.md"
    long_term_memory: str = "MEMORY.md"
    extract_principle: str = "extract.md"
    trace_dir: str = "traces"
    replace_retries: int = 40
    replace_backoff_s: float = 0.002


#: 全局默认实例（只读）。
DEFAULT_CONTEXT = ContextDefaults()
DEFAULT_LOOP = LoopDefaults()
DEFAULT_EXECUTION = ExecutionDefaults()
DEFAULT_TOOLS = ToolDefaults()
DEFAULT_SESSIONS = SessionDefaults()
DEFAULT_RATE = RateDefaults()
DEFAULT_MEMORY = MemoryDefaults()


def as_dict() -> dict[str, dict[str, object]]:
    """把全部默认值导出为嵌套字典（供文档生成与快照测试使用）。"""
    return {
        "context": asdict(DEFAULT_CONTEXT),
        "loop": asdict(DEFAULT_LOOP),
        "execution": asdict(DEFAULT_EXECUTION),
        "tools": asdict(DEFAULT_TOOLS),
        "sessions": asdict(DEFAULT_SESSIONS),
        "rate": asdict(DEFAULT_RATE),
        "memory": asdict(DEFAULT_MEMORY),
    }
