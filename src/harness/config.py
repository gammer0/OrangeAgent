"""从环境变量加载 LLM 配置并构建反向代理。

对应文档：docs/01-系统架构设计.md §9（配置项）

约定：

* 密钥**只从环境变量读取**，不写入代码、不写入日志、不进入 ``repr``。
* 依赖 ``python-dotenv`` 在加载时读取工作目录下的 ``.env``；已存在的真实环境变量
  优先级更高（``override=False``），便于 CI 注入。
* 只提供 OpenAI 兼容端点（DeepSeek / Moonshot / vLLM / OpenAI 同一套协议）；
  需要 Anthropic 协议时显式传 ``adapter``。

用法::

    from harness.config import load_llm_config, build_proxy_from_env

    cfg = load_llm_config()          # 缺 key 时抛 ConfigError
    proxy = build_proxy_from_env()
    proxy.chat(messages, session_id="s1")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.model import (
    Adapter,
    OpenAIAdapter,
    OpenAITransport,
    ProviderConfig,
    ReverseProxy,
)
from harness.rate import RateWindow

__all__ = [
    "ConfigError",
    "LLMConfig",
    "load_llm_config",
    "build_proxy",
    "build_proxy_from_env",
    "load_dotenv_if_present",
]

#: 环境变量名（与 ``.env.example`` 保持一致）。
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_TEMPERATURE = "LLM_TEMPERATURE"
ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_TIMEOUT = "LLM_TIMEOUT"
ENV_MAX_RETRIES = "LLM_MAX_RETRIES"

#: 兜底默认值（密钥除外——缺失即报错，绝不使用占位符静默运行）。
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
#: 单次输出上限。
#: ponytail: 4096 不够用——实测 agent 写一个 100 行的文件时，参数长度
#: 正好撞上 4096 被截断（`_rjust(` 处断掉），导致写入不完整。
#: DeepSeek 系列支持 8192，宁大勿小；成本只与实际生成量相关。
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


class ConfigError(RuntimeError):
    """配置缺失或非法。

    错误信息只说明**哪个变量缺失**，绝不回显变量值。
    """


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """一次模型接入所需的全部配置。

    Attributes:
        api_key: 密钥；``repr`` 中被掩码，不会出现在日志里。
        base_url: OpenAI 兼容端点基址。
        model: 模型名。
        temperature / max_tokens / timeout / max_retries: 生成与重试参数。
        endpoint: 具体路径，默认 ``/chat/completions``。
    """

    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    endpoint: str = "/chat/completions"

    def masked_key(self) -> str:
        """返回可用于日志的掩码密钥（仅保留首尾 4 位）。"""
        key = self.api_key
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"


def load_dotenv_if_present(dotenv_path: str | Path | None = None) -> None:
    """若存在 ``.env`` 则加载到环境变量（不覆盖已存在的值）。

    ``python-dotenv`` 是项目已声明的依赖；缺失时静默跳过，由调用方从
    真实环境变量读取——这正是 CI 场景。
    """
    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 依赖已声明，仅防御
        return
    load_dotenv(dotenv_path=path, override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 不是合法数字") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 不是合法整数") from exc


def load_llm_config(*, dotenv_path: str | Path | None = None) -> LLMConfig:
    """从环境变量构造 :class:`LLMConfig`。

    Args:
        dotenv_path: 可选的 ``.env`` 路径；省略时使用当前工作目录下的 ``.env``。

    Returns:
        配置对象；``api_key`` 在 ``repr`` 中已掩码。

    Raises:
        ConfigError: 缺少 ``DEEPSEEK_API_KEY``，或数值型变量非法。
    """
    load_dotenv_if_present(dotenv_path)

    api_key = (os.getenv(ENV_API_KEY) or "").strip()
    if not api_key:
        raise ConfigError(
            f"缺少环境变量 {ENV_API_KEY}；请在 .env 或进程环境中设置后重试"
        )
    if api_key in {"sk-your-key-here", "sk-placeholder"}:
        raise ConfigError(
            f"环境变量 {ENV_API_KEY} 仍是模板占位符，请填入真实密钥"
        )

    return LLMConfig(
        api_key=api_key,
        base_url=(os.getenv(ENV_BASE_URL) or DEFAULT_BASE_URL).strip().rstrip("/"),
        model=(os.getenv(ENV_MODEL) or DEFAULT_MODEL).strip(),
        temperature=_env_float(ENV_TEMPERATURE, DEFAULT_TEMPERATURE),
        max_tokens=_env_int(ENV_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        timeout=_env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT),
        max_retries=_env_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
    )


def build_proxy(
    config: LLMConfig,
    *,
    adapter: Adapter | None = None,
    max_requests: int = 60,
    max_tokens_per_window: int = 200_000,
    window_s: int = 60,
    transport: object | None = None,
) -> ReverseProxy:
    """按配置构建 :class:`ReverseProxy`。

    Args:
        config: 已加载的配置。
        adapter: 适配器；省略时用 :class:`OpenAIAdapter`。
        max_requests / max_tokens_per_window / window_s: 全局限流参数。
        transport: 可选的传输实现；省略时用真实 HTTP（``OpenAITransport``）。

    Returns:
        可直接调用的反向代理实例。
    """
    chosen = adapter or OpenAIAdapter()
    if transport is None:
        transport = OpenAITransport(
            config.base_url, config.api_key, timeout=config.timeout
        )

    proxy = ReverseProxy(
        providers=[
            ProviderConfig(
                name="default",
                base_url=config.base_url,
                model=config.model,
                api_key=config.api_key,
                adapter=chosen,
                endpoint=config.endpoint,
            )
        ],
        rate_window=RateWindow(
            max_requests=max_requests,
            max_tokens=max_tokens_per_window,
            window_s=window_s,
        ),
        health={},
        transport=transport,  # type: ignore[arg-type]
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_retries=config.max_retries,
    )
    return proxy


def build_proxy_from_env(**kwargs: object) -> ReverseProxy:
    """便捷入口：读环境变量并直接构建代理。"""
    return build_proxy(load_llm_config(), **kwargs)  # type: ignore[arg-type]
