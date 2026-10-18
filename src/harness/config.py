"""Load LLM configuration from environment variables and build the reverse proxy.

See docs/01-architecture.md §9 (configuration items).

Conventions:

* Secrets are **read from environment variables only**; they are never written
  into code, logs or ``repr``.
* ``python-dotenv`` reads the ``.env`` in the working directory at load time;
  real environment variables already set take precedence (``override=False``),
  which suits CI injection.
* Only OpenAI-compatible endpoints are provided (DeepSeek / Moonshot / vLLM /
  OpenAI share the same protocol); pass ``adapter`` explicitly for the
  Anthropic protocol.

Usage::

    from harness.config import load_llm_config, build_proxy_from_env

    cfg = load_llm_config()          # raises ConfigError when the key is missing
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

#: Environment variable names (kept in sync with ``.env.example``).
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_MODEL = "DEEPSEEK_MODEL"
ENV_TEMPERATURE = "LLM_TEMPERATURE"
ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_TIMEOUT = "LLM_TIMEOUT"
ENV_MAX_RETRIES = "LLM_MAX_RETRIES"

#: Fallback defaults (except the key -- a missing key is an error, never a
#: placeholder that lets the system run silently).
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
#: Per-call output limit.
#: Note: 4096 is not enough -- we measured the agent writing a 100-line file
#: hitting the 4096 cap mid-argument (truncated at ``_rjust(``), leaving an
#: incomplete write. DeepSeek supports 8192; better too large than too small,
#: since cost tracks actual generation only.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


class ConfigError(RuntimeError):
    """Configuration is missing or invalid.

    Error messages state **which variable is missing** and never echo its value.
    """


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Everything needed to attach one model endpoint.

    Attributes:
        api_key: Secret; masked in ``repr`` and never logged.
        base_url: OpenAI-compatible endpoint base URL.
        model: Model name.
        temperature / max_tokens / timeout / max_retries: Generation and retry
            parameters.
        endpoint: Concrete path, default ``/chat/completions``.
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
        """Return a log-safe masked key (first and last 4 characters only)."""
        key = self.api_key
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"


def load_dotenv_if_present(dotenv_path: str | Path | None = None) -> None:
    """Load ``.env`` into the environment if present (never overriding existing values).

    ``python-dotenv`` is a declared project dependency; when it is missing we
    skip silently and let the caller read real environment variables -- exactly
    the CI scenario.
    """
    path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency declared; defensive only
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
    """Build an :class:`LLMConfig` from environment variables.

    Args:
        dotenv_path: Optional ``.env`` path; defaults to ``.env`` in the current
            working directory.

    Returns:
        The configuration object; ``api_key`` is masked in ``repr``.

    Raises:
        ConfigError: ``DEEPSEEK_API_KEY`` is missing, or a numeric variable is
            invalid.
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
    """Build a :class:`ReverseProxy` from the configuration.

    Args:
        config: Loaded configuration.
        adapter: Adapter; defaults to :class:`OpenAIAdapter`.
        max_requests / max_tokens_per_window / window_s: Global rate-limit
            parameters.
        transport: Optional transport implementation; defaults to real HTTP
            (``OpenAITransport``).

    Returns:
        A ready-to-use reverse proxy instance.
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
    """Convenience entry point: read the environment and build the proxy."""
    return build_proxy(load_llm_config(), **kwargs)  # type: ignore[arg-type]
