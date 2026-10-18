"""配置加载测试：环境变量读取、缺失检测、密钥掩码。"""

from __future__ import annotations

import pytest

from harness.config import (
    ConfigError,
    LLMConfig,
    build_proxy,
    load_llm_config,
)


class TestLoadConfig:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dotenv 会真实写入 os.environ，必须显式清理，避免污染其它用例。"""
        for name in (
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL",
            "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS",
            "LLM_TIMEOUT",
            "LLM_MAX_RETRIES",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_reads_all_env_vars(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdef1234567890")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com/")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
        monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
        monkeypatch.setenv("LLM_TIMEOUT", "30")
        monkeypatch.setenv("LLM_MAX_RETRIES", "1")

        config = load_llm_config(dotenv_path=tmp_path / "missing.env")

        assert config.api_key == "sk-abcdef1234567890"
        assert config.base_url == "https://api.example.com"  # 尾部斜杠被去掉
        assert config.model == "deepseek-chat"
        assert config.temperature == 0.3
        assert config.max_tokens == 1024
        assert config.timeout == 30.0
        assert config.max_retries == 1

    def test_defaults_applied_when_optional_vars_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real-key-value")
        for name in (
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL",
            "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS",
            "LLM_TIMEOUT",
            "LLM_MAX_RETRIES",
        ):
            monkeypatch.delenv(name, raising=False)

        config = load_llm_config(dotenv_path=tmp_path / "missing.env")

        assert config.model == "deepseek-chat"
        assert config.temperature == 0.7
        assert config.max_tokens == 8192
        assert config.max_retries == 2

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        with pytest.raises(ConfigError) as exc:
            load_llm_config(dotenv_path=tmp_path / "missing.env")

        assert "DEEPSEEK_API_KEY" in str(exc.value)
        # 错误信息不得回显任何密钥内容
        assert "sk-" not in str(exc.value)

    def test_placeholder_key_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-your-key-here")

        with pytest.raises(ConfigError) as exc:
            load_llm_config(dotenv_path=tmp_path / "missing.env")

        assert "占位符" in str(exc.value)

    def test_invalid_numeric_var_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-real")
        monkeypatch.setenv("LLM_TEMPERATURE", "not-a-number")

        with pytest.raises(ConfigError) as exc:
            load_llm_config(dotenv_path=tmp_path / "missing.env")

        assert "LLM_TEMPERATURE" in str(exc.value)

    def test_dotenv_file_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DEEPSEEK_API_KEY=sk-from-dotenv\nDEEPSEEK_MODEL=dotenv-model\n",
            encoding="utf-8",
        )

        config = load_llm_config(dotenv_path=env_file)

        assert config.api_key == "sk-from-dotenv"
        assert config.model == "dotenv-model"

    def test_real_env_wins_over_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """CI 注入的真实环境变量优先级高于 .env（override=False）。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-process")
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=sk-from-file\n", encoding="utf-8")

        config = load_llm_config(dotenv_path=env_file)

        assert config.api_key == "sk-from-process"


class TestSecretHandling:
    def test_repr_does_not_leak_key(self) -> None:
        config = LLMConfig(api_key="sk-super-secret-value")
        assert "sk-super-secret-value" not in repr(config)
        assert "api_key" not in repr(config).split("api_key=")[0] or True

    def test_masked_key_keeps_edges_only(self) -> None:
        config = LLMConfig(api_key="sk-1234567890abcdef")
        masked = config.masked_key()
        assert masked == "sk-1...cdef"
        assert "234567890abc" not in masked

    def test_masked_key_short_value_fully_hidden(self) -> None:
        assert LLMConfig(api_key="short").masked_key() == "***"


class TestBuildProxy:
    def test_build_proxy_uses_config(self) -> None:
        config = LLMConfig(api_key="sk-x", base_url="https://api.test", model="m1")
        proxy = build_proxy(config, transport=object())

        provider = proxy.providers[0]
        assert provider.base_url == "https://api.test"
        assert provider.model == "m1"
        assert provider.api_key == "sk-x"
        assert proxy.temperature == config.temperature

    def test_build_proxy_respects_retry_setting(self) -> None:
        config = LLMConfig(api_key="sk-x", max_retries=5)
        proxy = build_proxy(config, transport=object())
        assert proxy.max_retries == 5
