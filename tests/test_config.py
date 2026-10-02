from legal_agent.config import Settings


def test_settings_use_defaults_when_environment_values_are_missing(monkeypatch):
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_EMBED_MODEL",
        "LEGAL_AGENT_MIN_RELEVANCE",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.chat_model == "qwen3:1.7b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.min_relevance == 0.28


def test_settings_use_defaults_for_blank_or_invalid_values(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "   ")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", " ")
    monkeypatch.setenv("LEGAL_AGENT_MIN_RELEVANCE", "not-a-number")

    settings = Settings()

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.chat_model == "qwen3:1.7b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.min_relevance == 0.28