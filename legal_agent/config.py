from dataclasses import dataclass
from dataclasses import field
import os

from dotenv import load_dotenv

load_dotenv()


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _min_relevance() -> float:
    try:
        value = float(_env_value("LEGAL_AGENT_MIN_RELEVANCE", "0.28"))
    except ValueError:
        return 0.28
    return value if 0.0 <= value <= 0.9 else 0.28


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = field(
        default_factory=lambda: _env_value(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        or "http://127.0.0.1:11434"
    )
    chat_model: str = field(
        default_factory=lambda: _env_value("OLLAMA_CHAT_MODEL", "qwen3:1.7b")
    )
    embedding_model: str = field(
        default_factory=lambda: _env_value("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    min_relevance: float = field(default_factory=_min_relevance)
    chunk_words: int = 180
    overlap_words: int = 30
    max_upload_bytes: int = 30 * 1024 * 1024


settings = Settings()
