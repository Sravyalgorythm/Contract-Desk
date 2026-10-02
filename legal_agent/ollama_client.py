from urllib.parse import urlparse

import requests


class OllamaError(RuntimeError):
    """Raised when the local Ollama service or a configured model is unavailable."""


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str) -> None:
        host = urlparse(base_url).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("For privacy, Ollama must use a localhost URL.")
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}", json=payload, timeout=(5, 180)
            )
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as exc:
            raise OllamaError(
                "Cannot connect to Ollama at 127.0.0.1:11434. Start the Ollama app "
                "and try again."
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError("Ollama took too long to respond. Try a shorter document.") from exc
        except requests.HTTPError as exc:
            detail = response.text[:500]
            raise OllamaError(f"Ollama returned an error: {detail}") from exc
        except (ValueError, requests.RequestException) as exc:
            raise OllamaError(f"Could not read Ollama's response: {exc}") from exc

    def available_models(self) -> set[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(5, 15))
            response.raise_for_status()
            models = response.json().get("models", [])
        except requests.ConnectionError as exc:
            raise OllamaError(
                "Cannot connect to Ollama at 127.0.0.1:11434. Start the Ollama app "
                "and try again."
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Could not list local Ollama models: {exc}") from exc
        return {model.get("name", "") for model in models}

    def ensure_models_available(self) -> None:
        available = self.available_models()

        def canonical(name: str) -> str:
            return name if ":" in name else f"{name}:latest"

        missing = [
            model
            for model in (self.chat_model, self.embedding_model)
            if canonical(model) not in available
        ]
        if missing:
            names = ", ".join(missing)
            raise OllamaError(
                f"Missing local Ollama model(s): {names}. Download them before using "
                "the app, then restart it."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            "/api/embed", {"model": self.embedding_model, "input": texts}
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an unexpected embedding response.")
        return embeddings

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        content = response.get("message", {}).get("content", "").strip()
        if not content:
            raise OllamaError("Ollama returned an empty answer. Please try again.")
        return content
