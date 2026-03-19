"""Serviço de geração de embeddings via OpenAI."""

from openai import OpenAI


class EmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Gera embeddings para uma lista de textos em lotes."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Gera embedding para um único texto."""
        return self.embed([text])[0]
