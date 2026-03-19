"""Wrapper do ChromaDB para armazenamento e busca vetorial."""

import chromadb
from core.embeddings import EmbeddingService


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_service: EmbeddingService,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_service = embedding_service

    def add_documents(self, chunks: list[dict], metadata: dict) -> None:
        """Indexa chunks de um documento no ChromaDB."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed(texts)
        filename = metadata["filename"]

        ids = [f"{filename}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "filename": filename,
                "file_type": metadata.get("file_type", ""),
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Busca semântica nos documentos indexados."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return items

    def delete_document(self, filename: str) -> None:
        """Remove todos os chunks de um documento."""
        self.collection.delete(where={"filename": filename})

    def list_documents(self) -> list[str]:
        """Lista documentos únicos indexados."""
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])
        filenames = set()
        for m in all_meta["metadatas"]:
            filenames.add(m["filename"])
        return sorted(filenames)
