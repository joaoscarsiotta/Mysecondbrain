"""Wrapper do ChromaDB para armazenamento e busca vetorial."""

import chromadb
from core.embeddings import EmbeddingService


class VectorStore:
    def __init__(
        self,
        chroma_client: chromadb.PersistentClient,
        collection_name: str,
        embedding_service: EmbeddingService,
    ):
        self.client = chroma_client
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
        """Busca semântica nos documentos indexados.

        Cada documento é buscado individualmente para garantir que todos
        sejam considerados. Os melhores chunks de cada documento são
        reunidos e ordenados globalmente por relevância.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query_text)
        all_docs = self.list_documents()
        if not all_docs:
            return []

        # Chunks a buscar por documento: distribui top_k igualmente, mínimo 1
        chunks_per_doc = max(1, (top_k + len(all_docs) - 1) // len(all_docs))

        candidates = []
        for filename in all_docs:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=chunks_per_doc,
                    where={"filename": filename},
                    include=["documents", "metadatas", "distances"],
                )
                for i in range(len(results["ids"][0])):
                    candidates.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    })
            except Exception:
                continue

        candidates.sort(key=lambda x: x["distance"])
        return candidates[:top_k]

    def delete_document(self, filename: str) -> None:
        """Remove todos os chunks de um documento."""
        self.collection.delete(where={"filename": filename})

    def list_documents(self) -> list[str]:
        """Lista documentos únicos indexados."""
        total = self.collection.count()
        if total == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"], limit=total)
        filenames = set()
        for m in all_meta["metadatas"]:
            filenames.add(m["filename"])
        return sorted(filenames)
