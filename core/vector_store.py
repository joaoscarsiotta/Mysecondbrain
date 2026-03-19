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
        tags_str = ",".join(sorted(set(
            t.lower() for t in metadata.get("tags", []) if t.strip()
        )))

        ids = [f"{filename}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "filename": filename,
                "file_type": metadata.get("file_type", ""),
                "chunk_index": c["chunk_index"],
                "file_hash": metadata.get("file_hash", ""),
                "tags": tags_str,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def update_document_tags(self, filename: str, tags: list[str]) -> None:
        """Atualiza as tags de todos os chunks de um documento."""
        results = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
            limit=10000,
        )
        if not results["ids"]:
            return
        tags_str = ",".join(sorted(set(t.lower() for t in tags if t.strip())))
        updated_metadatas = [{**m, "tags": tags_str} for m in results["metadatas"]]
        self.collection.update(ids=results["ids"], metadatas=updated_metadatas)

    def get_document_hash(self, filename: str) -> str | None:
        """Retorna o hash SHA-256 armazenado do documento, ou None se não indexado."""
        results = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
            limit=1,
        )
        if results["metadatas"]:
            return results["metadatas"][0].get("file_hash") or None
        return None

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        tags_filter: list[str] | None = None,
    ) -> list[dict]:
        """Busca semântica nos documentos indexados.

        Cada documento é buscado individualmente para garantir que todos
        sejam considerados. Os melhores chunks de cada documento são
        reunidos e ordenados globalmente por relevância.

        Se tags_filter for fornecido, considera apenas documentos que
        possuam pelo menos uma das tags especificadas.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query_text)
        all_docs = self.list_documents()
        if not all_docs:
            return []

        if tags_filter:
            all_docs = [
                d for d in all_docs
                if any(t in d["tags"] for t in tags_filter)
            ]
            if not all_docs:
                return []

        # Chunks a buscar por documento: distribui top_k igualmente, mínimo 1
        chunks_per_doc = max(1, (top_k + len(all_docs) - 1) // len(all_docs))

        candidates = []
        for doc in all_docs:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=chunks_per_doc,
                    where={"filename": doc["filename"]},
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

    def list_documents(self) -> list[dict]:
        """Lista documentos únicos indexados com suas tags.

        Retorna lista de dicts: [{"filename": str, "tags": list[str]}, ...]
        """
        total = self.collection.count()
        if total == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"], limit=total)
        docs: dict[str, dict] = {}
        for m in all_meta["metadatas"]:
            fname = m["filename"]
            if fname not in docs:
                tags_str = m.get("tags", "")
                tags = [t for t in tags_str.split(",") if t] if tags_str else []
                docs[fname] = {"filename": fname, "tags": tags}
        return sorted(docs.values(), key=lambda d: d["filename"])
