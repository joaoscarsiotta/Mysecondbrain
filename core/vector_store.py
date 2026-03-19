"""Wrapper do ChromaDB para armazenamento e busca vetorial."""

from collections import defaultdict

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
        """Busca semântica nos documentos indexados com diversidade de fontes.

        Busca um pool maior de candidatos e distribui os resultados entre
        todos os documentos indexados via round-robin, garantindo que nenhum
        documento seja ignorado.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_single(query_text)

        # Buscar pool maior para garantir cobertura de todos os documentos
        fetch_k = min(top_k * 4, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        # Agrupar resultados por documento (mantendo ordem de relevância)
        by_doc = defaultdict(list)
        for i in range(len(results["ids"][0])):
            item = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            filename = item["metadata"].get("filename", "")
            by_doc[filename].append(item)

        # Round-robin: pegar os melhores chunks de cada documento alternadamente
        items = []
        doc_iters = {k: iter(v) for k, v in by_doc.items()}
        while len(items) < top_k and doc_iters:
            exhausted = []
            for filename, it in doc_iters.items():
                if len(items) >= top_k:
                    break
                chunk = next(it, None)
                if chunk is not None:
                    items.append(chunk)
                else:
                    exhausted.append(filename)
            for filename in exhausted:
                del doc_iters[filename]

        # Ordenar resultado final por relevância (menor distância = mais similar)
        items.sort(key=lambda x: x["distance"])
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
