"""Armazenamento e indexação de conversas para contexto futuro."""

import json
import os
from datetime import datetime
from pathlib import Path

import chromadb
from core.embeddings import EmbeddingService


class ConversationStore:
    def __init__(
        self,
        conversations_dir: str,
        chroma_client: chromadb.PersistentClient,
        embedding_service: EmbeddingService,
    ):
        self.conversations_dir = conversations_dir
        os.makedirs(conversations_dir, exist_ok=True)
        self.collection = chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_service = embedding_service

    def save_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> None:
        """Salva um turno de conversa em JSON e indexa no ChromaDB."""
        # Salvar no JSON
        conv_path = Path(self.conversations_dir) / f"{conversation_id}.json"
        if conv_path.exists():
            with open(conv_path, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
        else:
            conv_data = {
                "id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "turns": [],
            }

        turn = {
            "question": question,
            "answer": answer,
            "sources": [s.get("metadata", {}).get("filename", "") for s in sources],
            "timestamp": datetime.now().isoformat(),
        }
        conv_data["turns"].append(turn)

        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False, indent=2)

        # Indexar no ChromaDB para busca semântica
        qa_text = f"Pergunta: {question}\nResposta: {answer}"
        embedding = self.embedding_service.embed_single(qa_text)
        turn_id = f"{conversation_id}_turn_{len(conv_data['turns']) - 1}"

        self.collection.upsert(
            ids=[turn_id],
            embeddings=[embedding],
            documents=[qa_text],
            metadatas=[
                {
                    "conversation_id": conversation_id,
                    "timestamp": turn["timestamp"],
                }
            ],
        )

    def query_relevant(self, query: str, top_k: int = 3) -> list[dict]:
        """Busca conversas passadas relevantes à pergunta."""
        if self.collection.count() == 0:
            return []

        embedding = self.embedding_service.embed_single(query)
        results = self.collection.query(
            query_embeddings=[embedding],
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

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Carrega uma conversa completa do JSON."""
        conv_path = Path(self.conversations_dir) / f"{conversation_id}.json"
        if not conv_path.exists():
            return None
        with open(conv_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_conversations(self) -> list[dict]:
        """Lista todas as conversas salvas."""
        conversations = []
        for f in sorted(Path(self.conversations_dir).glob("*.json"), reverse=True):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                conversations.append(
                    {
                        "id": data["id"],
                        "created_at": data["created_at"],
                        "num_turns": len(data["turns"]),
                        "preview": data["turns"][0]["question"][:80] if data["turns"] else "",
                    }
                )
        return conversations
