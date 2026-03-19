"""Motor de Q&A com RAG - busca contexto nos documentos e conversas."""

from openai import OpenAI
import anthropic

from core.vector_store import VectorStore
from core.conversation_store import ConversationStore
import config


class QAEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        conversation_store: ConversationStore,
    ):
        self.vector_store = vector_store
        self.conversation_store = conversation_store

    def answer(
        self,
        question: str,
        conversation_id: str,
        tags_filter: list[str] | None = None,
    ) -> dict:
        """Responde uma pergunta usando RAG sobre documentos e conversas."""
        # 1. Buscar contexto nos documentos
        doc_results = self.vector_store.query(
            question, top_k=config.TOP_K_DOCUMENTS, tags_filter=tags_filter
        )

        # 2. Buscar conversas passadas relevantes
        conv_results = self.conversation_store.query_relevant(
            question, top_k=config.TOP_K_CONVERSATIONS
        )

        # 3. Construir prompt
        prompt = self._build_prompt(question, doc_results, conv_results)

        # 4. Chamar LLM
        answer_text = self._call_llm(prompt)

        # 5. Salvar conversa
        self.conversation_store.save_turn(
            conversation_id, question, answer_text, doc_results
        )

        return {
            "answer": answer_text,
            "sources": doc_results,
            "conversation_sources": conv_results,
        }

    def _build_prompt(
        self,
        question: str,
        doc_results: list[dict],
        conv_results: list[dict],
    ) -> list[dict]:
        """Monta as mensagens para o LLM."""
        system = (
            "Você é um assistente de conhecimento pessoal (Segundo Cérebro). "
            "Responda perguntas usando APENAS o contexto fornecido abaixo. "
            "Se o contexto não contiver informação suficiente, diga isso claramente. "
            "Cite quais documentos você usou na resposta. "
            "Responda no mesmo idioma da pergunta."
        )

        context_parts = []

        if doc_results:
            context_parts.append("=== CONTEXTO DOS DOCUMENTOS ===")
            for i, r in enumerate(doc_results, 1):
                filename = r["metadata"].get("filename", "desconhecido")
                context_parts.append(f"\n[Documento {i}: {filename}]\n{r['text']}")

        if conv_results:
            context_parts.append("\n=== CONVERSAS ANTERIORES RELEVANTES ===")
            for i, r in enumerate(conv_results, 1):
                context_parts.append(f"\n[Conversa {i}]\n{r['text']}")

        if not context_parts:
            context_parts.append(
                "(Nenhum documento ou conversa encontrada na base de conhecimento.)"
            )

        context = "\n".join(context_parts)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n---\nPergunta: {question}"},
        ]

    def _call_llm(self, messages: list[dict]) -> str:
        """Chama o LLM configurado (OpenAI ou Anthropic)."""
        if config.LLM_PROVIDER == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    def _call_openai(self, messages: list[dict]) -> str:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, messages: list[dict]) -> str:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        # Separar system message
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        response = client.messages.create(
            model=config.LLM_MODEL,
            system=system_msg,
            messages=user_messages,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.content[0].text
