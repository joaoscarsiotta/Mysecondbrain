"""Meu Segundo Cérebro - Interface Web com Streamlit."""

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import chromadb

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

import config
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.conversation_store import ConversationStore
from core.qa_engine import QAEngine
from core.document_loader import extract_text, SUPPORTED_EXTENSIONS
from core.chunker import chunk_text

# ── Configuração da página ──────────────────────────────────────────
st.set_page_config(
    page_title="Meu Segundo Cérebro",
    page_icon="🧠",
    layout="wide",
)


# ── Inicialização dos serviços (cached) ─────────────────────────────
@st.cache_resource
def init_services(openai_key: str):
    """Inicializa todos os serviços uma única vez."""
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    os.makedirs(config.CONVERSATIONS_DIR, exist_ok=True)
    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)

    embedding_service = EmbeddingService(api_key=openai_key, model=config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    vector_store = VectorStore(
        chroma_client=chroma_client,
        collection_name=config.DOCUMENTS_COLLECTION,
        embedding_service=embedding_service,
    )

    conversation_store = ConversationStore(
        conversations_dir=config.CONVERSATIONS_DIR,
        chroma_client=chroma_client,
        embedding_service=embedding_service,
    )

    qa_engine = QAEngine(
        vector_store=vector_store,
        conversation_store=conversation_store,
    )

    return vector_store, conversation_store, qa_engine


# ── Estado da sessão ────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Segundo Cérebro")
    st.markdown("---")

    # Configuração de API
    with st.expander("⚙️ Configurações", expanded=not config.OPENAI_API_KEY):
        openai_key = st.text_input(
            "OpenAI API Key",
            value=config.OPENAI_API_KEY,
            type="password",
            help="Necessária para embeddings e LLM (se usar OpenAI)",
        )

        provider = st.selectbox(
            "Provedor LLM",
            ["openai", "anthropic"],
            index=0 if config.LLM_PROVIDER == "openai" else 1,
        )

        if provider == "anthropic":
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=config.ANTHROPIC_API_KEY,
                type="password",
            )
            config.ANTHROPIC_API_KEY = anthropic_key

        config.LLM_PROVIDER = provider

    if not openai_key:
        st.warning("Configure sua OpenAI API Key para começar.")
        st.stop()

    # Inicializar serviços
    vector_store, conversation_store, qa_engine = init_services(openai_key)

    st.markdown("---")

    # Upload de documentos
    st.subheader("📄 Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Envie seus arquivos",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "epub"],
    )

    if uploaded_files and st.button("📥 Indexar Documentos", use_container_width=True):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processando {uploaded_file.name}..."):
                # Salvar arquivo
                file_path = os.path.join(config.UPLOADS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extrair texto
                result = extract_text(file_path)
                if not result["text"].strip():
                    st.warning(f"⚠️ {uploaded_file.name}: nenhum texto extraído.")
                    continue

                # Chunkar e indexar
                chunks = chunk_text(
                    result["text"],
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                )
                vector_store.add_documents(chunks, result["metadata"])
                st.success(f"✅ {uploaded_file.name} — {len(chunks)} chunks indexados")

    # Lista de documentos indexados
    st.markdown("---")
    st.subheader("📚 Documentos Indexados")
    docs = vector_store.list_documents()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.text(doc)
            if col2.button("🗑️", key=f"del_{doc}"):
                vector_store.delete_document(doc)
                st.rerun()
    else:
        st.caption("Nenhum documento indexado ainda.")

    # Conversas anteriores
    st.markdown("---")
    st.subheader("💬 Conversas")
    if st.button("➕ Nova Conversa", use_container_width=True):
        st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.messages = []
        st.rerun()

    conversations = conversation_store.list_conversations()
    for conv in conversations[:10]:  # Mostrar últimas 10
        label = f"{conv['created_at'][:10]} — {conv['preview'][:40]}..."
        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
            # Carregar conversa
            conv_data = conversation_store.get_conversation(conv["id"])
            if conv_data:
                st.session_state.conversation_id = conv["id"]
                st.session_state.messages = []
                for turn in conv_data["turns"]:
                    st.session_state.messages.append(
                        {"role": "user", "content": turn["question"]}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": turn["answer"]}
                    )
                st.rerun()


# ── Área principal: Chat ────────────────────────────────────────────
st.header("🧠 Meu Segundo Cérebro")
st.caption(f"Conversa: {st.session_state.conversation_id}")

# Mostrar histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Fontes"):
                for src in msg["sources"]:
                    st.text(f"• {src}")

# Input do usuário
if question := st.chat_input("Faça uma pergunta ao seu Segundo Cérebro..."):
    # Mostrar pergunta
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Gerar resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = qa_engine.answer(question, st.session_state.conversation_id)

        st.markdown(result["answer"])

        # Mostrar fontes
        source_files = list(set(
            s["metadata"].get("filename", "") for s in result["sources"] if s.get("metadata")
        ))
        if source_files:
            with st.expander("📎 Fontes utilizadas"):
                for src in source_files:
                    st.text(f"• {src}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": source_files if source_files else [],
        }
    )
