import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Diretórios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
UPLOADS_DIR = str(DATA_DIR / "uploads")
CONVERSATIONS_DIR = str(DATA_DIR / "conversations")

# Chunking
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 100  # tokens

# Retrieval
TOP_K_DOCUMENTS = 5
TOP_K_CONVERSATIONS = 3

# ChromaDB
DOCUMENTS_COLLECTION = "documents"
CONVERSATIONS_COLLECTION = "conversations"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o" if LLM_PROVIDER == "openai" else "claude-sonnet-4-20250514")
