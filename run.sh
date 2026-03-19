#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ── Venv ──
if [ ! -d .venv ]; then
  echo "Creating venv..."
  uv venv .venv
fi
source .venv/bin/activate

# ── Deps ──
echo "Installing dependencies..."
uv pip install -r requirements.txt

# ── .env ──
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo ">>> Created .env from .env.example — edit it with your API keys before running again."
  echo ">>> At minimum set OPENAI_API_KEY (needed for embeddings)."
  exit 1
fi

# ── Run ──
echo "Starting Streamlit..."
streamlit run app.py
