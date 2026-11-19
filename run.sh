#!/usr/bin/env bash
# Start ResearchMind. Usage: ./run.sh [port]   (default port 8010)
set -e
cd "$(dirname "$0")"
PORT="${1:-${PORT:-8010}}"

if [ ! -x .venv/bin/uvicorn ]; then
  echo "✗ Python environment missing. Create it with:"
  echo "    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi

if ! curl -s -m 3 http://localhost:11434/api/version >/dev/null; then
  echo "✗ Ollama is not running. Open the Ollama app (or run: ollama serve) and try again."
  exit 1
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "✗ Port $PORT is busy:"
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
  echo "  Use another port, e.g.  ./run.sh $((PORT + 1))"
  exit 1
fi

[ -f .env ] || cp .env.example .env
echo "✓ Starting ResearchMind → http://localhost:$PORT   (Ctrl+C to stop)"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --reload
