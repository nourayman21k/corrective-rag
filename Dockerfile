FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    INDEX_DIR=/app/faiss_index \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Download the embedding model and build the FAISS index at build time
COPY *.pdf ./data/
RUN python -c "from corrective_rag.pipeline import build_vector_store; build_vector_store()"

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/corrective_rag/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
