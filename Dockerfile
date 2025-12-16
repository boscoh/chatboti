FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY tinyrag/ ./tinyrag/
RUN uv sync --frozen --no-dev

RUN mkdir -p /app/tinyrag/data

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 80

CMD ["uv", "run", "-m", "tinyrag.cli", "server", "--host", "0.0.0.0", "--port", "80"]
