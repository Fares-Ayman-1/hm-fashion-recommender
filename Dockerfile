# API + demo image (no training stack). The serving bundle and thumbnails are mounted at runtime.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app \
    && mkdir /app && chown app /app

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Build as the unprivileged user so no slow recursive chown is needed afterwards.
USER app
WORKDIR /app
COPY --chown=app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra dashboard --no-install-project
COPY --chown=app src ./src
COPY --chown=app app ./app
RUN uv sync --frozen --no-dev --extra dashboard

ENV PATH="/app/.venv/bin:$PATH" \
    SERVING_DIR=/app/reports/serving \
    THUMBS_DIR=/app/data/thumbs \
    PYTHONUNBUFFERED=1

EXPOSE 8010 8511
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fsS http://localhost:8010/health || exit 1
CMD ["uvicorn", "hmrec.serve.api:app", "--host", "0.0.0.0", "--port", "8010"]
