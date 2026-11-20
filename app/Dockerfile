# Build from the repository root:  docker build -f app/Dockerfile -t btc-forecast-api .

# ---- Stage 1: install dependencies into an isolated prefix -------------------
FROM python:3.10-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
# Copy only the requirements first so this layer is cached until they change
COPY app/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Stage 2: small runtime image running as a non-root user -----------------
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_DIR=/app/models \
    SEED_MODEL_DIR=/app/seed_model

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/features.py ./features.py
COPY app/app.py ./app.py
# Seed model: used when /app/models (a PVC or volume) is still empty
COPY models/seed ./seed_model
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=3)"

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 --access-logfile - app:app"]
