# ---- build a tiny image for Flask API ----
FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# copy only requirements first for better caching
COPY server/requirements.txt ./server-requirements.txt
RUN pip install --no-cache-dir -r server-requirements.txt && pip install gunicorn

# copy app code
COPY server/ ./server

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GUNICORN_CMD_ARGS="--bind 0.0.0.0:8080 --workers 2 --threads 4 --timeout 120"

# expose for App Runner
EXPOSE 8080

# start the server as a package so relative imports work
CMD ["gunicorn", "-k", "gthread", "server.app:app"]
