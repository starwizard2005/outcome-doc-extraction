FROM python:3.12-slim

ENV OMP_THREAD_LIMIT=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY tests ./tests
COPY tools ./tools

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# one process: API + embedded worker threads; the SQLite job store lives on the /data volume
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning", "--no-access-log"]
