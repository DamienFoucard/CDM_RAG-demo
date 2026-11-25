FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CDM_RAG_DATA_DIR=/data ANONYMIZED_TELEMETRY=False
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY app ./app
COPY scripts ./scripts
COPY seed ./seed
COPY evaluation ./evaluation
COPY web ./web
COPY docs/images/cdm-rag-readme-tour.svg ./docs/images/cdm-rag-readme-tour.svg
RUN useradd --uid 10001 --create-home appuser && mkdir -p /data && chown -R appuser:appuser /app /data
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
