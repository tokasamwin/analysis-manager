# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
WORKDIR /app
# No proxy / no telemetry inside the image build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
# Install dependencies from an isolated build layer first
COPY pyproject.toml ./
COPY analysisManager ./analysisManager
RUN pip install --prefix=/install ".[server]"

FROM python:3.11-slim
ENV ANALYSISMANAGER_DATADIR=/data \
    ANALYSIS_STAGE_REGISTRY=/etc/analysis/registry.json \
    ANALYSIS_WORKERS=4 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /install /usr/local
COPY analysisManager ./analysisManager
# The registry is an optional mount; /api/config works either way.
RUN rm -rf /root/.cache
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
	CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"
# One gunicorn worker keeps the FIFO run queue and in-process state
# (inline-code functions, registries) consistent; threads serve the API.
ENTRYPOINT ["gunicorn", "analysisManager.server:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--access-logfile", "-"]
