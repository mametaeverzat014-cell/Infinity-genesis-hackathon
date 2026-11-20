# Official check: no network, no GPU, no paid APIs. Runtime uses only the Python standard library.
FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /app
COPY auditgym ./auditgym
COPY configs ./configs
COPY episodes ./episodes
COPY schemas ./schemas
COPY tests ./tests
COPY pyproject.toml README.md LICENSE ./

RUN useradd --create-home auditor && mkdir -p /app/runs && chown -R auditor /app/runs
USER auditor

# Single command: unit tests + data/truth verification + 4 baselines (+ expert) + paired invariance checks
CMD ["python", "-m", "auditgym", "check", "--out", "/app/runs/check"]
