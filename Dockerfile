# ═══════════════════════════════════════════════════════════════════
# Delivery Tracker OpenEnv Benchmark — Production Dockerfile
# ═══════════════════════════════════════════════════════════════════
#
# Target:   2 vCPU · 8 GB RAM  (Hugging Face Spaces free tier)
# Base:     Python 3.11 slim   → final image ≈ 200 MB
# Health:   GET /healthz  on $PORT (default 7860)
# Reset:    GET /reset    on $PORT → deterministic env reset
#
# Build:    docker build -t delivery-tracker-openenv .
# Run:      docker run --rm -p 7860:7860 \
#             -e API_BASE_URL=... -e MODEL_NAME=... -e OPENAI_API_KEY=... \
#             delivery-tracker-openenv
#
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: dependency install (cached layer) ─────────────────────
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy ONLY requirements for maximum layer-cache hit rate
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: final runtime image ──────────────────────────────────
FROM python:3.11-slim AS runtime

# OCI metadata labels
LABEL maintainer="u1krsh" \
      description="OpenEnv Delivery Tracker Benchmark" \
      version="1.0.0" \
      org.opencontainers.image.title="delivery-tracker-openenv" \
      org.opencontainers.image.description="A real-world delivery dispatch benchmark for evaluating AI agents" \
      org.opencontainers.image.source="https://github.com/u1krsh/AccurateFoodDelivery" \
      org.opencontainers.image.licenses="MIT"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    # Graceful shutdown timeout (seconds)
    SHUTDOWN_TIMEOUT=10 \
    # Default task subset (empty = all tasks)
    TASKS=""

WORKDIR /app

# Copy pre-installed packages from deps stage (avoids carrying pip/build artifacts)
COPY --from=deps /install/lib /usr/local/lib
COPY --from=deps /install/bin /usr/local/bin

# Create non-root user before copying code (security best practice)
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy application code (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Build-time smoke test: validate env loads, tasks parse, grader initialises
RUN python -c "\
from env import DeliveryEnvironment; \
from env.tasks import get_task, TASK_IDS; \
from env.graders import TaskGrader; \
cfg = get_task('easy'); \
env = DeliveryEnvironment(cfg); \
obs = env.reset(); \
assert len(obs.nodes) == 4, f'Expected 4 nodes, got {len(obs.nodes)}'; \
g = TaskGrader(); \
report = g.grade_detailed(env); \
print(f'BUILD OK  │ tasks={len(TASK_IDS)}  nodes={len(obs.nodes)}  grader={report.score:.2f}')"

# Expose health-check / probe port
EXPOSE ${PORT}

# ── Docker HEALTHCHECK ─────────────────────────────────────────────
# Liveness probe: GET /healthz must return HTTP 200.
# Timing: first check after 5 s, then every 30 s, 3 s timeout, 3 retries.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
try: \
    r = urllib.request.urlopen('http://localhost:${PORT}/healthz', timeout=2); \
    sys.exit(0 if r.status == 200 else 1); \
except: sys.exit(1)"

# Drop to non-root
USER appuser

# ── Startup ────────────────────────────────────────────────────────
# entrypoint.py:  1) starts /healthz server (daemon thread)
#                 2) runs inference against all (or subset) tasks
#                 3) exits cleanly
#
# Override CMD to customise behaviour:
#   docker run ... delivery-tracker-openenv python healthcheck.py   # health-only
#   docker run ... delivery-tracker-openenv python inference.py     # inference-only
ENTRYPOINT ["python", "entrypoint.py"]
