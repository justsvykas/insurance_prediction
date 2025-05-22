FROM python:3.11-slim AS builder

# I need to install system wide compilers for scikit as it doeesnt have prebuilt wheel.
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# I enable caching for faster build in development
ENV UV_PROJECT_ENVIRONMENT="/usr/local/" \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=240 \
    UV_FAST_INSTALL=1 \
    UV_NO_CACHE=0

RUN uv sync --frozen --no-dev

COPY uv_app/ ./uv_app/

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /usr/local/ /usr/local/
EXPOSE 8000

CMD ["python", "uv_app/api.py"]
