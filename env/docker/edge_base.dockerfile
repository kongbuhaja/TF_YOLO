FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "== Install system tools ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        curl ca-certificates libgl1 libglib2.0-0 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN echo "== Install uv ==" && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
