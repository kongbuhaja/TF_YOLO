FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "== Install system tools ==" && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --allow-unauthenticated \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        curl ca-certificates \
        libgl1-mesa-glx libglu1-mesa-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "== Install uv ==" && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN echo "== Set default python ==" && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11
