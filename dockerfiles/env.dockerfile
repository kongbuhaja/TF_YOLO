FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update 

RUN echo "== Install system Tools ==" &&\
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --allow-unauthenticated \
        openssh-server vim nano htop tmux sudo \
        git zip build-essential iputils-ping net-tools ufw \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        curl dpkg libgtk2.0-dev cmake libwebp-dev ca-certificates gnupg \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps \
        sysstat lm-sensors && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN echo "== Install uv ==" &&\
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN echo "== Set default python == " && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN echo "== Install Dev Tools ==" &&\
    uv pip install --system \
    tensorflow==2.15 \
    opencv-python \
    matplotlib \
    pillow \
    tqdm \
    tensorflow_datasets \
    gdown \
    pyyaml \
    numpy \
    scipy