FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "== Install system tools ==" && \
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

RUN echo "== Install uv ==" && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN echo "== Set default python ==" && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN echo "== Install dev tools from the uv lock (env/uv) ==" && \
    mkdir -p /tmp/dev
COPY env/uv/pyproject.toml env/uv/uv.lock /tmp/dev/
RUN uv export --project /tmp/dev --frozen --no-hashes -o /tmp/dev/requirements.txt && \
    uv pip install --system -r /tmp/dev/requirements.txt && \
    rm -rf /tmp/dev

RUN echo "== Install Coral Edge TPU runtime (libedgetpu) + compiler ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.ubuntu22.04_amd64.deb" -o /tmp/libedgetpu1-std.deb && \
    dpkg -i /tmp/libedgetpu1-std.deb && \
    rm -f /tmp/libedgetpu1-std.deb && \
    mkdir -p /usr/local/bin/edgetpu_compiler_bin && \
    curl -fsSL "https://raw.githubusercontent.com/mbrooksx/edgetpu-compilers/main/16.0/edgetpu_compiler" -o /usr/local/bin/edgetpu_compiler && \
    for f in NOTICE edgetpu_compiler ld-linux-x86-64.so.2 libc++.so.1 libc++abi.so.1 libc.so.6 libdl.so.2 libm.so.6 libpthread.so.0 libresolv.so.2; do \
        curl -fsSL "https://raw.githubusercontent.com/mbrooksx/edgetpu-compilers/main/16.0/edgetpu_compiler_bin/$f" -o "/usr/local/bin/edgetpu_compiler_bin/$f"; \
    done && \
    chmod +x /usr/local/bin/edgetpu_compiler /usr/local/bin/edgetpu_compiler_bin/edgetpu_compiler /usr/local/bin/edgetpu_compiler_bin/ld-linux-x86-64.so.2

ARG OPENVINO_LEGACY=2022.3.2
ARG OPENVINO_LEGACY_REV=9279.e2c7e4d7b4d
RUN echo "== Install OpenVINO ${OPENVINO_LEGACY} LTS runtime for NCS2 (MYRIAD plugin) ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends libpugixml1v5 libtbb2 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fL "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_LEGACY}/linux/l_openvino_toolkit_ubuntu20_${OPENVINO_LEGACY}.${OPENVINO_LEGACY_REV}_x86_64.tgz" \
        -o /tmp/openvino.tgz && \
    mkdir -p /opt/intel && \
    tar -xzf /tmp/openvino.tgz -C /tmp && \
    mv "/tmp/l_openvino_toolkit_ubuntu20_${OPENVINO_LEGACY}.${OPENVINO_LEGACY_REV}_x86_64" \
       "/opt/intel/openvino_${OPENVINO_LEGACY}" && \
    rm /tmp/openvino.tgz

ENV TF_USE_LEGACY_KERAS=1
# cuDNN 9 (pip nvidia-cudnn-cu12) for onnxruntime-gpu; base image ships cuDNN 8
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
