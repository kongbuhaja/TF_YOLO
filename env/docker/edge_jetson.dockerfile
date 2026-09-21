FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "== Install system tools ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-setuptools python3-wheel python3-libnvinfer \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# JetPack 4.6 ships Python 3.6 (below the uv floor), so pins cannot come from env/uv
RUN echo "== Install inference base (JetPack 4.6 / TensorRT 8.2 / CUDA 10.2) ==" && \
    python3 -m pip install --no-cache-dir \
    numpy==1.19.5 \
    opencv-python==4.5.5.64

ENV OPENBLAS_CORETYPE=ARMV8

RUN echo "== Smoke check ==" && \
    python3 -c "import numpy, cv2, tensorrt; print('TensorRT', tensorrt.__version__)"
