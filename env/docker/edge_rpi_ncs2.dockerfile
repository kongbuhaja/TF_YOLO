FROM debian:bullseye-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "== Install system tools ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.9 python3.9-venv python3.9-dev \
        curl ca-certificates libgl1 libglib2.0-0 \
        libpugixml1v5 libtbb2 libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN echo "== Install uv ==" && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN echo "== Install NCS2 deps from the uv lock (env/uv/export/openvino_legacy) ==" && \
    mkdir -p /tmp/openvino_legacy
COPY env/uv/export/openvino_legacy/pyproject.toml env/uv/export/openvino_legacy/uv.lock /tmp/openvino_legacy/
RUN uv export --project /tmp/openvino_legacy --frozen --no-hashes -o /tmp/openvino_legacy/requirements.txt && \
    uv pip install --system -r /tmp/openvino_legacy/requirements.txt && \
    rm -rf /tmp/openvino_legacy

ARG OPENVINO_VERSION=2022.3.2
ARG OPENVINO_REV=9279.e2c7e4d7b4d
RUN echo "== Install OpenVINO ${OPENVINO_VERSION} LTS runtime (last MYRIAD release) ==" && \
    curl -fL "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/l_openvino_toolkit_debian9_${OPENVINO_VERSION}.${OPENVINO_REV}_arm64.tgz" \
        -o /tmp/openvino.tgz && \
    mkdir -p /opt/intel && \
    tar -xzf /tmp/openvino.tgz -C /tmp && \
    mv "/tmp/l_openvino_toolkit_debian9_${OPENVINO_VERSION}.${OPENVINO_REV}_arm64" \
       "/opt/intel/openvino_${OPENVINO_VERSION}" && \
    rm /tmp/openvino.tgz

ENV PYTHONPATH="/opt/intel/openvino_2022.3.2/python/python3.9"
ENV LD_LIBRARY_PATH="/opt/intel/openvino_2022.3.2/runtime/lib/aarch64"
