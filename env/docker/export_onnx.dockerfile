FROM tf_yolo_export_base:latest

RUN echo "== Install ONNX Runtime (GPU) from the uv lock (env/uv/export/onnx) ==" && \
    mkdir -p /tmp/onnx
COPY env/uv/export/onnx/pyproject.toml env/uv/export/onnx/uv.lock /tmp/onnx/
RUN uv export --project /tmp/onnx --frozen --no-hashes -o /tmp/onnx/requirements.txt && \
    uv pip install --system -r /tmp/onnx/requirements.txt && \
    rm -rf /tmp/onnx

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
