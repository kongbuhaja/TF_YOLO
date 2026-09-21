FROM tf_yolo_export_base:latest

RUN echo "== Install TensorRT runtime from the uv lock (env/uv/export/tensorrt) ==" && \
    mkdir -p /tmp/tensorrt
COPY env/uv/export/tensorrt/pyproject.toml env/uv/export/tensorrt/uv.lock /tmp/tensorrt/
RUN uv export --project /tmp/tensorrt --frozen --no-hashes -o /tmp/tensorrt/requirements.txt && \
    uv pip install --system -r /tmp/tensorrt/requirements.txt && \
    rm -rf /tmp/tensorrt
