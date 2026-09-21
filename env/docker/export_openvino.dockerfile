FROM tf_yolo_export_base:latest

RUN echo "== Install OpenVINO runtime from the uv lock (env/uv/export/openvino) ==" && \
    mkdir -p /tmp/openvino
COPY env/uv/export/openvino/pyproject.toml env/uv/export/openvino/uv.lock /tmp/openvino/
RUN uv export --project /tmp/openvino --frozen --no-hashes -o /tmp/openvino/requirements.txt && \
    uv pip install --system -r /tmp/openvino/requirements.txt && \
    rm -rf /tmp/openvino
