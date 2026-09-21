FROM tf_yolo_edge_base:latest

RUN echo "== Install CPU runtimes from the uv locks (tflite + onnx + openvino) ==" && \
    mkdir -p /tmp/tflite /tmp/onnx /tmp/openvino
COPY env/uv/export/tflite/pyproject.toml env/uv/export/tflite/uv.lock /tmp/tflite/
COPY env/uv/export/onnx/pyproject.toml env/uv/export/onnx/uv.lock /tmp/onnx/
COPY env/uv/export/openvino/pyproject.toml env/uv/export/openvino/uv.lock /tmp/openvino/
RUN uv export --project /tmp/tflite --frozen --no-hashes -o /tmp/tflite/requirements.txt && \
    uv export --project /tmp/onnx --frozen --no-hashes -o /tmp/onnx/requirements.txt && \
    uv export --project /tmp/openvino --frozen --no-hashes -o /tmp/openvino/requirements.txt && \
    uv pip install --system \
        -r /tmp/tflite/requirements.txt \
        -r /tmp/onnx/requirements.txt \
        -r /tmp/openvino/requirements.txt && \
    rm -rf /tmp/tflite /tmp/onnx /tmp/openvino
