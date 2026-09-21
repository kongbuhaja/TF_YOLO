FROM tf_yolo_edge_base:latest

RUN echo "== Install Coral Edge TPU runtime ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libedgetpu1-std && \
    rm -rf /var/lib/apt/lists/*

RUN echo "== Install LiteRT runtime from the uv lock (env/uv/export/tflite) ==" && \
    mkdir -p /tmp/tflite
COPY env/uv/export/tflite/pyproject.toml env/uv/export/tflite/uv.lock /tmp/tflite/
RUN uv export --project /tmp/tflite --frozen --no-hashes -o /tmp/tflite/requirements.txt && \
    uv pip install --system -r /tmp/tflite/requirements.txt && \
    rm -rf /tmp/tflite
