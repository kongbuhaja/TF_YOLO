FROM tf_yolo_edge_base:latest

RUN echo "== Install Coral Edge TPU runtime ==" && \
    apt-get update && \
    apt-get install -y --no-install-recommends libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.bookworm_arm64.deb" -o /tmp/libedgetpu1-std.deb && \
    dpkg -i /tmp/libedgetpu1-std.deb && \
    rm -f /tmp/libedgetpu1-std.deb

RUN echo "== Install LiteRT runtime from the uv lock (env/uv/export/tflite) ==" && \
    mkdir -p /tmp/tflite
COPY env/uv/export/tflite/pyproject.toml env/uv/export/tflite/uv.lock /tmp/tflite/
RUN uv export --project /tmp/tflite --frozen --no-hashes -o /tmp/tflite/requirements.txt && \
    uv pip install --system -r /tmp/tflite/requirements.txt && \
    rm -rf /tmp/tflite
