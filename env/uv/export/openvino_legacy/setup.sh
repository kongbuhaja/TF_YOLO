#!/bin/bash
set -e

OV_VERSION=2022.3.2
OV_REV=9279.e2c7e4d7b4d
ARCH="$(uname -m)"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

case "$ARCH" in
  x86_64)
    OV_PKG="l_openvino_toolkit_ubuntu20_${OV_VERSION}.${OV_REV}_x86_64.tgz"
    OV_LIBDIR="intel64"
    PY_VER=3.10
    ;;
  aarch64)
    OV_PKG="l_openvino_toolkit_debian9_${OV_VERSION}.${OV_REV}_arm64.tgz"
    OV_LIBDIR="aarch64"
    PY_VER=3.9
    ;;
  *)
    echo "Unsupported arch: $ARCH (NCS2 MYRIAD exists only for x86_64 and aarch64)"
    exit 1
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

OV_DIR="/opt/intel/openvino_${OV_VERSION}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; install manually: libpugixml1v5 libtbb2 libusb-1.0-0"
else
  echo "== Install system deps (pugixml, tbb, libusb) =="
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libpugixml1v5 libtbb2 libusb-1.0-0
fi

if [ ! -d "$OV_DIR" ]; then
  echo "== Download OpenVINO ${OV_VERSION} LTS runtime (last MYRIAD/NCS2 release) =="
  $SUDO mkdir -p /opt/intel
  curl -fL "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OV_VERSION}/linux/${OV_PKG}" \
      -o /tmp/openvino.tgz
  $SUDO tar -xzf /tmp/openvino.tgz -C /opt/intel
  $SUDO mv "/opt/intel/${OV_PKG%.tgz}" "$OV_DIR"
  rm /tmp/openvino.tgz
fi

if command -v udevadm >/dev/null 2>&1; then
  echo "== Install NCS2 udev rules (host) =="
  $SUDO cp "$OV_DIR/install_dependencies/97-myriad-usbboot.rules" /etc/udev/rules.d/ || true
  $SUDO udevadm control --reload-rules || true
  $SUDO udevadm trigger || true
else
  echo "== udevadm not found (container); install 97-myriad-usbboot.rules on the host =="
fi

echo "== Create python ${PY_VER} venv and install deps =="
cd "$DIR"
uv sync --python "$PY_VER"

cat > "$DIR/env.sh" <<EOF
export PYTHONPATH="$OV_DIR/python/python${PY_VER}:\$PYTHONPATH"
export LD_LIBRARY_PATH="$OV_DIR/runtime/lib/${OV_LIBDIR}:\$LD_LIBRARY_PATH"
EOF

echo
echo "Check the NCS2 (expect MYRIAD in the list):"
echo "  source $DIR/env.sh"
echo "  $DIR/.venv/bin/python -c \"from openvino.runtime import Core; print(Core().available_devices)\""
echo
echo "Run exported ONNX models on the NCS2:"
echo "  source $DIR/env.sh"
echo "  $DIR/.venv/bin/python $DIR/../../../test.py --mode run --model <...>/onnx_fp32/model.onnx --image <img> --gpu ncs2"
