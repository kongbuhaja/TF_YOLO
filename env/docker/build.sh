#!/bin/bash
set -e

TARGET="${1:-tf}"
DOCKER="${DOCKER:-sudo docker}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONTEXT="$(cd "$HERE/../.." && pwd)"
MOUNT="$(cd "$CONTEXT/.." && pwd)"

case "$TARGET" in
  tf)
    $DOCKER build -f "${HERE}/dev.dockerfile" -t tf_yolo:latest "${CONTEXT}"
    $DOCKER run --gpus '"device=0"' --privileged -v /dev/bus/usb:/dev/bus/usb -m 64g --shm-size=8g -it -v "${MOUNT}":/home/ML -e HOST_PERMS="$(id -u):$(id -g)" --name tf_yolo tf_yolo:latest
    ;;
  tensorrt|tflite|onnx|openvino)
    $DOCKER build -f "${HERE}/export_base.dockerfile" -t tf_yolo_export_base:latest "${CONTEXT}"
    $DOCKER build -f "${HERE}/export_${TARGET}.dockerfile" -t "tf_yolo_export_${TARGET}:latest" "${CONTEXT}"
    echo "Built tf_yolo_export_${TARGET}:latest (PC: x86_64 + NVIDIA GPU)"
    echo "Run: $DOCKER run --rm --gpus '\"device=0\"' --privileged -v /dev/bus/usb:/dev/bus/usb -it -v \"${MOUNT}\":/home/ML tf_yolo_export_${TARGET}:latest"
    ;;
  rpi_cpu|rpi_coral|rpi_ncs2)
    $DOCKER build -f "${HERE}/edge_base.dockerfile" -t tf_yolo_edge_base:latest "${CONTEXT}"
    $DOCKER build -f "${HERE}/edge_${TARGET}.dockerfile" -t "tf_yolo_edge_${TARGET}:latest" "${CONTEXT}"
    echo "Built tf_yolo_edge_${TARGET}:latest (arm64, no GPU needed)"
    case "$TARGET" in
      rpi_cpu)
        echo "Run: $DOCKER run --rm -it -v \"${MOUNT}\":/home/ML tf_yolo_edge_rpi_cpu:latest"
        ;;
      rpi_coral)
        echo "Run: $DOCKER run --rm -it --privileged -v /dev/bus/usb:/dev/bus/usb -v \"${MOUNT}\":/home/ML tf_yolo_edge_rpi_coral:latest"
        echo "     (M.2/PCIe Coral: replace with --device /dev/apex_0:/dev/apex_0)"
        ;;
      rpi_ncs2)
        echo "Host udev rules (once):"
        echo "  sudo cp /opt/intel/openvino_2022.3.2/install_dependencies/97-myriad-usbboot.rules /etc/udev/rules.d/"
        echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
        echo "Run: $DOCKER run --rm -it --privileged -v /dev/bus/usb:/dev/bus/usb -v \"${MOUNT}\":/home/ML tf_yolo_edge_rpi_ncs2:latest"
        ;;
    esac
    ;;
  jetson)
    $DOCKER build -f "${HERE}/edge_jetson.dockerfile" -t tf_yolo_edge_jetson:latest "${CONTEXT}"
    echo "Built tf_yolo_edge_jetson:latest (build ON the Jetson)"
    echo "Run: $DOCKER run --rm -it --runtime nvidia -v \"${MOUNT}\":/home/ML tf_yolo_edge_jetson:latest"
    ;;
  *)
    echo "Usage: ./build.sh [tf|tensorrt|tflite|onnx|openvino|rpi_cpu|rpi_coral|rpi_ncs2|jetson]"
    echo "  tf|tensorrt|tflite|onnx|openvino : PC (x86_64 + NVIDIA GPU)"
    echo "  rpi_cpu|rpi_coral|rpi_ncs2       : Raspberry Pi 4 (arm64), build on device"
    echo "  jetson                           : Jetson JetPack 4.6 / TensorRT 8.2, build on device"
    echo ""
    echo "Or use compose (same images):"
    echo "  docker compose --profile dev up -d dev"
    echo "  docker compose --profile pc build"
    echo "  docker compose --profile edge up -d edge-rpi-coral"
    exit 1
    ;;
esac
