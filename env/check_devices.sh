#!/bin/bash

if [ ! -f /.dockerenv ] && [ ! -d /opt/intel/openvino_2022.3.2 ] && command -v docker >/dev/null 2>&1; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx tf_yolo; then
    echo "(Host detected -> checking inside 'tf_yolo' container)"
    exec docker exec tf_yolo sh /home/ML/TF_YOLO/env/check_devices.sh
  fi
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGACY="$ROOT/env/uv/export/openvino_legacy"
TFLITE="$ROOT/env/uv/export/tflite"

echo "== /dev/bus/usb =="
if [ -d /dev/bus/usb ]; then
  total=0
  readable=0
  for n in /dev/bus/usb/*/*; do
    [ -c "$n" ] || continue
    total=$((total + 1))
    if { true < "$n"; } 2>/dev/null; then
      readable=$((readable + 1))
    fi
  done
  echo "  present: $total nodes, readable: $readable"
  if [ "$total" -gt 0 ] && [ "$readable" -eq 0 ]; then
    echo "  ACCESS DENIED -> needs '--privileged -v /dev/bus/usb:/dev/bus/usb' (or --device-cgroup-rule 'c 189:* rmw')"
  fi
else
  echo "  MISSING -> no USB passthrough; recreate with --privileged -v /dev/bus/usb:/dev/bus/usb"
fi

echo
echo "== USB ids =="
ncs2=0
coral=0
for d in /sys/bus/usb/devices/*/; do
  v=$(cat "$d/idVendor" 2>/dev/null)
  p=$(cat "$d/idProduct" 2>/dev/null)
  case "$v:$p" in
    03e7:2485) echo "  NCS2 detected (03e7:2485)"; ncs2=1 ;;
    18d1:9302) echo "  Coral detected (18d1:9302, firmware loaded)"; coral=1 ;;
    1a6e:089a) echo "  Coral detected (1a6e:089a, bootloader; switches to 18d1:9302 once opened)"; coral=1 ;;
  esac
done
[ "$ncs2" -eq 0 ] && echo "  NCS2 not detected"
[ "$coral" -eq 0 ] && echo "  Coral not detected"

echo
echo "== NCS2 (OpenVINO MYRIAD) =="
if [ -x "$LEGACY/.venv/bin/python" ]; then
  [ -f "$LEGACY/env.sh" ] && . "$LEGACY/env.sh"
  "$LEGACY/.venv/bin/python" - <<'EOF'
try:
    from openvino.runtime import Core
    devices = Core().available_devices
    print("  available_devices:", devices)
    if "MYRIAD" not in devices:
        print("  MYRIAD missing -> host udev rules (97-myriad-usbboot.rules) or run inside tf_yolo container")
except Exception as e:
    print("  openvino import failed:", e)
EOF
else
  echo "  env missing -> cd env/uv/export/openvino_legacy && bash setup.sh"
fi

echo
echo "== Coral (libedgetpu) =="
if ldconfig -p 2>/dev/null | grep -q libedgetpu; then
  echo "  libedgetpu: $(ldconfig -p | grep libedgetpu | head -1 | awk '{print $1}')"
else
  echo "  libedgetpu MISSING -> run inside tf_yolo container (or on host: sudo dpkg -i libedgetpu1-std.deb)"
fi
PY_BIN="python3"
if [ -x "$TFLITE/.venv/bin/python" ] && "$TFLITE/.venv/bin/python" -c "import ai_edge_litert" 2>/dev/null; then
  PY_BIN="$TFLITE/.venv/bin/python"
fi
"$PY_BIN" - <<'EOF'
try:
    try:
        from ai_edge_litert.interpreter import load_delegate
    except ImportError:
        try:
            from tflite_runtime.interpreter import load_delegate
        except ImportError:
            import tensorflow as tf
            load_delegate = tf.lite.experimental.load_delegate
    load_delegate("libedgetpu.so.1")
    print("  delegate load: OK")
except Exception as e:
    print("  delegate load:", e)
EOF
