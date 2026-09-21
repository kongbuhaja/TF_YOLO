#!/bin/bash

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
  [ -f "$LEGACY/env.sh" ] && source "$LEGACY/env.sh"
  "$LEGACY/.venv/bin/python" - <<'EOF'
try:
    from openvino.runtime import Core
    devices = Core().available_devices
    print("  available_devices:", devices)
    if "MYRIAD" not in devices:
        print("  MYRIAD missing -> host udev rules (97-myriad-usbboot.rules) or USB passthrough")
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
  echo "  libedgetpu MISSING -> sudo apt install libedgetpu1-std"
fi
if [ -x "$TFLITE/.venv/bin/python" ]; then
  "$TFLITE/.venv/bin/python" - <<'EOF'
try:
    from ai_edge_litert.interpreter import load_delegate
    load_delegate("libedgetpu.so.1")
    print("  delegate load: OK")
except Exception as e:
    print("  delegate load:", e)
EOF
fi
