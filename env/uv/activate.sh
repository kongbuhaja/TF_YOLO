SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TF_USE_LEGACY_KERAS=1
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi
# Expose pip-installed NVIDIA libs (cuDNN 9 for onnxruntime-gpu, etc.) to the dynamic loader
_NVIDIA_LIBS=$(python3 -c "import glob,sysconfig; print(':'.join(d for p in {sysconfig.get_path('purelib'), sysconfig.get_path('platlib')} for d in glob.glob(p + '/nvidia/*/lib'))" 2>/dev/null)
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _NVIDIA_LIBS
