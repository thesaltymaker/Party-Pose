#!/bin/sh
#
# Rebuild the Orin's Party-Pose venv (env/) from scratch.
#
# Usage (on the Orin, from the repo root): sh tools/orin_setup/bootstrap_env.sh [WHEEL]
#
# Needs, outside the repo:
#   ~/opencv-build/install              CUDA OpenCV 4.10.0 (recipe: ~/build_opencv_cuda.sh + CMakeCache.txt)
#   WHEEL (default below)               onnxruntime-gpu 1.23.0 cp310 aarch64, TensorRT + CUDA providers
#   ~/.local numpy 1.26.1               the OpenCV build is compiled against numpy 1.x; do not let pip bring numpy 2
#
# Moves an existing env/ aside instead of deleting it.

set -eu

WHEEL=${1:-$HOME/wheels-recovered/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl}
OPENCV_SITE=$HOME/opencv-build/install/lib/python3.10/site-packages

[ -f "$WHEEL" ] || { echo "Missing wheel: $WHEEL" >&2; exit 1; }
[ -d "$OPENCV_SITE/cv2" ] || { echo "Missing CUDA OpenCV: $OPENCV_SITE" >&2; exit 1; }

if [ -e env ]; then
    mv env "env.old-$(date +%Y%m%d-%H%M%S)"
fi

/usr/bin/python3.10 -m venv --system-site-packages env
echo "$OPENCV_SITE" > env/lib/python3.10/site-packages/00-opencv-cuda.pth
env/bin/pip install -q "$WHEEL" onnx 'ml_dtypes==0.5.4'
# pip pulls numpy 2 as a dependency; remove it so ~/.local numpy 1.26.1 is used.
env/bin/pip uninstall -y -q numpy

env/bin/python -c '
import cv2, numpy, onnxruntime as ort
print("cv2", cv2.__version__, "cuda devices", cv2.cuda.getCudaEnabledDeviceCount())
print("numpy", numpy.__version__)
print("onnxruntime", ort.__version__, ort.get_available_providers())
assert cv2.cuda.getCudaEnabledDeviceCount() == 1
assert numpy.__version__.startswith("1.")
assert "TensorrtExecutionProvider" in ort.get_available_providers()
'
echo "OK"
