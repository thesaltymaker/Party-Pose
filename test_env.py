#!/usr/bin/env python3
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import onnxruntime as ort
    print(f"\nonnxruntime installed: YES")
    print(f"onnxruntime version: {ort.__version__}")
    print(f"onnxruntime providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"\nonnxruntime installed: NO")
    print(f"Error: {e}")
