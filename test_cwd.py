#!/usr/bin/env python3
import os
import sys

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print(f"\nChecking for models directory:")
print(f"  ./models exists: {os.path.exists('./models')}")
print(f"  ./models/pose_landmark_full_body.onnx exists: {os.path.exists('./models/pose_landmark_full_body.onnx')}")

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
print(f"\nAbsolute models path: {models_path}")
print(f"Absolute models path exists: {os.path.exists(models_path)}")

if os.path.exists(models_path):
    print(f"\nFiles in models directory:")
    for f in os.listdir(models_path):
        print(f"  - {f}")
