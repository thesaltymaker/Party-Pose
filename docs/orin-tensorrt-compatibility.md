# Orin TensorRT compatibility report
| Model | Status | Cold (s) | Warm (s) | Shapes Match | Error |
|---|---|---|---|---|---|
| person_detector | ERROR | n/a | n/a | False | [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during initialization: /opt/onnxruntime/onnxruntime/core/providers/tensorrt/tensorrt_execution_provider.cc:2315 SubGraphCollection_t onnxruntime::TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t, int, int, const onnxruntime::GraphViewer&, bool*) const [ONNXRuntimeError] : 1 : FAIL : TensorRT input: NonMaxSuppression__618:0 has no shape specified. Please run shape inference on the onnx model first. Details can be found in https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#shape-inference-for-tensorrt-subgraphs  |
| face_landmarks | PASS | 0.057 | 0.007 | True |  |
| face_blendshapes | PASS | 0.067 | 0.002 | True |  |
| hand_landmarks | PASS | 0.006 | 0.004 | True |  |
| pose_landmarks | ERROR | n/a | n/a | False | timed out after 120s |
