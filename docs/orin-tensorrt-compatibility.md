# Orin TensorRT compatibility report
| Model | Status | Cold (s) | Warm (s) | Shapes Match | Error |
|---|---|---|---|---|---|
| person_detector | PASS | 0.856 | 0.016 | True |  |
| face_landmarks | PASS | 0.048 | 0.006 | True |  |
| face_blendshapes | PASS | 0.062 | 0.002 | True |  |
| hand_landmarks | PASS | 0.014 | 0.004 | True |  |
| pose_landmarks | PASS | 0.050 | 0.007 | True |  |

## Update 2026-09-26 (issue #2)

The person_detector "PASS" above reflected a clean fallback to the CUDA EP, because the model was listed in `_TRT_EXCLUDED_MODELS`. That exclusion is gone. A full-model TensorRT build takes about 6 min and then fails with `TRT-16198 - Layers missing empty tensor support` in the NMS post-processing; it does not hang. The backbone now runs in TensorRT through `ModelManager._TRT_EXTRA_OPTS`. Op types that appear only after NMS are passed to `trt_op_types_to_exclude`, and `trt_min_subgraph_size=20` is set.

| person_detector | Provider | Inference (clocks pinned) | Real-frame detections |
|---|---|---|---|
| before | CUDA EP | 8.1 ms | reference |
| after | TensorRT backbone + CUDA NMS | 3.1 ms | same boxes (±0.4 px) |

Rebuilding the other engines removed the "different models of devices" warning but did not change speed. See `docs/orin-fps-issue-2.md`.
