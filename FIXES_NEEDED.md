# FIXES NEEDED

Issues found during code generation that were NOT fixed inline. Each entry includes chunk name, file, description, and suggested fix.

---

## Chunk 2 — topology.py

### Issue 1: FACE_CONNECTIONS is empty stub [STILL OPEN — devstral failed twice]
- **File**: `src/topology.py`
- **Description**: The model hallucinated sequential index pairs (1→2→3→…→478) rather than the real MediaPipe FaceMesh edge topology. Output was also truncated mid-block. Filed as empty list stub.
- **Suggested fix**: Populate `FACE_CONNECTIONS` with the canonical MediaPipe FaceMesh tessellation connection pairs. These can be obtained from the MediaPipe Python library: `mediapipe.solutions.face_mesh.FACEMESH_TESSELATION`, or from the MediaPipe source at `mediapipe/python/solutions/face_mesh_connections.py`. Alternatively, use the well-known static list of ~388 edge pairs from the MediaPipe Face Mesh model documentation.

### Issue 2: CANONICAL_BLENDSHAPE_INDICES is empty stub [STILL OPEN — devstral failed twice]
- **File**: `src/topology.py`
- **Description**: The model's output was truncated before generating this constant. Filed as empty list stub.
- **Suggested fix**: Populate `CANONICAL_BLENDSHAPE_INDICES` with the 146 specific landmark indices that the MediaPipe face blendshapes model expects. These indices are documented in the MediaPipe Tasks source at `mediapipe/tasks/python/vision/face_landmarker.py` or in the ONNX model's metadata. The list is sometimes called the "canonical face model subset" or "blendshape subset indices".

---

## Chunk 3 — config.py

### Issue 3: `--fps` flag maps to `args.fps` but code reads `args.show_fps`
- **File**: `src/config.py`
- **Description**: `parser.add_argument("--fps", ...)` creates `args.fps`, but `parse_args()` reads `args.show_fps`, which does not exist → `AttributeError` at startup.
- **Suggested fix**: Change `show_fps=args.show_fps` to `show_fps=args.fps` in the `Config(...)` constructor call.

### Issue 4: `store_false` args default to `None` instead of `True`
- **File**: `src/config.py`
- **Description**: `--no-face`, `--no-body`, `--no-hands` use `action="store_false"` with no explicit `default=True`. When those flags are absent, argparse sets `args.face/body/hands` to `None` instead of `True`, causing `Config(face=None)` rather than `Config(face=True)`.
- **Suggested fix**: Add `parser.set_defaults(face=True, body=True, hands=True)` after registering the three `--no-*` arguments, or add `default=True` to each of those `add_argument` calls.

---

## Chunk 5 — model_manager.py

### Issue 5: CUDA provider options passed as `SessionOptions` object instead of a dict
- **File**: `src/model_manager.py`
- **Description**: The code creates a second `ort.SessionOptions()` object (`cuda_opts`) and sets attributes like `cuda_opts.arena_extend_strategy`, `cuda_opts.gpu_mem_limit`, `cuda_opts.cudnn_conv_algo_search`, `cuda_opts.do_copy_in_default_stream` on it. These are NOT valid `SessionOptions` attributes. `CUDAExecutionProvider` expects its options as a plain Python `dict`. The `providers` list then passes this `SessionOptions` object where a dict is required → `AttributeError` and/or incorrect CUDA configuration at runtime.
- **Suggested fix**: Replace the `cuda_opts` `SessionOptions` object with a plain dict:
  ```python
  cuda_provider_opts = {
      'arena_extend_strategy': 'kNextPowerOfTwo',
      'gpu_mem_limit': 2 * 1024 ** 3,
      'cudnn_conv_algo_search': 'EXHAUSTIVE',
      'do_copy_in_default_stream': True,
  }
  providers = [('CUDAExecutionProvider', cuda_provider_opts), 'CPUExecutionProvider']
  ```
  Remove the second `ort.SessionOptions()` object entirely.

---

## Chunk 8 — face_processor.py

### Issue 6: `model_manager.run_session()` does not exist
- **File**: `src/face_processor.py`
- **Description**: `process()` and `_run_blendshapes()` call `self.model_manager.run_session(name=..., inputs=..., outputs=...)`. `ModelManager` has no such method — it only exposes `get_session(name) -> InferenceSession`. This will raise `AttributeError` at runtime for every inference call.
- **Suggested fix**: Replace each `self.model_manager.run_session(name, inputs, outputs)` with:
  ```python
  session = self.model_manager.get_session(name)
  results = session.run(outputs, inputs)
  ```
  Note that `session.run(output_names, input_feed)` takes output names as a list and input feed as a dict. Also applies to `hand_processor.py` and `body_processor.py` for the same pattern.

### Issue 7: `FaceResult.landmarks` is `List[List[Landmark]]` instead of `List[Landmark]`
- **File**: `src/face_processor.py`
- **Description**: In `process()`, the code iterates over batch dim N and appends `lm` (a `List[Landmark]`) into `landmarks`, then passes `landmarks` (a list of lists) to `FaceResult(landmarks=landmarks)`. `FaceResult.landmarks` is typed `List[Landmark]`. Since the face landmark model runs per-face (one face per bbox), N is always 1; the loop is unnecessary and adds a nesting level.
- **Suggested fix**: Replace the landmarks loop with a direct call:
  ```python
  landmarks = Preprocessor.to_image_space(
      landmarks_raw[0], LANDMARK_W, LANDMARK_H, crop_bbox, frame_w, frame_h, mirror
  )
  ```

---

## Chunk 9 — hand_processor.py

### Issue 8: Anchor grid formula uses stride instead of grid size
- **File**: `src/hand_processor.py`
- **Description**: `_generate_anchors()` sets `g=8` (stride) for Layer 1 and `g=16` for Layer 2, then computes `cx = (c + 0.5) / g`. This gives anchor center values up to 3.0, not normalized [0,1]. The correct divisor is the grid size: `192 // 8 = 24` for Layer 1 and `192 // 16 = 12` for Layer 2.
- **Suggested fix**: Change `g = 8` to `g = 192 // 8` (= 24) for Layer 1, and `g = 16` to `g = 192 // 16` (= 12) for Layer 2.

### Issue 9: `model_manager.hand_detector_session()` and `hand_landmarks_session()` do not exist
- **File**: `src/hand_processor.py`
- **Description**: `process()` calls `self.model_manager.hand_detector_session(inputs=...)` and `self.model_manager.hand_landmarks_session(inputs=...)`. These methods don't exist on `ModelManager`. See also Issue 6.
- **Suggested fix**: Use `session = self.model_manager.get_session('hand_detector'); outputs = session.run(['Identity', 'Identity_1'], {'input_1': preprocessed})` and similarly for `hand_landmarks`.

### Issue 10: `crop_roi_nhwc` result not unpacked; `to_image_space` uses wrong bbox
- **File**: `src/hand_processor.py`
- **Description**: `crop = Preprocessor.crop_roi_nhwc(...)` assigns the tuple `(arr, crop_bbox)` to a single variable `crop`. The `crop_bbox` is discarded. Then `to_image_space` is called with the original `bbox` (detection box) instead of `crop_bbox` (the padded crop box). Landmark coordinates will be mapped to the wrong region of the image.
- **Suggested fix**: Unpack: `roi_arr, crop_bbox = Preprocessor.crop_roi_nhwc(...)`, then pass `crop_bbox` to `to_image_space` and `roi_arr` as the session input.

### Issue 11: `BoundingBox` constructed with wrong field names in `_decode_detections`
- **File**: `src/hand_processor.py`
- **Description**: `BoundingBox(xmin=..., ymin=..., xmax=..., ymax=..., score=...)` uses field names that don't exist on `BoundingBox`. The correct fields are `x, y, w, h, confidence`. The code also computes corner coordinates (xmin/xmax) but BoundingBox requires top-left + size (x, y, w, h).
- **Suggested fix**: Convert corner → top-left+size before constructing: `BoundingBox(x=float(box[0]), y=float(box[1]), w=float(box[2]-box[0]), h=float(box[3]-box[1]), confidence=float(box[4]))`.

---

## Chunk 11 — renderer.py

### Issue 12: Landmark coordinates multiplied by frame_w/h again (already in pixel space)
- **File**: `src/renderer.py`
- **Description**: All `draw_*` methods compute `x = int(lm.x * frame_w)`, treating landmark coordinates as normalized [0,1]. But `Landmark.x` and `Landmark.y` are already in **image pixel space** (set by `Preprocessor.to_image_space`). Multiplying by `frame_w/h` scales e.g. x=640 to 640×1280=819,200 — all landmarks will be massively out-of-bounds and never drawn. The out-of-bounds check will therefore reject all landmarks.
- **Suggested fix**: Replace `x, y = int(lm.x * frame_w), int(lm.y * frame_h)` with `x, y = int(lm.x), int(lm.y)` in all four draw methods. Update the out-of-bounds check accordingly: `x < -0.05 * frame_w or x > 1.05 * frame_w` remains correct since x is now in pixel space.

### Issue 13: BoundingBox accessed with wrong field names in ROI drawing
- **File**: `src/renderer.py`
- **Description**: `draw_faces` and `draw_hands` access `bbox.x1`, `bbox.y1`, `bbox.x2`, `bbox.y2`, which don't exist on `BoundingBox`. `BoundingBox` has fields `x, y, w, h`.
- **Suggested fix**: Replace with `cv2.rectangle(cpu_frame, (int(bbox.x), int(bbox.y)), (int(bbox.x + bbox.w), int(bbox.y + bbox.h)), COLOR, 2)`. Also remove the `* frame_w / frame_h` scaling that appears alongside these (same Issue 12 — bbox coords are already in pixel space).

---

## Final Integration Review

### Issue 14: `_nms()` in hand_processor.py computes IoU against only the last box, not all remaining boxes
- **File**: `src/hand_processor.py`
- **Description**: Inside `_nms()`, `ovr = inter / (areas[i] + areas[last] - inter)` computes overlap between the max-score box (`i`) and only `boxes[last]` (a single pair). `np.where(ovr <= threshold)` then returns a 0- or 1-element result. The loop removes at most one box per iteration, meaning suppression of nearby boxes never happens correctly. The NMS will return too many overlapping detections.
- **Suggested fix**: Standard NMS should compare the top-scoring box against ALL remaining boxes simultaneously. Replace the single-pair IoU with a vectorized IoU across the remaining boxes, then filter out all with IoU > threshold:
  ```python
  xx1 = np.maximum(x1[i], x1)
  yy1 = np.maximum(y1[i], y1)
  xx2 = np.minimum(x2[i], x2)
  yy2 = np.minimum(y2[i], y2)
  w = np.maximum(0, xx2 - xx1)
  h = np.maximum(0, yy2 - yy1)
  ovr = (w * h) / (areas[i] + areas - w * h)
  inds = np.where(ovr <= self.NMS_IOU_THRESHOLD)[0]
  ```

### Issue 15: `renderer.py` performs 4 separate GPU↔CPU round-trips per frame
- **File**: `src/renderer.py`
- **Description**: Each of `draw_faces`, `draw_hands`, `draw_body`, `draw_fps` independently downloads the GpuMat, draws on CPU, and re-uploads. This results in 4 full-frame GPU→CPU→GPU transfers per frame, contradicting the architecture's GPU-residency design goal.
- **Suggested fix**: Refactor `Renderer` to accept a single CPU frame (numpy array) rather than a GpuMat. Download once in the main loop before all draw calls, pass the CPU frame to each method, then re-upload once after all drawing is done. Or combine all drawing into a single `draw_all()` method that downloads once, calls all sub-draws, re-uploads once.

---
