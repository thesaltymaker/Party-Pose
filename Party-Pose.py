#!/usr/bin/env python3
import cv2
import time
import numpy as np
import onnxruntime as ort
import threading
import queue
import argparse
import os

# ===== MODEL PATHS =====
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

POSE_DET_ONNX = os.path.join(MODELS_DIR, "pose_detection_128x128_float32.onnx")
POSE_LMK_ONNX = os.path.join(MODELS_DIR, "pose_landmark_full_body.onnx")
FACE_DET_ONNX = os.path.join(MODELS_DIR, "det_2.5g.onnx")
FACE_MESH_ONNX = os.path.join(MODELS_DIR, "face_landmark_192x192.onnx")
HAND_DET_ONNX = os.path.join(MODELS_DIR, "MediaPipeHandDetector.onnx")
HAND_LMK_ONNX = os.path.join(MODELS_DIR, "hand_landmark.onnx")

# ===== INPUT SIZES =====
POSE_DET_INP = 128
POSE_LMK_INP = 256
FACE_DET_INP = 320
FACE_MESH_INP = 256
HAND_DET_INP = 256
HAND_LMK_INP = 224

# ===== THRESHOLDS =====
POSE_CONF_THR = 0.7
POSE_IOU_THR = 0.2
POSE_ROI_SCALE = 2.0
FACE_SCORE_THR = 0.50
FACE_IOU_THR = 0.4
HAND_SCORE_THR = 0.85
HAND_IOU_THR = 0.5

# ===== VISUALIZATION =====
DRAW_POSE = True
DRAW_FACE_MESH = True
DRAW_HANDS = True
DRAW_DEBUG_INFO = True
DRAW_POSE_ROI = False  # Debug: show body detection bounding box
SKELLY_MODE = False
DRAW_SANTA_HAT = False

# Pose
POSE_DOT_SIZE = 4
POSE_LINE_WIDTH = 3
POSE_DOT_COLOR = (0, 255, 0)
POSE_LINE_COLOR = (0, 255, 255)

# FaceMesh
FACE_MESH_DOT_SIZE = 1
FACE_MESH_LINE_WIDTH = 1
FACE_MESH_COLOR = (200, 230, 245)

# Hands
HAND_DOT_SIZE = 3
HAND_LINE_WIDTH = 2
HAND_DOT_COLOR = (0, 255, 0)
HAND_LINE_COLOR = (200, 230, 245)
HAND_FINGERTIP_COLOR = (0, 0, 255)
HAND_WRIST_COLOR = (255, 0, 0)

# Santa Hat Colors
SANTA_HAT_RED = (0, 0, 200)
SANTA_HAT_WHITE = (255, 255, 255)

GREEN = (0, 255, 0)

# ===== POSE CONNECTIONS (upper body only) =====
POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 23), (23, 24), (24, 12),  # Torso
    (12, 14), (14, 16),  # Right arm
    (11, 13), (13, 15),  # Left arm
]

# ===== HAND CONNECTIONS =====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ===== FACEMESH CONTOUR =====
FACE_MESH_CONTOUR = [
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
    (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    (133, 155), (155, 154), (154, 153), (153, 145),
    (145, 144), (144, 163), (163, 7), (7, 33),
    (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263),
    (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362),
    (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (291, 375), (375, 321),
    (321, 405), (405, 314), (314, 17), (17, 84),
    (84, 181), (181, 91), (91, 146), (146, 61),
]

# ===== FACEMESH FEATURE DEFINITIONS (for filled face mode) =====
LEFT_EYE = [189, 221, 222, 223, 224, 225, 113, 226, 31, 228, 229, 230, 231, 232, 112, 189]
RIGHT_EYE = [413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 341, 413]
NOSE = [4, 44, 237, 218, 131, 198, 196, 197, 419, 420, 360, 438,  457, 274, 4]
MOUTH_OUTER = [187, 207, 216, 57, 82, 312, 287, 436, 427, 434, 422, 405, 181, 202, 214,  187]

FACE_CONTOUR_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 352, 411, 416,  397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 213, 147,
     234, 127, 162, 21, 54, 103, 67, 109
]

class ThreadedCamera:
    def __init__(self, src=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {src}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(3, width)
        self.cap.set(4, height)

        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera {src} opened but cannot read frames")

        self.q = queue.Queue(maxsize=2)
        self.stopped = False

        print(f"[INFO] Camera {src} initialized: {self.cap.get(3):.0f}x{self.cap.get(4):.0f}")

    def start(self):
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        return self

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            if not self.q.full():
                self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def generate_blazepose_anchors():
    """Generate anchors for BlazePose detection model (128x128 input, 896 anchors)."""
    anchors = []
    # Layer 0: 16x16 grid, 2 anchors per cell
    for y in range(16):
        for x in range(16):
            for _ in range(2):
                anchors.append([(x + 0.5) / 16, (y + 0.5) / 16])
    # Layer 1: 8x8 grid, 6 anchors per cell
    for y in range(8):
        for x in range(8):
            for _ in range(6):
                anchors.append([(x + 0.5) / 8, (y + 0.5) / 8])
    return np.array(anchors, dtype=np.float32)


def decode_pose_detections(outputs, anchors, score_thresh=0.5):
    """Decode BlazePose detection model outputs for ALL detections above threshold.

    Returns:
        List of detection dicts with cx, cy, w, h, score, keypoints
    """
    # Output 0: scores (1, 896, 1)
    # Output 1: regressors (1, 896, 12)
    scores_raw = outputs[0][0]  # Shape: (896, 1)
    regressors = outputs[1][0]  # Shape: (896, 12)

    # Sigmoid with clipping to avoid overflow
    scores_raw = np.clip(scores_raw, -50, 50)
    scores = 1 / (1 + np.exp(-scores_raw))
    scores = scores.flatten()

    # Find ALL detections above threshold
    valid_indices = np.where(scores >= score_thresh)[0]

    if len(valid_indices) == 0:
        return []

    detections = []
    for idx in valid_indices:
        detection = regressors[idx]
        anchor = anchors[idx]
        score = scores[idx]

        # Decode bounding box (MediaPipe format: yc, xc, h, w)
        cy = detection[0] / 128.0 + anchor[1]
        cx = detection[1] / 128.0 + anchor[0]
        h = detection[2] / 128.0
        w = detection[3] / 128.0

        # Decode keypoints (for ROI rotation calculation)
        keypoints = []
        for i in range(4):
            kp_y = detection[4 + i * 2] / 128.0 + anchor[1]
            kp_x = detection[4 + i * 2 + 1] / 128.0 + anchor[0]
            keypoints.append((kp_x, kp_y))

        detections.append({
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'score': score,
            'keypoints': keypoints
        })

    return detections


def nms_pose_detections(detections, iou_thresh=0.3):
    """Apply Non-Maximum Suppression to pose detections."""
    if len(detections) == 0:
        return []

    # Convert to boxes format for NMS
    boxes = np.array([[d['cx'] - d['w']/2, d['cy'] - d['h']/2,
                       d['cx'] + d['w']/2, d['cy'] + d['h']/2]
                      for d in detections])
    scores = np.array([d['score'] for d in detections])

    # Apply NMS
    keep = nms_boxes(boxes, scores, iou_thresh)

    return [detections[i] for i in keep]


def get_pose_roi(detection, frame_w, frame_h, downward_scale=POSE_ROI_SCALE):
    """Get SQUARE ROI crop coordinates from pose detection bounding box.

    Uses the raw bounding box from the detection model, extending downward
    to capture more torso, then makes the ROI square (required by landmark model).

    Args:
        detection: Detection dict with 'cx', 'cy', 'w', 'h' keys
        frame_w, frame_h: Original frame dimensions
        downward_scale: How much to extend downward (1.0 = original, 2.0 = 100% more)

    Returns:
        Tuple of (x1, y1, x2, y2) crop coordinates (always square)
    """
    # Use raw bounding box from detection
    cx = detection['cx'] * frame_w
    cy = detection['cy'] * frame_h
    w = detection['w'] * frame_w
    h = detection['h'] * frame_h

    # Extend height downward (gravity-aware)
    extended_h = h * downward_scale

    # Make it square using the larger dimension
    size = max(w, extended_h)

    # Center horizontally on detection center, anchor top vertically
    top_y = cy - h / 2  # Original top of detection box

    x1 = int(max(0, cx - size / 2))
    y1 = int(max(0, top_y))
    x2 = int(min(frame_w, cx + size / 2))
    y2 = int(min(frame_h, y1 + size))

    # Ensure square by adjusting if we hit frame boundaries
    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w != actual_h:
        # Use the smaller dimension to keep it square
        final_size = min(actual_w, actual_h)
        # Re-center
        x1 = int(max(0, cx - final_size / 2))
        x2 = x1 + final_size
        y2 = y1 + final_size

    return x1, y1, x2, y2


def generate_blazepalm_anchors(input_size=256):
    strides = [8, 16, 32]
    anchor_scales = [[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0]]

    anchors = []
    for stride_idx, stride in enumerate(strides):
        grid_size = input_size // stride
        scales = anchor_scales[stride_idx]

        for y in range(grid_size):
            for x in range(grid_size):
                for scale in scales:
                    cx = (x + 0.5) / grid_size
                    cy = (y + 0.5) / grid_size
                    anchor_w = anchor_h = scale / input_size * stride
                    anchors.append([cx, cy, anchor_w, anchor_h])

    return np.array(anchors, dtype=np.float32)

def decode_palm_with_anchors(outs, anchors):
    a0, a1 = np.array(outs[0]), np.array(outs[1])

    if a0.shape[-1] == 1 and a1.shape[-1] >= 4:
        scores_arr, boxes_arr = a0, a1
    elif a1.shape[-1] == 1 and a0.shape[-1] >= 4:
        scores_arr, boxes_arr = a1, a0
    else:
        scores_arr, boxes_arr = (a0, a1) if a0.shape[-1] < a1.shape[-1] else (a1, a0)

    scores = scores_arr.reshape(-1)
    if scores.min() < 0.0 or scores.max() > 1.0:
        logits = np.clip(scores, -50.0, 50.0)
        scores = 1.0 / (1.0 + np.exp(-logits))

    boxes_arr = boxes_arr.reshape(-1, boxes_arr.shape[-1])
    offsets = boxes_arr[:, :4]

    anchor_cx, anchor_cy = anchors[:, 0], anchors[:, 1]
    anchor_w, anchor_h = anchors[:, 2], anchors[:, 3]
    scale_x = scale_y = 1.0 / 256.0

    pred_cx = np.clip(anchor_cx + offsets[:, 0] * scale_x, 0, 1)
    pred_cy = np.clip(anchor_cy + offsets[:, 1] * scale_y, 0, 1)
    pred_w = np.clip(offsets[:, 2] * scale_x, 0.01, 1)
    pred_h = np.clip(offsets[:, 3] * scale_y, 0.01, 1)

    boxes = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)
    return boxes, scores

def decode_scrfd_detections(scores, bboxes, img_shape, input_size=640, score_thresh=0.5):
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2

    H, W = img_shape[:2]
    scale_h = H / input_size
    scale_w = W / input_size

    all_faces = []

    for idx, stride in enumerate(feat_stride_fpn):
        score = scores[idx].reshape(-1, 1)
        bbox = bboxes[idx].reshape(-1, 4)

        pos_inds = np.where(score[:, 0] >= score_thresh)[0]
        if len(pos_inds) == 0:
            continue

        score = score[pos_inds]
        bbox = bbox[pos_inds]

        fh = fw = input_size // stride
        anchor_centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape(-1, 2)

        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(-1, 2)

        anchor_centers = anchor_centers[pos_inds]

        x1 = (anchor_centers[:, 0] - bbox[:, 0] * stride) * scale_w
        y1 = (anchor_centers[:, 1] - bbox[:, 1] * stride) * scale_h
        x2 = (anchor_centers[:, 0] + bbox[:, 2] * stride) * scale_w
        y2 = (anchor_centers[:, 1] + bbox[:, 3] * stride) * scale_h

        faces = np.stack([x1, y1, x2, y2, score[:, 0]], axis=1)
        all_faces.append(faces)

    if len(all_faces) == 0:
        return np.array([])

    return np.vstack(all_faces)

def nms_boxes(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return np.array([])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

def boxes_to_xyxy(boxes, W, H):
    boxes = boxes.astype(np.float32)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    out = np.stack([x1, y1, x2, y2], axis=1)
    out[:, 0::2] = np.clip(out[:, 0::2], 0, W-1)
    out[:, 1::2] = np.clip(out[:, 1::2], 0, H-1)
    return out.astype(int)

def suppress_overlapping_detections(hand_boxes, face_boxes, iou_threshold=0.2):
    if len(hand_boxes) == 0 or len(face_boxes) == 0:
        return np.arange(len(hand_boxes), dtype=np.int32)

    keep = []
    for i, hand_box in enumerate(hand_boxes):
        hx1, hy1, hx2, hy2 = hand_box
        hand_area = (hx2 - hx1) * (hy2 - hy1)

        overlaps_face = False
        for face in face_boxes:
            fx1, fy1, fx2, fy2 = face[:4]

            xx1 = max(hx1, fx1)
            yy1 = max(hy1, fy1)
            xx2 = min(hx2, fx2)
            yy2 = min(hy2, fy2)

            if xx2 > xx1 and yy2 > yy1:
                inter_area = (xx2 - xx1) * (yy2 - yy1)
                iou = inter_area / hand_area

                if iou > iou_threshold:
                    overlaps_face = True
                    break

        if not overlaps_face:
            keep.append(i)

    return np.array(keep, dtype=np.int32)

def expand_box_for_hand(box, W, H, scale=2.6, shift_y=-0.1):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    new_size = size * scale
    cy_shifted = cy + (new_size * shift_y)

    new_x1 = max(0, int(cx - new_size / 2))
    new_y1 = max(0, int(cy_shifted - new_size / 2))
    new_x2 = min(W, int(cx + new_size / 2))
    new_y2 = min(H, int(cy_shifted + new_size / 2))

    return [new_x1, new_y1, new_x2, new_y2]

def expand_face_box(box, W, H, scale=1.5):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    size = max(w, h)
    new_size = size * scale

    new_x1 = max(0, int(cx - new_size / 2))
    new_y1 = max(0, int(cy - new_size / 2))
    new_x2 = min(W, int(cx + new_size / 2))
    new_y2 = min(H, int(cy + new_size / 2))

    crop_w = new_x2 - new_x1
    crop_h = new_y2 - new_y1

    if crop_w > crop_h:
        diff = crop_w - crop_h
        new_y1 = max(0, new_y1 - diff // 2)
        new_y2 = min(H, new_y2 + diff // 2)
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        new_x1 = max(0, new_x1 - diff // 2)
        new_x2 = min(W, new_x2 + diff // 2)

    return [new_x1, new_y1, new_x2, new_y2]

def preprocess_for_detection(frame, size, norm='unit', output_format='NHWC'):
    """Preprocess frame for model inference.

    Args:
        frame: BGR input frame
        size: Target size as int (square) or tuple (w, h)
        norm: 'unit' [0,1] or 'centered' [-1,1]
        output_format: 'NHWC' or 'NCHW'

    Returns:
        Preprocessed tensor with batch dimension

    Note: SCRFD face detection uses custom preprocessing inline (not this helper).
    """
    if isinstance(size, int):
        size = (size, size)

    img = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32)  # BGR to RGB

    if norm == 'unit':
        img = img / 255.0
    elif norm == 'centered':
        img = (img / 127.5) - 1.0

    if output_format == 'NCHW':
        img = np.transpose(img, (2, 0, 1))

    return img[None, ...]

def crop_for_hand_lmk(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.resize(crop, (HAND_LMK_INP, HAND_LMK_INP),
                          interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    crop_rgb = crop_rgb.astype(np.float32) / 255.0
    return np.transpose(crop_rgb, (2, 0, 1))[None, ...]

def crop_for_facemesh(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.resize(crop, (FACE_MESH_INP, FACE_MESH_INP),
                          interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2RGB)
    crop_rgb = crop_rgb.astype(np.float32) / 255.0

    return crop_rgb[None, ...]

def draw_santa_hat(canvas, mesh, W, H, face_index=0):
    """Draw a Santa hat anchored between keypoints 103 and 332.

    Args:
        canvas: Image to draw on
        mesh: Face mesh landmarks
        W: Canvas width
        H: Canvas height
        face_index: Index of face (determines flop direction - even=right, odd=left)
    """
    if mesh is None or len(mesh) < 478:
        return

    # Get anchor points (103 = left temple, 332 = right temple)
    if 103 >= len(mesh) or 332 >= len(mesh):
        return

    left_anchor = mesh[103]
    right_anchor = mesh[332]

    # Calculate hat dimensions based on face width
    hat_base_left = (int(left_anchor[0]), int(left_anchor[1]))
    hat_base_right = (int(right_anchor[0]), int(right_anchor[1]))

    # Calculate center and width
    center_x = (hat_base_left[0] + hat_base_right[0]) // 2
    center_y = (hat_base_left[1] + hat_base_right[1]) // 2
    hat_width = int(np.sqrt((hat_base_right[0] - hat_base_left[0])**2 +
                            (hat_base_right[1] - hat_base_left[1])**2))

    # Hat proportions
    hat_initial_height = int(hat_width * 0.6)
    hat_flop_length = int(hat_width * 1.1)
    hat_thickness = int(hat_width * 0.2)
    base_extension = int(hat_width * 0.15)
    brim_height = int(hat_width * 0.15)
    pompom_radius = int(hat_width * 0.12)

    # Determine flop direction based on face index
    flop_direction = 1 if face_index % 2 == 0 else -1

    # Extend the base points
    hat_base_left_extended = (hat_base_left[0] - base_extension, hat_base_left[1])
    hat_base_right_extended = (hat_base_right[0] + base_extension, hat_base_right[1])

    # Key points for the floppy hat shape
    peak_x = center_x + int(hat_width * 0.2 * flop_direction)
    peak_y = center_y - hat_initial_height

    mid_flop_x = center_x + int(hat_flop_length * 0.5 * flop_direction)
    mid_flop_y = center_y - int(hat_initial_height * 0.4)

    tip_x = center_x + int(hat_flop_length * flop_direction)
    tip_y = center_y

    # Top edge of hat
    hat_top_edge = np.array([
        hat_base_left_extended,
        (peak_x - int(hat_thickness * 0.7), peak_y - int(hat_thickness * 0.5)),
        (mid_flop_x, mid_flop_y - int(hat_thickness * 0.4)),
        (tip_x, tip_y)
    ], dtype=np.int32)

    # Bottom edge of hat
    hat_bottom_edge = np.array([
        (tip_x, tip_y),
        (mid_flop_x, mid_flop_y + int(hat_thickness * 0.4)),
        (peak_x + int(hat_thickness * 0.7), peak_y + int(hat_thickness * 0.5)),
        hat_base_right_extended
    ], dtype=np.int32)

    # Combine edges
    hat_points = np.vstack([hat_top_edge, hat_bottom_edge])

    # Draw filled red hat body
    cv2.fillPoly(canvas, [hat_points], SANTA_HAT_RED)

    # Draw hat outline
    cv2.polylines(canvas, [hat_top_edge], False, (0, 0, 150), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [hat_bottom_edge], False, (0, 0, 150), 2, cv2.LINE_AA)

    # Draw white fluffy brim
    num_brim_puffs = 12
    for i in range(num_brim_puffs + 1):
        t = i / num_brim_puffs
        brim_x = int(hat_base_left_extended[0] * (1 - t) + hat_base_right_extended[0] * t)
        brim_y = int(hat_base_left_extended[1] * (1 - t) + hat_base_right_extended[1] * t)

        wave_offset = int(brim_height * 0.3 * np.sin(t * np.pi * 3))
        brim_y += wave_offset

        puff_radius = int(brim_height * 0.8)
        cv2.circle(canvas, (brim_x, brim_y), puff_radius, SANTA_HAT_WHITE, -1, cv2.LINE_AA)

    # Draw white fluffy pompom
    pompom_puffs = 8
    for i in range(pompom_puffs):
        angle = (i / pompom_puffs) * 2 * np.pi
        puff_offset_x = int(pompom_radius * 0.3 * np.cos(angle))
        puff_offset_y = int(pompom_radius * 0.3 * np.sin(angle))
        cv2.circle(canvas,
                  (tip_x + puff_offset_x, tip_y + puff_offset_y),
                  int(pompom_radius * 0.8),
                  SANTA_HAT_WHITE, -1, cv2.LINE_AA)

    # Central pompom
    cv2.circle(canvas, (tip_x, tip_y), pompom_radius, SANTA_HAT_WHITE, -1, cv2.LINE_AA)

    # Add highlights
    cv2.circle(canvas, (tip_x - pompom_radius//3, tip_y - pompom_radius//3),
              pompom_radius//3, (255, 255, 255), -1, cv2.LINE_AA)

def ort_gpu_preferred(path, tag, use_tensorrt=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False

    if use_tensorrt:
        print(f"[INFO] {tag}: Attempting TensorRT...")
        try:
            providers_trt = [
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                    "trt_max_workspace_size": 2147483648,
                }),
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                }),
                "CPUExecutionProvider",
            ]

            sess = ort.InferenceSession(path, sess_options=so, providers=providers_trt)
            prov = sess.get_providers()

            if "TensorrtExecutionProvider" in prov:
                print(f"[✓] {tag}: Using TensorRT (FP16)")
                return sess
            elif "CUDAExecutionProvider" in prov:
                print(f"[✓] {tag}: Using CUDA")
                return sess
        except Exception as e:
            print(f"[!] {tag}: TensorRT failed: {str(e)[:100]}")

    providers_cuda = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
        }),
        "CPUExecutionProvider",
    ]

    sess = ort.InferenceSession(path, sess_options=so, providers=providers_cuda)
    prov = sess.get_providers()

    if "CUDAExecutionProvider" in prov:
        print(f"[✓] {tag}: Using CUDA")
    else:
        print(f"[!] {tag}: Using CPU")

    return sess

def draw_landmarks(canvas, landmarks, connections, dot_color, line_color,
                   dot_size=3, line_width=2, draw_dots=True):
    """Draw landmarks and connections on canvas."""
    if landmarks is None or len(landmarks) == 0:
        return

    H, W = canvas.shape[:2]

    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_pt = tuple(landmarks[start_idx, :2].astype(int))
            end_pt = tuple(landmarks[end_idx, :2].astype(int))

            if (0 <= start_pt[0] < W and 0 <= start_pt[1] < H and
                0 <= end_pt[0] < W and 0 <= end_pt[1] < H):
                cv2.line(canvas, start_pt, end_pt, line_color, line_width)

    if draw_dots:
        for lmk in landmarks:
            x, y = int(lmk[0]), int(lmk[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), dot_size, dot_color, -1)

class FPSCounter:
    def __init__(self, window=30):
        self.window = window
        self.times = []

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.window:
            self.times.pop(0)

    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        return len(self.times) / (self.times[-1] - self.times[0])

def main():
    parser = argparse.ArgumentParser(description='Complete Holistic Tracking with GPU')
    parser.add_argument('--camera', type=int, default=0, help='Camera device')
    parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT')
    parser.add_argument('--width', type=int, default=1920, help='Camera width')
    parser.add_argument('--height', type=int, default=1080, help='Camera height')
    parser.add_argument('--display-width', type=int, default=0, help='Display width')
    parser.add_argument('--display-height', type=int, default=0, help='Display height')
    parser.add_argument('--mirror', action='store_true', help='Mirror output')
    parser.add_argument('--no-debug', action='store_true', help='Hide debug info')
    parser.add_argument('--no-pose', action='store_true', help='Disable pose')
    parser.add_argument('--no-face', action='store_true', help='Disable face')
    parser.add_argument('--no-hands', action='store_true', help='Disable hands')
    parser.add_argument('--skelly', action='store_true', help='Enable filled face skeleton mode')
    parser.add_argument('--santa-hat', action='store_true', help='Draw Santa hats on faces')
    parser.add_argument('--show-pose-roi', action='store_true', help='Show body detection bounding box (debug)')
    args = parser.parse_args()

    global DRAW_POSE, DRAW_FACE_MESH, DRAW_HANDS, DRAW_DEBUG_INFO, DRAW_POSE_ROI, SKELLY_MODE, DRAW_SANTA_HAT
    if args.no_pose:
        DRAW_POSE = False
    if args.no_face:
        DRAW_FACE_MESH = False
    if args.no_hands:
        DRAW_HANDS = False
    if args.no_debug:
        DRAW_DEBUG_INFO = False
    if args.show_pose_roi:
        DRAW_POSE_ROI = True
    if args.skelly:
        SKELLY_MODE = True
    if args.santa_hat:
        DRAW_SANTA_HAT = True

    print("="*70)
    print("Complete Holistic Tracking - GPU Accelerated")
    print("Body (33) + FaceMesh (478) + Hands (21×2)")
    if SKELLY_MODE:
        print("Mode: SKELLY (Filled Face)")
    if DRAW_SANTA_HAT:
        print("Feature: Santa Hats Enabled! 🎅")
    print("="*70)
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")
    print("="*70)

    if args.tensorrt:
        os.makedirs("./trt_cache", exist_ok=True)

    print("\n[INIT] Loading models...")
    pose_det_sess = ort_gpu_preferred(POSE_DET_ONNX, "PoseDet", args.tensorrt)
    pose_lmk_sess = ort_gpu_preferred(POSE_LMK_ONNX, "PoseLmk", args.tensorrt)
    face_det_sess = ort_gpu_preferred(FACE_DET_ONNX, "FaceDet", args.tensorrt)
    face_mesh_sess = ort_gpu_preferred(FACE_MESH_ONNX, "FaceMesh", args.tensorrt)
    hand_det_sess = ort_gpu_preferred(HAND_DET_ONNX, "HandDet", args.tensorrt)
    hand_lmk_sess = ort_gpu_preferred(HAND_LMK_ONNX, "HandLmk", args.tensorrt)

    print("\n[INIT] Generating anchors...")
    pose_anchors = generate_blazepose_anchors()
    hand_anchors = generate_blazepalm_anchors(HAND_DET_INP)

    print(f"\n[INIT] Starting camera {args.camera}...")
    try:
        cam = ThreadedCamera(args.camera, args.width, args.height).start()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        return

    time.sleep(1.0)

    fps_counter = FPSCounter()
    canvas = None

    print("\n" + "="*70)
    print("Press 'q' or ESC to quit\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        H, W = frame.shape[:2]
        if canvas is None or canvas.shape[:2] != (H, W):
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            canvas.fill(0)

        t_start = time.time()

        # ===== FACE DETECTION (run first) =====
        faces = np.array([])
        if DRAW_FACE_MESH:
            # SCRFD preprocessing - keep original inline for correctness
            img_face = cv2.resize(frame, (FACE_DET_INP, FACE_DET_INP), interpolation=cv2.INTER_LINEAR)
            img_face = img_face.astype(np.float32)
            img_face = (img_face - 127.5) / 128.0
            img_face = np.transpose(img_face[:, :, ::-1], (2, 0, 1))[None, ...]

            face_outs = face_det_sess.run(None, {face_det_sess.get_inputs()[0].name: img_face})

            num_strides = 3
            scores_list = [face_outs[i] for i in range(num_strides)]
            bboxes_list = [face_outs[i + num_strides] for i in range(num_strides)]

            faces = decode_scrfd_detections(
                scores_list, bboxes_list,
                frame.shape, FACE_DET_INP, FACE_SCORE_THR
            )

            if len(faces) > 0:
                keep = nms_boxes(faces[:, :4], faces[:, 4], FACE_IOU_THR)
                faces = faces[keep]
                print(f"[FACE DET] Detected {len(faces)} face(s)")

        # ===== BODY POSE DETECTION =====
        all_pose_lmks = []
        pose_lmks = None
        if DRAW_POSE:
            # Stage 1: Detect person and get ROI
            img_det = preprocess_for_detection(frame, POSE_DET_INP, norm='centered', output_format='NHWC')

            pose_det_outs = pose_det_sess.run(None, {pose_det_sess.get_inputs()[0].name: img_det})
            pose_detections = decode_pose_detections(pose_det_outs, pose_anchors, POSE_CONF_THR)
            pose_detections = nms_pose_detections(pose_detections, iou_thresh=POSE_IOU_THR)

            for detection in pose_detections:
                # Get ROI from detection
                x1, y1, x2, y2 = get_pose_roi(detection, W, H)

                # Debug: Draw detection ROI on canvas (cyan rectangle)
                if DRAW_POSE_ROI:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Crop ROI and run landmark model
                roi_crop = frame[y1:y2, x1:x2]
                if roi_crop.size == 0:
                    continue

                img_pose = cv2.resize(roi_crop, (POSE_LMK_INP, POSE_LMK_INP), interpolation=cv2.INTER_LINEAR)
                img_pose = cv2.cvtColor(img_pose, cv2.COLOR_BGR2RGB)
                img_pose = img_pose.astype(np.float32) / 255.0
                img_pose = img_pose[None, ...]

                pose_outs = pose_lmk_sess.run(None, {pose_lmk_sess.get_inputs()[0].name: img_pose})
                ld_3d = pose_outs[0]
                poseflag = pose_outs[1]

                if poseflag[0, 0] >= POSE_CONF_THR:
                    landmarks_flat = ld_3d[0]
                    try:
                        landmarks = landmarks_flat.reshape(39, 5)[:33]

                        pose_lmks = landmarks.copy()
                        # Transform from 256x256 ROI space back to original frame
                        roi_w = x2 - x1
                        roi_h = y2 - y1
                        pose_lmks[:, 0] = (landmarks[:, 0] / POSE_LMK_INP) * roi_w + x1
                        pose_lmks[:, 1] = (landmarks[:, 1] / POSE_LMK_INP) * roi_h + y1
                        all_pose_lmks.append(pose_lmks)
                    except (ValueError, IndexError):
                        pass

            # For backward compatibility, set pose_lmks to first person if any detected
            pose_lmks = all_pose_lmks[0] if len(all_pose_lmks) > 0 else None

        # ===== FACEMESH =====
        face_meshes = []
        if len(faces) > 0:
            for face in faces:
                x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_box = [x1, y1, x2, y2]
                face_expanded = expand_face_box(face_box, W, H, scale=1.5)

                face_crop = crop_for_facemesh(frame, face_expanded)

                if face_crop is not None:
                    mesh_outs = face_mesh_sess.run(None,
                        {face_mesh_sess.get_inputs()[0].name: face_crop})

                    landmarks_flat = mesh_outs[0].reshape(-1, 3)[:478]
                    confidence = mesh_outs[1][0, 0, 0, 0]

                    if confidence < 0.5:
                        continue

                    landmarks_normalized = landmarks_flat.copy()
                    landmarks_normalized[:, 0] = landmarks_flat[:, 0] / 256.0
                    landmarks_normalized[:, 1] = landmarks_flat[:, 1] / 256.0

                    box_w = face_expanded[2] - face_expanded[0]
                    box_h = face_expanded[3] - face_expanded[1]

                    mesh_px = landmarks_normalized.copy()
                    mesh_px[:, 0] = landmarks_normalized[:, 0] * box_w + face_expanded[0]
                    mesh_px[:, 1] = landmarks_normalized[:, 1] * box_h + face_expanded[1]

                    # DEBUG: Print face position only if body detected
                    if pose_lmks is not None:
                        print(f"[FACE DEBUG] Face box: ({x1}, {y1}) to ({x2}, {y2})")
                        print(f"[FACE DEBUG] Expanded box: ({face_expanded[0]}, {face_expanded[1]}) to ({face_expanded[2]}, {face_expanded[3]})")
                        print(f"[FACE DEBUG] Nose tip (landmark 1): x={mesh_px[1, 0]:.1f}, y={mesh_px[1, 1]:.1f}")
                        print(f"[FACE DEBUG] Chin (landmark 152): x={mesh_px[152, 0]:.1f}, y={mesh_px[152, 1]:.1f}")

                    face_meshes.append(mesh_px)

        # ===== HAND DETECTION =====
        hand_boxes_xyxy = np.array([])
        if DRAW_HANDS:
            inp_hand = preprocess_for_detection(frame, HAND_DET_INP, norm='unit', output_format='NCHW')

            hand_outs = hand_det_sess.run(None, {hand_det_sess.get_inputs()[0].name: inp_hand})
            boxes_raw, scores = decode_palm_with_anchors(hand_outs, hand_anchors)

            mask = scores >= HAND_SCORE_THR
            boxes_raw = boxes_raw[mask]
            scores = scores[mask]

            if len(scores) > 0:
                boxes_xyxy = boxes_to_xyxy(boxes_raw, W, H)
                keep = nms_boxes(boxes_xyxy, scores, HAND_IOU_THR)
                hand_boxes_xyxy = boxes_xyxy[keep]

                if len(faces) > 0:
                    hand_keep = suppress_overlapping_detections(hand_boxes_xyxy, faces[:, :4], iou_threshold=0.2)
                    hand_boxes_xyxy = hand_boxes_xyxy[hand_keep]

            # ===== HAND LANDMARKS =====
            for b in hand_boxes_xyxy:
                b_expanded = expand_box_for_hand(b, W, H, scale=2.6, shift_y=-0.1)

                crop = crop_for_hand_lmk(frame, b_expanded)
                if crop is not None:
                    lmk_outs = hand_lmk_sess.run(None,
                        {hand_lmk_sess.get_inputs()[0].name: crop})
                    lmk_out = lmk_outs[0]

                    if len(lmk_out.shape) == 3:
                        lmk_out = lmk_out.reshape(lmk_out.shape[1], lmk_out.shape[2])
                    elif len(lmk_out.shape) == 2 and lmk_out.shape[0] == 1:
                        lmk_out = lmk_out.reshape(-1)

                    if len(lmk_out.shape) == 1:
                        if lmk_out.shape[0] == 63:
                            lmk_out = lmk_out.reshape(21, 3)[:, :2]
                        elif lmk_out.shape[0] == 42:
                            lmk_out = lmk_out.reshape(21, 2)
                        else:
                            lmk_out = lmk_out.reshape(-1, 2)
                    elif lmk_out.shape[-1] == 3:
                        lmk_out = lmk_out[:, :2]

                    pts = lmk_out
                    pts_normalized = pts / HAND_LMK_INP

                    box_w = b_expanded[2] - b_expanded[0]
                    box_h = b_expanded[3] - b_expanded[1]
                    pts_pixel = pts_normalized.copy()
                    pts_pixel[:, 0] = pts_normalized[:, 0] * box_w + b_expanded[0]
                    pts_pixel[:, 1] = pts_normalized[:, 1] * box_h + b_expanded[1]

                    # Draw hand landmarks
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(pts_pixel) and end_idx < len(pts_pixel):
                            start_pt = tuple(pts_pixel[start_idx].astype(int))
                            end_pt = tuple(pts_pixel[end_idx].astype(int))

                            if (0 <= start_pt[0] < W and 0 <= start_pt[1] < H and
                                0 <= end_pt[0] < W and 0 <= end_pt[1] < H):
                                cv2.line(canvas, start_pt, end_pt, HAND_LINE_COLOR, HAND_LINE_WIDTH)

                    for i, (x, y) in enumerate(pts_pixel.astype(int)):
                        if 0 <= x < W and 0 <= y < H:
                            if i == 0:
                                cv2.circle(canvas, (x, y), HAND_DOT_SIZE + 2, HAND_WRIST_COLOR, -1)
                            elif i in [4, 8, 12, 16, 20]:
                                cv2.circle(canvas, (x, y), HAND_DOT_SIZE + 1, HAND_FINGERTIP_COLOR, -1)
                            else:
                                cv2.circle(canvas, (x, y), HAND_DOT_SIZE, HAND_DOT_COLOR, -1)

        # ===== DRAW POSE (all detected people) =====
        if DRAW_POSE and len(all_pose_lmks) > 0:
            for person_lmks in all_pose_lmks:
                draw_landmarks(canvas, person_lmks, POSE_CONNECTIONS,
                             POSE_DOT_COLOR, POSE_LINE_COLOR,
                             POSE_DOT_SIZE, POSE_LINE_WIDTH, draw_dots=False)

                # Draw upper body keypoints only (11-22)
                for i in range(11, 23):
                    if i < len(person_lmks):
                        x, y = int(person_lmks[i, 0]), int(person_lmks[i, 1])
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(canvas, (x, y), POSE_DOT_SIZE, POSE_DOT_COLOR, -1)

        # ===== DRAW FACE MESH =====
        if DRAW_FACE_MESH and len(face_meshes) > 0:
            for mesh in face_meshes:
                if SKELLY_MODE:
                    # Filled face mode with cutouts for features
                    face_mask = np.zeros((H, W), dtype=np.uint8)

                    # Fill face contour
                    face_pts = np.array([[int(mesh[i, 0]), int(mesh[i, 1])]
                                        for i in FACE_CONTOUR_INDICES
                                        if i < len(mesh) and
                                        0 <= int(mesh[i, 0]) < W and
                                        0 <= int(mesh[i, 1]) < H], dtype=np.int32)
                    if len(face_pts) > 3:
                        cv2.fillPoly(face_mask, [face_pts], 255)

                    # Cut out features (eyes, nose, mouth)
                    for feature_indices in [LEFT_EYE, RIGHT_EYE, NOSE, MOUTH_OUTER]:
                        feature_pts = np.array([[int(mesh[i, 0]), int(mesh[i, 1])]
                                               for i in feature_indices
                                               if i < len(mesh) and
                                               0 <= int(mesh[i, 0]) < W and
                                               0 <= int(mesh[i, 1]) < H], dtype=np.int32)
                        if len(feature_pts) > 3:
                            cv2.fillPoly(face_mask, [feature_pts], 0)

                    # Apply filled face to canvas
                    canvas[face_mask > 0] = FACE_MESH_COLOR
                else:
                    # Original contour drawing mode
                    draw_landmarks(canvas, mesh, FACE_MESH_CONTOUR,
                                 FACE_MESH_COLOR, FACE_MESH_COLOR,
                                 FACE_MESH_DOT_SIZE, FACE_MESH_LINE_WIDTH, draw_dots=True)

        # ===== DRAW SANTA HATS =====
        if DRAW_SANTA_HAT and len(face_meshes) > 0:
            for face_idx, mesh in enumerate(face_meshes):
                draw_santa_hat(canvas, mesh, W, H, face_idx)

        inference_time = time.time() - t_start

        # ===== DISPLAY INFO =====
        fps_counter.update()
        fps = fps_counter.get_fps()

        if args.mirror:
            canvas = cv2.flip(canvas, 1)

        if DRAW_DEBUG_INFO:
            detections = []
            if len(all_pose_lmks) > 0:
                detections.append(f"Body({len(all_pose_lmks)})")
            if len(face_meshes) > 0:
                detections.append(f"Face({len(face_meshes)})")
            if len(hand_boxes_xyxy) > 0:
                detections.append(f"Hands({len(hand_boxes_xyxy)})")
            if DRAW_SANTA_HAT and len(face_meshes) > 0:
                detections.append("🎅")

            status = " | ".join(detections) if detections else "No detection"

            cv2.putText(canvas,
                       f"FPS: {fps:.1f} | Time: {inference_time*1000:.0f}ms | {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        if args.display_width > 0 and args.display_height > 0:
            display_canvas = cv2.resize(canvas, (args.display_width, args.display_height))
        else:
            display_canvas = canvas

        cv2.imshow("Complete Holistic Tracking", display_canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
