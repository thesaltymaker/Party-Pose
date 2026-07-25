"""Standalone test: YOLOX body/head/hand detection on camera 2.

Draws colored boxes:
  Green  = body
  Yellow = head
  Blue   = hand

Press 'q' to quit, 'd' to toggle debug overlay, +/- to raise/lower threshold.
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL  = Path(__file__).parent / 'models' / 'yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.onnx'
CAMERA = 2
INPUT_W, INPUT_H = 320, 256

COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 0, 0)}
LABELS = {0: 'body', 1: 'head', 2: 'hand'}

sess = ort.InferenceSession(str(MODEL), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f"Providers in use: {sess.get_providers()}")

cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA}")

conf    = 0.1   # start very low so we can see what's there
debug   = True
frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_n += 1
    h, w = frame.shape[:2]

    # Letterbox to 320x256
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x  = (INPUT_W - nw) // 2
    pad_y  = (INPUT_H - nh) // 2

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh))
    padded  = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    arr = padded.astype(np.float32)[np.newaxis]

    raw = sess.run(['batchno_classid_score_x1y1x2y2'], {'input': arr})[0]
    # raw: [60, 7] — [batch, class_id, score, x1, y1, x2, y2]

    # Every 30 frames, print top-5 scores to console
    if frame_n % 30 == 1:
        scores = raw[:, 2]
        top    = np.sort(scores)[::-1][:5]
        print(f"frame {frame_n}  top-5 scores: {top.round(3).tolist()}  "
              f"  above 0.1: {(scores > 0.1).sum()}  above 0.3: {(scores > 0.3).sum()}")

    for row in raw:
        _, class_id, score, x1, y1, x2, y2 = row
        if score < conf:
            continue
        fx1 = int(max(0, (x1 - pad_x) / scale))
        fy1 = int(max(0, (y1 - pad_y) / scale))
        fx2 = int(min(w,  (x2 - pad_x) / scale))
        fy2 = int(min(h,  (y2 - pad_y) / scale))
        cid   = int(class_id)
        color = COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(frame, f"{LABELS.get(cid,'?')} {score:.2f}",
                    (fx1, max(fy1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if debug:
        cv2.putText(frame, f"conf={conf:.2f}  +/- to adjust", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('YOLOX detection test', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        conf = min(conf + 0.05, 0.95)
        print(f"conf threshold -> {conf:.2f}")
    elif key == ord('-'):
        conf = max(conf - 0.05, 0.01)
        print(f"conf threshold -> {conf:.2f}")
    elif key == ord('d'):
        debug = not debug

cap.release()
cv2.destroyAllWindows()
