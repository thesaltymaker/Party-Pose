# False positives on the Orin (issue #3)

Status: merged to `main`. Faces are no longer drawn on non-faces, and a detector box next to a person no longer gets its own skeleton. A static object on its own (the living-room lamp) can still get a skeleton; that moves to venue calibration (#7), which builds on person tracking (#6).

## Problem

Party-Pose on the Orin drew faces and skeletons where there was no person: a face mesh on a black exercise ball, a skeleton on a table lamp.

## Method

1. `--dump-detections DIR --dump-every SECONDS` (in `poser.py`) saves, per dump: the frame with the overlay plus detector boxes and scores drawn (`frame_NNNN.jpg`), the frame before drawing (`frame_NNNN_raw.jpg`), and a JSON of every detector box with its score and the presence score of each face, hand and body result.
2. Runs on the Orin with `tools/orin_bench/run_poser.sh <name> --dump-detections /tmp/ppdump --dump-every 1`.
3. Offline replay: crop each dumped box from the raw frame the same way the processors do, run the landmark model on the laptop (CPU is fine), and compare the model's scores for real people against the false detections. Crops were labelled by eye on contact sheets.

## Findings

Living room, daylight, 121 frames:

| False detection | Frames | Detector score (min / median / max) |
|---|---|---|
| Black exercise ball as a head, face mesh drawn | 108 | 0.30 / 0.56 / 0.85 |
| Table lamp as a body, skeleton drawn | 40 | 0.30 / 0.40 / 0.74 |
| Lampshade as a head | 36 | 0.30 / 0.42 / 0.64 |

Detector scores for these overlap real people, so raising the person detector's threshold (0.3) does not fix them.

| # | Cause | Evidence | Action |
|---|---|---|---|
| 1 | `_find_head_for_body` fell back to the best head anywhere in the frame when no head was inside the body box. | The ball got a face mesh from a body far away. | Only a head whose centre is inside the body box is used (`d388bc5`). Cut faces run from 105 to 84 on the 121 frames. |
| 2 | The face landmark model's own face score (`Identity_1`, a logit) was ignored. | Clean test face: +17; wall: −28. On 143 labelled head crops from the Orin, with the old 25% padding real faces scored 0 to +14 (some negative); with no padding, frontal faces +8 to +27, and the ball, fists, arms, backs of heads and dark clutter −1 to −23. | Crop the head box without padding; skip the face when sigmoid(score) < 0.5 (`e5c33b3`). Live: no mesh on the ball, 52 faces drawn in 2 min, all real. Dim, blurred or side-on faces can drop out for a frame. |
| 3 | The detector keeps a small body box beside a person when its IoU with the person's box is low. | Lamp beside a person; a second box on a raised arm. | Drop a body box that lies ≥50% inside a higher-scoring body box (`845f127`). On 273 body boxes from three runs it removed exactly those two, no real bodies. |

FPS is unchanged (about 30 with one person, all models on TensorRT).

## Not fixed: a static object on its own

Per-frame scores cannot separate the lamp from a person sitting on the couch:

| Signal | Lamp | Person seated on couch |
|---|---|---|
| Detector body score | 0.31–0.53 | 0.33–0.49 |
| Pose model person score (`Identity_1`) | 0.55–0.88 | 0.71–0.85 |
| Shoulder / hip visibility | 0.87–0.99 | same |
| Person mask coverage (`Identity_2`) | 21–30% | 24–49% |

Any threshold that removes the lamp also removes the seated person. The lamp never moves, so the plan is a per-venue calibration run that records static detections (#7), using person tracks (#6).

## Also seen

- A man seated on the couch in daylight was not detected at all (missed detection, not a false positive). Not addressed.
- In bright sun the ball and lamp were not detected; the false positives depend on lighting.
