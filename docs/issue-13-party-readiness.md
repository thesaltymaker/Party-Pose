# Issue #13: ready for the party (findings and conclusions, 2026-10-11 to 2026-10-18)

Goal for the Halloween party (2026-10-31): 3-4 dancers on the Jetson Orin with no missing faces or hands, smooth
figures, no flashing, and at least 30 FPS live (the camera's rate). Replays with `--pipeline` should reach 33 FPS with
3-4 people as a margin.

How it was measured: recorded Orin clips replayed with `tools/orin_bench/clip_bench.sh` (`--video`), GPU clocks
pinned, one benchmark at a time (`flock /tmp/orin_bench.lock`), A/B runs interleaved, a run counted only when the
capture stage stayed below ~5 ms. Quality from the app's own counters: people drawn per frame vs the true count,
`[FLICKER]` blinks (gaps of 1-5 frames per part), `[MOTION]` pops/jitter/lag (`--motion-stats`, quality runs only),
`[NEW-ID]` lines. Accuracy sweeps ran on half-size copies of the clips on the laptop.

| Clip | People | Notes |
|---|---|---|
| three_dancers | 3 | evening light, the owner big and close, a lamp at the right edge that must not be drawn |
| three_people | 4 | includes a short child |
| two_people | 2 | daylight |
| face_walk, hands_body | 1 | |

## Result: main vs the party settings (Orin, `--pipeline`, 2 rounds each)

| Clip | main FPS | main people drawn | new FPS | new people drawn | new IDs main -> new |
|---|---|---|---|---|---|
| three_dancers | 46.5 / 46.7 | 0.76 of 3 | 33.87 / 33.94 | 2.90 of 3 | 6 -> 6 |
| three_people | 33.47 / 33.05 | 1.70 of 4 | 30.84 / 30.86 | 3.81 of 4 | 7 -> 4 |
| two_people | 32.77 / 33.14 | 1.83 of 2 | 42.19 (r1 invalid: capture 64 ms) | 1.93 of 2 | 5 -> 5 |
| face_walk | 51.26 / 50.54 | 0.44 of 1 | 66.14 / 66.49 | 0.99 of 1 | 2 -> 1 |
| hands_body | 45.40 / 45.60 | 0.58 of 1 | 59.55 / 59.52 | 1.00 of 1 | 1 -> 2 |

main replays the group clips fast because it misses most people. With the party settings (`--hand-every 2
--face-every 2`), [FLICKER] on three_dancers: body 3 blinks, face 6, hands 24/21; on three_people: body 0, face 6,
hands 32/28. The lamp is not drawn (people drawn 2.90 of 3).

**Conclusion:** 3 dancers replay at ~34 FPS, 4 people at ~31: live (30 FPS camera) that is 30 with 3 people and
just about 30 with 4. With `--hand-every 3 --face-every 3`: three_dancers 35.25 / 35.46, three_people 32.65 /
32.42, the same blinks, but the raw hand points step more between model runs (see Smooth figures). Choose between 2
and 3 by eye in the live test.

## Finding the people: the person detector

- The app fed the YOLOX detector RGB; it was trained on BGR (`--detector-bgr`).
- The YOLOX tiny 288x480 model with the frame shrunk by pixel averaging (`--detector-area`) found the big, close
  person and the blurred dancers: three_dancers 2.03 -> 2.94 people drawn of 3, the close person 100% of frames, the
  lamp in 0 frames. Do not use `--detector-area` with the nano model: it draws the lamp.
- Following people with their own pose (MediaPipe's method) keeps a found person drawn when the detector misses
  them; the detector then only finds new or lost people.

## Faces and hands

- Face crop from the pose's face points instead of the head box: a face in 94% of body frames (head box 70-77%).
- `--face-keep -8` keeps a face found last frame down to logit -8 while the head is not turned away: face blinks
  86 -> 36 (three_people), 25 -> 9 (three_dancers), FPS the same.
- Hands: a palm detector to find lost hands added 0-2 points (branch ws-hands, not merged). The limit is the hand
  model on dim, blurred hands; misses score below 0.1.

## Smooth figures (sweeps on all 5 clips; Orin check of the chosen values)

Body pops per 100 frame pairs, base -> chosen (`--score-smooth 0.05 --ghost-frames 8 --smooth-body 0.3,0.5
--smooth-face 0.3,0.5 --smooth-hand 0.3,0.5`), Orin full clips:

| Clip | body pops/100 | body jitter | body lag (frames) | new IDs |
|---|---|---|---|---|
| two_people | 54.3 -> 7.3 | 7.9 -> 1.1 | 0.9 -> 0.8 | 5 -> 5 |
| three_dancers | 25.2 -> 9.3 | 2.1 -> 1.5 | 0.6 -> 0.8 | 6 -> 6 |
| three_people | (base run invalid) -> 7.1 | -> 0.9 | -> 0.8 | 4 -> 4 |
| face_walk | 29.1 -> 4.5 | 0.8 -> 0.6 | 0.5 -> 0.7 | 1 -> 1 |
| hands_body | 36.7 -> 0.5 | 0.6 -> 0.5 | 0.5 -> 0.8 | 2 -> 2 |

- `--score-smooth`: lower is better; 0.05 was at least as good as 0.1-0.3 on all clips (two_people laptop: 9.7 vs
  11.1 / 14.4 / 17.3), jitter and lag unchanged.
- One Euro beta 0.5 instead of 1.5: less jitter (face 1.0 -> 0.7, hand 1.5 -> 1.1 on the laptop), 0.1-0.3 frames
  more lag. Beta 5 made pops and jitter worse.
- `--ghost-frames 8` (from 5): two_people blinks body/face/hand 11/28/120 -> 6/12/78 with people drawn within 0.04
  of base; three_dancers face blinks 21 -> 4.
- Not used: `--hold-joints` (helps alone, adds nothing with score smoothing), `--ghost-follow-wrist` (hand pops 1.2 ->
  5.0).
- Hands/faces every 1 / 2 / 3 frames, three_dancers (`--motion-stats`): drawn face jitter 0.9 / 0.9 / 1.1, hand
  1.5 / 1.2 / 0.8, lag 0.7-0.9 frames throughout; raw (before smoothing) hand jitter 4.2 / 6.1 / 6.2, face 3.6 /
  4.8 / 6.2. The smoothing hides most of the step when a skipped part's model runs again.

## Speed (Orin)

Steps in the order they were added, three_dancers / three_people replay FPS:

| Step | three_dancers | three_people |
|---|---|---|
| Start (new detector, `--face-keep -8`) | 23.7 | 20.0 |
| Batching `--pose-batch 4 --hand-batch 4 --face-batch 4` | 26.0 | 22.5 |
| `--detect-every 3` | 26.9 | 23.7 |
| Load every session at startup | 30.2 | 26.1 |
| Smooth settings, `--hand-every 2 --face-every 2` | 32.1 | 28.6 |
| Ghost face moved when read, `is_still()` every 10th box | 34.0 | 30.8 |

Part of the startup-loading step is measurement: the replay used to count 1-3 s of session loading. Live, the same
loading froze the picture the first time 3 or 4 people were in view.

What was tried and left off:

| Option | Gain | Why left off |
|---|---|---|
| `--detect-ahead` (detector of frame N+1 in its own thread) | +1.0 FPS with a copy per frame, +1.8 with a ring of 6 frames; +1.4 / +0.7 on top of batching | the GPU is shared: detector 8.6 -> 16 ms, other models 25-35% slower; not combined with `--detect-every` |
| Lite pose model (`--pose-model`, MediaPipe pose_landmarker_lite) | +0.4-0.6 FPS | 13-19% fewer hands drawn, body pops 8.8 -> 15.9, body blinks 4 -> 11 |
| `--switch-ms 1` | none (within 0.3 FPS) | |
| OpenCV crops on their own CUDA stream | 0.72 vs 0.66 ms per 4 crops (slower) | |
| `--hand-every 3 --face-every 3` | +1.3-1.7 FPS over every 2nd | raw hand points step more; kept as an option |

Where the time is now (three_dancers, party settings, model thread ~29 ms): body ~8.2 ms, hands ~6.2, face ~6.3,
person detector ~4.2, capture ~2.1, smoothing ~1.3; drawing ~9.7 ms in the main thread. py-spy: 45% of the model
thread inside `run_with_iobinding`; most of the rest is waiting for Python's lock (three Python threads).

## Design decisions

- Every new behaviour is a flag, off by default; the service file holds the chosen values. Tests and older
  benchmark commands keep their meaning, and each option can be switched off at the party if it misbehaves.
- Decide from last frame's state where it keeps the work in order (`--detect-every` decides from last frame's
  following, so the detector stays the first model; a person lost this frame gets the detector one frame later).
- Skip work per person in turns (`skip_turn`) so each frame has about the same number of model runs: an even frame
  time matters more on screen than the average.
- Load everything at startup, never during the show.

## Left to do

- Live test with 3-4 people (full screen, `--test-label` with a seconds counter, recorded with
  `tools/orin_bench/record_video.py`), every 2nd vs every 3rd frame.
- Merge to main; update the Orin checkout and `~/.config/party-pose/party-pose.env` (the old `--face-hold` no
  longer exists).
