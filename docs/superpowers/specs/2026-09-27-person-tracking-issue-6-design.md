# Person tracking: stable person IDs (issue #6)

## Problem

`person_id` was the body box's index in the detector output, which is ordered by score. Scores shift every frame, so two people swapped colours (`PERSON_COLORS[person_id % 8]`) whenever their scores swapped.

## Design

`src/tracker.py`, `PersonTracker.update(body_boxes) -> [track_id per box]`, called once per frame after `_drop_nested_bodies`. The track ID replaces the index as `person_id` for the body, its face and its hands; the renderer is unchanged.

- **Prediction.** Each track keeps its last box and a smoothed centre velocity (`velocity_smoothing=0.5`). Boxes are matched against the predicted box, so two people walking past each other keep their IDs after they separate.
- **Matching.** Greedy, best pair first. A pair matches on IoU ≥ 0.2 with the predicted box (score 1 + IoU), or, when boxes don't overlap enough (fast movement), on centre distance ≤ 0.6 × the track box's diagonal (score < 1). Any overlap match beats a distance-only match. With at most a handful of people, greedy is as good as Hungarian matching and needs no scipy.
- **Misses.** An unmatched track survives `max_missed=15` frames (0.5 s at 30 FPS), with its prediction carried forward, so a one-frame detector miss or a short occlusion keeps the ID.
- **New IDs.** Unmatched boxes start new tracks. IDs are never reused, so someone who leaves and comes back after the track expired gets a new ID and colour; there is no re-identification.
- **History.** Each track keeps its last 300 box centres (10 s) and its age, the input for venue calibration (#7): a prop's track never moves.
- **Cost.** O(tracks × boxes) arithmetic per frame, well under 0.1 ms; no effect on the Orin's ~30 FPS.
- **Debug dumps.** `--dump-detections` JSON has `track_id` on each body entry, and the overlay labels body boxes `#<id>`.

## After the first Orin run

With two people the Orin logged ~4.8 body boxes and ~2 heads per frame, and colours still changed every frame. Two causes, both fixed:

- **Unconfirmed tracks were drawn.** A flickering false positive got a skeleton and a fresh ID (colour) each time it reappeared. Tracks are now `confirmed` after `min_hits=3` matched frames; only confirmed tracks run the face/hand/pose models and are drawn. Fewer pose runs per frame, too.
- **A head (or hand) was drawn once per body box containing it**, so a false-positive box around a real person drew a second face mesh in another colour, with the top one changing by score. Bodies now get heads and hands exclusively (see below for how heads are matched).

`[STAGES]` now logs `people=` (confirmed tracks per frame) next to `bodies=` (raw boxes).

## Body false positives: require a face once per track

Second Orin run: colours stable, but many body false positives, not helped by more light. Per-frame detector and pose scores can't separate the lamp from a seated person (issue #3). The face model's face score can: real faces score high, the ball, lampshade, backs of heads and clutter low; that check already stops face meshes on the ball.

With tracks, it now gates bodies too: a confirmed track is drawn (and runs the hand and pose models) only after the face model has accepted a face in its head box at least once (`Track.face_hits`). After that the person stays drawn while tracked, even when they turn away. Cost: someone who enters facing away is drawn from the first frame they face the camera. `--no-require-face` turns it off; with `--no-face` it is off. `[STAGES]` logs `drawn=` beside `people=`.

### Third Orin run: three people, two facing away

`bodies=9.7 people=9.7 drawn=7-8` and rising, `heads=6-8`, FPS 13. False positives were passing the face check:

- **Head went to the wrong body.** With ~10 body boxes, a person's head sat inside several false-positive boxes, and the oldest (a static false positive) claimed it; the person's real face then verified that box. Heads are now matched by fit (`_assign_heads`): the head centre must be inside the body box horizontally and in its top 45%, the head at least 0.15 of the body's width, and the best fit (closest to the box's top-centre) wins, one head per body.
- **One weak accept was enough**, and over thousands of frames a borderline non-face eventually passes. A track now needs `FACE_VERIFY_HITS=2` frames with presence ≥ `FACE_VERIFY_PRESENCE=0.993` (logit +5; real frontal faces measured +8 to +27, non-faces −1 to −23). Faces are still drawn from 0.5.

People facing away are, by design, not drawn until they face the camera once. Showing them without a face needs another signal, e.g. movement (#7).

## Tests

`test_tracker.py`: detector order swapping, one missed frame, track expiry, leaving and re-entering (new ID, doesn't take another person's), two people crossing with one hidden at the crossing (fails without the velocity term), fast movement without overlap, a far box is a new person, history.

Also run in the cloud session with a fake camera/detector through `poser.main()`: the IDs stayed fixed while the detector's box order flipped every frame.

## To check on the Orin

`tools/orin_bench/run_poser.sh <name> --dump-detections /tmp/ppdump --dump-every 1` with two people moving around and crossing; `track_id`s in the JSON should stay fixed, and FPS should be unchanged.

## Open for later

- If people standing close together swap IDs on the Orin, add appearance (e.g. a colour histogram of the box) to the match score.
- The lamp from #3 will get a track like anyone else; #7 uses the history to ignore it.
