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
- **A head (or hand) was drawn once per body box containing it**, so a false-positive box around a real person drew a second face mesh in another colour, with the top one changing by score. Bodies now claim heads and hands exclusively, oldest track first (`_claim_parts`).

`[STAGES]` now logs `people=` (confirmed tracks per frame) next to `bodies=` (raw boxes).

## Tests

`test_tracker.py`: detector order swapping, one missed frame, track expiry, leaving and re-entering (new ID, doesn't take another person's), two people crossing with one hidden at the crossing (fails without the velocity term), fast movement without overlap, a far box is a new person, history.

Also run in the cloud session with a fake camera/detector through `poser.main()`: the IDs stayed fixed while the detector's box order flipped every frame.

## To check on the Orin

`tools/orin_bench/run_poser.sh <name> --dump-detections /tmp/ppdump --dump-every 1` with two people moving around and crossing; `track_id`s in the JSON should stay fixed, and FPS should be unchanged.

## Open for later

- If people standing close together swap IDs on the Orin, add appearance (e.g. a colour histogram of the box) to the match score.
- The lamp from #3 will get a track like anyone else; #7 uses the history to ignore it.
