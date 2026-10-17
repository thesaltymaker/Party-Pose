"""--pose-batch / --hand-batch: several people's pose crops (or hands) in one model run must give the same results
as one run per crop. Model stand-ins; test_gpu_model.py checks the real models on the GPU."""
import numpy as np
import pytest

from src.body_processor import BodyProcessor
from src.hand_processor import HandProcessor
from src.preprocessor import Preprocessor
from src.types import BoundingBox, Landmark


class _Session:
    """Stand-in for the pose or hand model: every output value is the crop's fill value (its id) plus a fixed
    pattern, so results show which crop they came from; presence is high for even ids, low for odd ones."""
    def __init__(self, kind):
        self.kind, self.batches = kind, []

    def get_providers(self):
        return ['CPUExecutionProvider']  # GpuModel / GpuBatch fall back to the numpy path

    def run(self, _names, feeds):
        x = feeds['input_1']
        self.batches.append(len(x))
        ids = x[:, 0, 0, 0].astype(np.float32)
        n = 195 if self.kind == 'pose' else 63
        lms = (np.arange(n, dtype=np.float32)[None] % 50) + ids[:, None]
        presence = np.where(ids.astype(int) % 2 == 0, 5.0, -5.0).astype(np.float32)[:, None]
        if self.kind == 'pose':
            return [lms, 1 / (1 + np.exp(-presence))]
        return [lms, 1 / (1 + np.exp(-presence)), np.full((len(x), 1), 0.9, np.float32)]


class _Models:
    def __init__(self, kind):
        self.single, self.batch = _Session(kind), _Session(kind)

    def get_session(self, _name):
        return self.single

    def get_batch_session(self, _name, _slots):
        return self.batch


@pytest.fixture(autouse=True)
def crop_ids(monkeypatch):
    """The crop of ROI (cx, ...) is filled with the id cx // 100."""
    def crop(frame, M, target):
        cx = M[0, 2] + (M[0, 0] - M[1, 0]) * target / 2  # roi_affine's centre
        return np.full((1, target, target, 3), round(cx) // 100, np.float32)
    monkeypatch.setattr(Preprocessor, 'crop_rotated_nhwc', staticmethod(crop))


def _rois(n):
    return [(100.0 * k + 50, 300.0, 200.0, 0.1 * k) for k in range(n)]


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.allclose(a.pts, b.pts) and a.presence == b.presence and a.roi == pytest.approx(b.roi)


@pytest.mark.parametrize('n,slots,batches', [(3, 4, [4]), (4, 4, [4]), (5, 4, [4, 4]), (1, 2, [2])])
def test_poses_in_one_run_match_one_run_each(n, slots, batches):
    single = BodyProcessor(_Models('pose'))
    batched = BodyProcessor(_Models('pose'), batch=slots)
    one_each = [single.process(None, BoundingBox(r[0] - r[2] / 2, r[1] - r[2] / 2, r[2], r[2], 1.0), 1920, 1080, False,
                               roi=r) for r in _rois(n)]
    together = batched.process_many(None, _rois(n))
    assert batched.model_manager.batch.batches == batches  # always the full batch (one CUDA graph)
    assert single.model_manager.single.batches == [1] * n
    assert len(together) == n
    assert all(_same(a, b) for a, b in zip(one_each, together))
    assert [r is None for r in together] == [k % 2 == 1 for k in range(n)]


def test_no_poses_runs_nothing():
    batched = BodyProcessor(_Models('pose'), batch=4)
    assert batched.process_many(None, []) == []
    assert batched.model_manager.batch.batches == []


def _pose(x0):
    """33+6 pose landmarks around x0 with both wrists visible (wrists 15/16, pinky 17/18, index 19/20)."""
    pts = [Landmark(x0 + 10 * (i % 7), 200 + 15 * (i % 11), 0.0, 1.0, 1.0) for i in range(39)]
    for side in (0, 1):
        pts[15 + side] = Landmark(x0 + 100 * side, 400, 0, 1.0, 1.0)
        pts[17 + side] = Landmark(x0 + 100 * side + 10, 440, 0, 1.0, 1.0)
        pts[19 + side] = Landmark(x0 + 100 * side - 10, 440, 0, 1.0, 1.0)
    return pts


def test_hands_of_all_people_in_one_run_match_one_run_each():
    people = [(_pose(200), (None, None)), (_pose(600), (None, (900.0, 300.0, 120.0, 0.2))),
              (_pose(1000), (None, None))]
    single = HandProcessor(_Models('hand'))
    batched = HandProcessor(_Models('hand'), batch=4)
    one_each = [single.process(None, pose, rois) for pose, rois in people]
    together = batched.process_many(None, people)
    assert batched.model_manager.batch.batches == [4, 4]  # 6 hands: two full runs
    assert len(together) == 3
    for a, b in zip(one_each, together):
        assert len(b) == 2
        assert all(_same(x, y) for x, y in zip(a, b))
    assert any(h is not None for hs in together for h in hs)


def test_hand_without_visible_wrist_is_not_run():
    pose = _pose(200)
    pose[15] = Landmark(pose[15].x, pose[15].y, 0, 0.1, 1.0)  # left wrist not visible, no previous crop
    batched = HandProcessor(_Models('hand'), batch=2)
    hands = batched.process_many(None, [(pose, (None, None))])
    assert hands[0][0] is None
    assert batched.model_manager.batch.batches == [2]
