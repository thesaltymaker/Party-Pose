"""All model sessions are loaded at startup: a TensorRT session loaded on first use froze the picture (issue #13)."""
from types import SimpleNamespace

from poser import _load_sessions


class _Models:
    def __init__(self):
        self.loaded = []

    def get_session(self, name):
        self.loaded.append(name)

    def get_batch_session(self, name, n):
        self.loaded.append(f'{name}_b{n}')


def test_every_batch_size_is_loaded_for_each_part_in_use():
    m = _Models()
    _load_sessions(m, SimpleNamespace(pose_batch=2, hand_batch=3, face_batch=0), True, True, True, True)
    assert sorted(m.loaded) == sorted(['person_detector', 'pose_landmarks', 'pose_landmarks_b1', 'pose_landmarks_b2',
                                       'hand_landmarks', 'hand_landmarks_b1', 'hand_landmarks_b2',
                                       'hand_landmarks_b3', 'face_landmarks'])


def test_parts_turned_off_are_not_loaded():
    m = _Models()
    _load_sessions(m, SimpleNamespace(pose_batch=4, hand_batch=4, face_batch=4), True, None, None, True)
    assert sorted(m.loaded) == ['person_detector', 'pose_landmarks', 'pose_landmarks_b1', 'pose_landmarks_b2',
                                'pose_landmarks_b3', 'pose_landmarks_b4']
