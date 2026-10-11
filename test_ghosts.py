import numpy as np
import pytest

from src.ghosts import Ghosts
from src.renderer import Renderer, PERSON_COLORS
from src.types import BodyResult, Landmark


def test_missing_part_dims_over_5_frames_then_is_gone():
    g = Ghosts(frames=5)
    assert g.update({(1, 'body'): 'b1'}) == []
    fades = [g.update({}) for _ in range(6)]
    assert fades[:4] == [[((1, 'body'), 'b1', f)] for f in (0.8, 0.6, 0.4, 0.2)]
    assert fades[4:] == [[], []]  # 5 frames (1/6 s at 30 fps) after the last detection: nothing drawn


def test_new_detection_replaces_the_ghost_at_once():
    g = Ghosts(frames=5)
    g.update({(1, 'face'): 'f1'})
    g.update({})
    assert g.update({(1, 'face'): 'f2'}) == []  # drawn fresh, no ghost alongside
    assert g.update({}) == [((1, 'face'), 'f2', 0.8)]  # the next ghost is the newer result, from full


def test_parts_are_ghosted_on_their_own():
    g = Ghosts(frames=5)
    g.update({(1, 'body'): 'b', (1, 'hand0'): 'h0', (1, 'hand1'): 'h1'})
    g.update({(1, 'body'): 'b', (1, 'hand1'): 'h1'})
    assert g.update({(1, 'body'): 'b', (1, 'hand1'): 'h1'}) == [((1, 'hand0'), 'h0', 0.6)]


def test_ghost_stays_with_its_track_and_never_moves_to_another():
    g = Ghosts(frames=5)
    g.update({(1, 'body'): 'person1'})
    out = g.update({(2, 'body'): 'person2'})  # track 1 gone, a new track where it was
    assert out == [((1, 'body'), 'person1', 0.8)]


def test_zero_frames_turns_ghosts_off():
    g = Ghosts(frames=0)
    g.update({(1, 'body'): 'b'})
    assert g.update({}) == []


def _body(person_id, fade=1.0):
    lms = [Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    lms[11], lms[12] = Landmark(100.0, 100.0, 0.0), Landmark(300.0, 100.0, 0.0)  # shoulders
    return BodyResult(landmarks=lms, presence=1.0, person_id=person_id, fade=fade)


@pytest.mark.parametrize('fade', [1.0, 0.4])
def test_renderer_dims_a_ghost_in_its_track_colour(fade):
    frame = np.zeros((200, 400, 3), np.uint8)
    Renderer().draw_body(frame, [_body(3, fade)], 400, 200)
    expected = tuple(int(c * fade) for c in PERSON_COLORS[3])
    assert tuple(frame[100, 200]) == expected  # middle of the shoulder line
