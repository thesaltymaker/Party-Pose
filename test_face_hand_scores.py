from src.types import BoundingBox, Landmark
from tools.orin_bench.face_hand_scores import over_torso, pose_face_box, scaled


def _pose():
    lms = [Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    for i in range(11):
        lms[i] = Landmark(100.0 + 4 * i, 50.0 + i, 0.0)  # face points spread 40 wide, 10 high
    lms[11], lms[12] = Landmark(80.0, 100.0, 0.0), Landmark(160.0, 100.0, 0.0)  # shoulders
    lms[23], lms[24] = Landmark(90.0, 250.0, 0.0), Landmark(150.0, 250.0, 0.0)  # hips
    lms[15], lms[16] = Landmark(120.0, 180.0, 0.0), Landmark(300.0, 180.0, 0.0)  # wrists: on the chest, out
    return lms


def test_scaled_box_keeps_its_centre():
    assert scaled(BoundingBox(100, 100, 40, 60, 0.9), 0.5) == BoundingBox(110, 115, 20, 30, 0.9)


def test_pose_face_box_is_a_square_around_the_face_points():
    box = pose_face_box(_pose(), 2.0)
    assert (box.w, box.h) == (80, 80)
    assert (box.x + box.w / 2, box.y + box.h / 2) == (120, 55)


def test_wrist_over_torso():
    assert over_torso(_pose(), 0) and not over_torso(_pose(), 1)
