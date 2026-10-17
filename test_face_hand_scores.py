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


def test_forearm_crop_is_past_the_wrist_and_sized_from_the_forearm():
    import math
    from tools.orin_bench.face_hand_scores import forearm_roi
    pose = _pose()
    pose[13], pose[15] = Landmark(100.0, 100.0, 0.0), Landmark(100.0, 200.0, 0.0)  # forearm pointing down, 100 long
    cx, cy, size, angle = forearm_roi(pose, 0, 1.0)
    assert (round(cx), round(cy), round(size)) == (100, 230, 100)
    assert abs(abs(angle) - math.pi) < 1e-6  # the forearm (elbow to wrist) points down: the crop is turned over


def test_shoulder_crop_keeps_its_size_when_the_forearm_points_at_the_camera():
    from tools.orin_bench.face_hand_scores import shoulder_roi
    pose = _pose()  # shoulders 80 apart
    pose[13], pose[15] = Landmark(120.0, 180.0, 0.0), Landmark(121.0, 181.0, 0.0)  # forearm almost a point
    cx, cy, size, angle = shoulder_roi(pose, 0, 0.8)
    assert round(size) == 64 and (round(cx), round(cy)) == (121, 165)  # straight up from the wrist, 0.2 x 80
    assert abs(angle) < 1e-6


def test_search_tries_the_wrist_crop_and_four_moved_copies():
    from tools.orin_bench.face_hand_scores import search_rois
    rois = search_rois(_pose(), 0, 1.0, 0.4)  # shoulders 80 apart: moves of 32 px
    centres = sorted((round(r[0] - rois[0][0]), round(r[1] - rois[0][1])) for r in rois)
    assert centres == sorted([(0, 0), (0, -32), (0, 32), (-32, 0), (32, 0)])
    assert all(r[2] == rois[0][2] for r in rois)
