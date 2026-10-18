import builtins
from unittest.mock import mock_open, patch

import pytest

from src.config import detect_platform


def _set_machine(return_value):
    return patch("platform.machine", return_value=return_value)


def test_detect_platform_orin_success():
    with _set_machine("aarch64"):
        with patch(
            "builtins.open",
            new=mock_open(read_data="NVIDIA Jetson Orin Developer Kit\n"),
        ) as mocked_file:
            assert detect_platform() == "orin"
            mocked_file().read.assert_called_once()


def test_detect_platform_laptop_due_to_model():
    with _set_machine("aarch64"):
        with patch(
            "builtins.open",
            new=mock_open(read_data="Raspberry Pi 4 Model B"),
        ) as mocked_file:
            assert detect_platform() == "laptop"
            mocked_file().read.assert_called_once()


def test_detect_platform_laptop_due_to_exception():
    with _set_machine("aarch64"):
        with patch("builtins.open", side_effect=OSError):
            assert detect_platform() == "laptop"


def test_detect_platform_laptop_non_aarch64():
    with _set_machine("x86_64"):
        assert detect_platform() == "laptop"


def test_detect_platform_case_sensitivity():
    with _set_machine("aarch64"):
        with patch(
            "builtins.open",
            new=mock_open(read_data="nvidia jetson orin"),
        ) as mocked_file:
            assert detect_platform() == "laptop"
            mocked_file().read.assert_called_once()


@pytest.mark.parametrize(
    "machine, model_content, expected",
    [
        ("aarch64", "NVIDIA Jetson Orin", "orin"),
        ("aarch64", "Some other board", "laptop"),
        ("armv7l", "NVIDIA Jetson Orin", "laptop"),
    ],
)
def test_detect_platform_return_values(machine, model_content, expected):
    with _set_machine(machine):
        model = model_content if machine == "aarch64" else ""
        with patch("builtins.open", new=mock_open(read_data=model)):
            result = detect_platform()
            assert isinstance(result, str)
            assert result in {"orin", "laptop"}


def test_smoothing_is_on_unless_no_smooth():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().smooth is True
    with patch("sys.argv", ["poser.py", "--no-smooth"]):
        assert parse_args().smooth is False


def test_pipeline_is_off_unless_asked():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().pipeline is False
    with patch("sys.argv", ["poser.py", "--pipeline"]):
        assert parse_args().pipeline is True


def test_detector_is_the_default_model_unless_given():
    from pathlib import Path
    from src.config import parse_args
    from src.model_manager import ModelManager
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().detector == ""
    with patch("sys.argv", ["poser.py", "--detector", "big.onnx"]):
        assert parse_args().detector == "big.onnx"
    default = ModelManager(Path("models"), platform="orin")
    assert default._MODEL_FILES["person_detector"].startswith("yolox_n_body_head_hand_post_0461")
    other = ModelManager(Path("models"), platform="orin", detector="big.onnx")
    assert other._MODEL_FILES["person_detector"] == "big.onnx"
    assert other._TRT_MODEL_FILE_OVERRIDES["person_detector"] == "big.shapeinferred.onnx"
    assert ModelManager._MODEL_FILES["person_detector"] == default._MODEL_FILES["person_detector"]


def test_detector_gets_rgb_unless_detector_bgr():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().detector_bgr is False
    with patch("sys.argv", ["poser.py", "--detector-bgr"]):
        assert parse_args().detector_bgr is True


def test_detector_shrinks_with_linear_unless_detector_area():
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        assert parse_args().detector_area is False
    with patch("sys.argv", ["poser.py", "--detector-area"]):
        assert parse_args().detector_area is True


def test_pose_and_hand_model_files_can_be_chosen():
    from pathlib import Path
    from unittest.mock import patch
    from src.config import parse_args
    from src.model_manager import ModelManager
    with patch("sys.argv", ["poser.py"]):
        args = parse_args()
    assert args.pose_model == "" and args.hand_model == ""
    with patch("sys.argv", ["poser.py", "--pose-model", "lite.onnx", "--hand-model", "h.onnx"]):
        args = parse_args()
    assert (args.pose_model, args.hand_model) == ("lite.onnx", "h.onnx")
    default = ModelManager(Path("models"), platform="orin")
    other = ModelManager(Path("models"), platform="orin", pose_model="lite.onnx", hand_model="h.onnx")
    assert other._MODEL_FILES["pose_landmarks"] == "lite.onnx"
    assert other._MODEL_FILES["hand_landmarks"] == "h.onnx"
    assert default._MODEL_FILES["pose_landmarks"] == "pose_landmarks_detector_opset15.onnx"
    assert ModelManager._MODEL_FILES["pose_landmarks"] == default._MODEL_FILES["pose_landmarks"]


def test_hand_and_face_every_default_to_every_frame():
    from unittest.mock import patch
    from src.config import parse_args
    with patch("sys.argv", ["poser.py"]):
        args = parse_args()
    assert (args.hand_every, args.face_every) == (0, 0)
    with patch("sys.argv", ["poser.py", "--hand-every", "2", "--face-every", "3"]):
        args = parse_args()
    assert (args.hand_every, args.face_every) == (2, 3)
