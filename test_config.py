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
