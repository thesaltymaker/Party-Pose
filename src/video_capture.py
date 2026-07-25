from __future__ import annotations

import cv2
import subprocess
from typing import Tuple, Optional

NIGHT_GAIN_MAX = 171
NIGHT_EXPOSURE_MAX = 683710
NIGHT_GAIN_THRESHOLD_FRAC = 0.95
NIGHT_EXPOSURE_THRESHOLD_FRAC = 0.5


def query_gain_exposure(device: str = "/dev/video0") -> Tuple[Optional[int], Optional[int]]:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--get-ctrl=gain,exposure"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return (None, None)
        gain: Optional[int] = None
        exposure: Optional[int] = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gain:"):
                try:
                    gain = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("exposure:"):
                try:
                    exposure = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
        return (gain, exposure)
    except Exception:
        return (None, None)


class NightModeDetector:
    def __init__(self, check_every_n_frames: int = 30) -> None:
        self._check_interval = max(1, check_every_n_frames)
        self._frame_count = 0
        self._is_night: bool = False

    def update(self) -> bool:
        self._frame_count += 1
        if self._frame_count % self._check_interval == 0:
            gain, exposure = query_gain_exposure()
            condition = (
                gain is not None
                and exposure is not None
                and gain >= NIGHT_GAIN_MAX * NIGHT_GAIN_THRESHOLD_FRAC
                and exposure >= NIGHT_EXPOSURE_MAX * NIGHT_EXPOSURE_THRESHOLD_FRAC
            )
            self._is_night = condition
        return self._is_night


class VideoCaptureModule:
    """GPU-resident video capture module for pose detection applications."""

    def __init__(self, config: 'Config') -> None:
        # Platform-specific pipeline creation
        if config.platform == "orin":
            gst_str = (
                f"nvarguscamerasrc sensor-id={config.camera} ! "
                f"video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, "
                f"format=(string)NV12, framerate=(fraction)30/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, format=(string)BGRx ! videoconvert ! "
                f"video/x-raw, format=(string)BGR ! appsink drop=True"
            )
            self._cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            # Default laptop / other platform
            self._cap = cv2.VideoCapture(config.camera)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {config.camera}")

        # Read actual camera dimensions — do not force a resolution the camera may not support
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mirror = config.mirror
        self._gpu_frame = cv2.cuda_GpuMat()
        # Initialize night mode detector for Orin platform
        self._night_mode = NightModeDetector() if config.platform == "orin" else None

    def read_frame(self) -> cv2.cuda_GpuMat:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError('Camera read failed')

        self._gpu_frame.upload(frame)
        if self.mirror:
            cv2.cuda.flip(self._gpu_frame, 1, self._gpu_frame)

        # Night-mode grayscale conversion
        if self._night_mode is not None and self._night_mode.update():
            gray = cv2.cuda.cvtColor(self._gpu_frame, cv2.COLOR_BGR2GRAY)
            self._gpu_frame = cv2.cuda.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return self._gpu_frame

    def release(self) -> None:
        self._cap.release()
