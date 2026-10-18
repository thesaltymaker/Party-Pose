from __future__ import annotations

import cv2
import numpy as np
import subprocess
import time
from typing import Tuple, Optional

NIGHT_GAIN_MAX = 171
NIGHT_EXPOSURE_MAX = 683710
NIGHT_GAIN_THRESHOLD_FRAC = 0.95
NIGHT_EXPOSURE_THRESHOLD_FRAC = 0.5
# Orin NoIR IMX219: per-channel BGR gain applied on top of wbmode=3, landing between
# the wbmode=2 and wbmode=3 looks (issue #4). Stored as gain*128 for an 8-bit GPU multiply.
ORIN_WB_GAIN_BGR = (1.145, 1.0, 0.937)


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

    REOPEN_TRIES = 5  # a failed camera read opens the camera again this many times before giving up

    def __init__(self, config: 'Config') -> None:
        self._clip = None
        self.clip_fps = 0.0
        video = getattr(config, 'video', None)
        # Platform-specific pipeline creation
        if video:
            # A recorded clip of raw camera frames: the same white-balance gain and mirroring are applied below.
            from src.raw_video import RawVideoReader
            self._clip = RawVideoReader(video)
            self._cap = None
        elif config.platform == "orin":
            gst_str = (
                f"nvarguscamerasrc sensor-id={config.camera} wbmode=3 ! "
                f"video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, "
                f"format=(string)NV12, framerate=(fraction)30/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, format=(string)BGRx ! videoconvert ! "
                f"video/x-raw, format=(string)BGR ! appsink drop=True"
            )
            self._cap_args = (gst_str, cv2.CAP_GSTREAMER)
        else:
            # Default laptop / other platform
            self._cap_args = (config.camera,)
        if not video:
            self._cap = cv2.VideoCapture(*self._cap_args)

        if self._clip is not None:
            self.width, self.height = self._clip.width, self._clip.height
            self.clip_fps = self._clip.fps  # replays smooth on the clip's own clock, not the replay speed
        else:
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open camera with index {config.camera}")
            # Read actual camera dimensions — do not force a resolution the camera may not support
            self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mirror = config.mirror
        self._gpu_frames = [cv2.cuda_GpuMat()]  # read_frame() uploads into these in turn
        self._next = 0
        self._wb_gain = None
        if config.platform == "orin":
            self._wb_gain = cv2.cuda_GpuMat()
            self._wb_gain.upload(np.full((self.height, self.width, 3),
                                         [round(g * 128) for g in ORIN_WB_GAIN_BGR], np.uint8))
        # Initialize night mode detector for Orin platform (not for a clip: it reads the live camera's gain)
        self._night_mode = NightModeDetector() if config.platform == "orin" and self._clip is None else None

    def read_frame(self) -> cv2.cuda_GpuMat:
        if self._clip is not None:
            frame = self._clip.read()  # EOFError at the end of the clip
        else:
            ret, frame = self._cap.read()
            if not ret:
                frame = self._reopen()

        gpu_frame = self._gpu_frames[self._next]
        self._next = (self._next + 1) % len(self._gpu_frames)
        gpu_frame.upload(frame)
        if self._wb_gain is not None:
            cv2.cuda.multiply(gpu_frame, self._wb_gain, gpu_frame, scale=1 / 128)
        if self.mirror:
            cv2.cuda.flip(gpu_frame, 1, gpu_frame)

        # Night-mode grayscale conversion
        if self._night_mode is not None and self._night_mode.update():
            gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gpu_frame

    def _reopen(self) -> np.ndarray:
        """The camera gave no frame (Orin, L4T 36.4.7: Argus could not get a frame buffer, NvMap error 12): open it
        again instead of stopping the show. RuntimeError after REOPEN_TRIES tries."""
        for n in range(1, self.REOPEN_TRIES + 1):
            print(f'[CAMERA] read failed, opening the camera again (try {n} of {self.REOPEN_TRIES})', flush=True)
            self._cap.release()
            time.sleep(1.0)
            self._cap = cv2.VideoCapture(*self._cap_args)
            ret, frame = self._cap.read()
            if ret:
                return frame
        raise RuntimeError('Camera read failed')

    def keep_frames(self, n: int) -> None:
        """Upload each frame into the next of N GPU buffers, so the last N frames stay valid (--detect-ahead)."""
        self._gpu_frames = [cv2.cuda_GpuMat() for _ in range(n)]
        self._next = 0

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
