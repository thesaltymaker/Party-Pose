from __future__ import annotations

import cv2
import numpy as np
import subprocess
import time
from typing import Tuple, Optional

NIGHT_GAIN_MAX = 171
NIGHT_GAIN_THRESHOLD_FRAC = 0.95
NIGHT_EXPOSURE_MIN_US = 27000  # ~the whole frame time at 30 fps: the camera cannot expose longer
DAY_GAIN_MAX = 64  # back to day only when this bright, well below night: IR light lowers the gain a little
SWITCH_CHECKS = 3  # readings in a row (one a second) before switching
IRCUT_CHIP, IRCUT_LINE = "gpiochip0", 144  # header pin 7 (PAC.06), tools/orin_setup/ircut: high = filter out
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


class IRCutFilter:
    """The camera's IR-cut filter on header pin 7: high moves it out (night, IR reaches the sensor), low puts it
    back (day). A `gpioset --mode=signal` process holds the pin while it runs; set() replaces it."""

    def __init__(self, popen=lambda args: subprocess.Popen(args)) -> None:
        self._popen = popen
        self._proc = None
        self.set(False)

    def set(self, night: bool) -> None:
        self.close()
        try:
            self._proc = self._popen(["gpioset", "--mode=signal", IRCUT_CHIP, f"{IRCUT_LINE}={int(night)}"])
        except OSError as e:
            print(f"[IRCUT] could not set pin 7: {e}", flush=True)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
            self._proc = None


class NightModeDetector:
    """Night when the camera is at full gain and full exposure for SWITCH_CHECKS readings; day again only when
    the gain falls to DAY_GAIN_MAX. on_change(night) is called on each switch."""

    def __init__(self, check_every_n_frames: int = 30, query=query_gain_exposure, on_change=None) -> None:
        self._check_interval = max(1, check_every_n_frames)
        self._frame_count = 0
        self._is_night: bool = False
        self._query = query
        self._on_change = on_change
        self._streak = 0

    def update(self) -> bool:
        self._frame_count += 1
        if self._frame_count % self._check_interval == 0:
            gain, exposure = self._query()
            if gain is not None and exposure is not None:
                if self._is_night:
                    switch = gain <= DAY_GAIN_MAX
                else:
                    switch = gain >= NIGHT_GAIN_MAX * NIGHT_GAIN_THRESHOLD_FRAC and exposure >= NIGHT_EXPOSURE_MIN_US
                self._streak = self._streak + 1 if switch else 0
                if self._streak >= SWITCH_CHECKS:
                    self._is_night = not self._is_night
                    self._streak = 0
                    print(f"[NIGHT] {'night' if self._is_night else 'day'} (gain {gain}, exposure {exposure} us)",
                          flush=True)
                    if self._on_change is not None:
                        self._on_change(self._is_night)
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
        self._ircut = None
        self._night_mode = None
        if config.platform == "orin" and self._clip is None:
            self._ircut = IRCutFilter()
            self._night_mode = NightModeDetector(on_change=self._ircut.set)

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
        if self._ircut is not None:
            self._ircut.close()
