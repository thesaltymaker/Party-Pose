from __future__ import annotations

import cv2

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
                f"video/x-raw, format=(string)BGRx ! "
                f"videoconvert ! "
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

    def read_frame(self) -> cv2.cuda_GpuMat:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError('Camera read failed')

        self._gpu_frame.upload(frame)
        if self.mirror:
            cv2.cuda.flip(self._gpu_frame, 1, self._gpu_frame)
        return self._gpu_frame

    def release(self) -> None:
        self._cap.release()
