import collections
import time

class FPSCounter:
    """A class to measure frames per second (FPS) in real-time video applications.

    Attributes:
        timestamps (collections.deque): A deque storing timestamps of recent frames.
        window (int): The number of frames to consider for FPS calculation.
    """

    def __init__(self, window: int = 30):
        self.timestamps = collections.deque(maxlen=window)
        self.window = window

    def tick(self) -> None:
        """Record the current time for FPS calculation."""
        self.timestamps.append(time.monotonic())

    def get_fps(self) -> float:
        """Calculate the current FPS based on the stored timestamps.

        Returns:
            float: The calculated FPS, or 0.0 if insufficient data is available.
        """
        if len(self.timestamps) < 2:
            return 0.0
        newest_ts = self.timestamps[-1]
        oldest_ts = self.timestamps[0]
        return self.window / (newest_ts - oldest_ts)
