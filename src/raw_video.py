"""Uncompressed BGR clips for repeatable benchmarks (`--video`).

A clip is `<path>.bgr` (frames back to back, uint8 HxWx3) plus `<path>.json` (width, height, count, fps). Frames
are read from disk one at a time, so a long 1080p clip doesn't have to fit in memory, and reading one takes
about what a camera read does (no video decoding on the Orin's CPU). Each frame is dropped from the file cache
after it is read: on the Orin the CPU and GPU share memory, and 11 GB of cached clip made the next run's GPU
allocation fail (NvMap error 12). Frames are read a few ahead in a background thread, as a camera captures
in the background: reading 6 MB inside read() made the replay's capture stage 11.6 ms against ~3 ms live.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import numpy as np


class RawVideoWriter:
    def __init__(self, path, width: int, height: int) -> None:
        self.path = Path(path)
        self.width, self.height = width, height
        self.count = 0
        self.fps = 0.0  # set by the recorder before closing
        self.meta = {}  # extra info saved in the .json (light, camera gain/exposure log)
        self._f = open(self.path.with_suffix('.bgr'), 'wb')

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f'frame {frame.shape} {frame.dtype}, clip is {self.height}x{self.width}x3 uint8')
        self._f.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self) -> None:
        self._f.close()
        self.path.with_suffix('.json').write_text(json.dumps(
            {**self.meta, 'width': self.width, 'height': self.height, 'count': self.count, 'fps': self.fps}))

    def __enter__(self) -> 'RawVideoWriter':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RawVideoReader:
    def __init__(self, path, ahead: int = 4) -> None:
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        self.width, self.height, self.count = meta['width'], meta['height'], meta['count']
        self.fps = meta.get('fps', 0.0)
        self._size = self.width * self.height * 3
        self._fd = os.open(path.with_suffix('.bgr'), os.O_RDONLY)
        self._frames: queue.Queue = queue.Queue(maxsize=ahead)
        threading.Thread(target=self._read_ahead, name='clip-reader', daemon=True).start()

    def _read_ahead(self) -> None:
        for i in range(self.count):
            offset = i * self._size
            data = os.pread(self._fd, self._size, offset)
            os.posix_fadvise(self._fd, offset, self._size, os.POSIX_FADV_DONTNEED)
            self._frames.put(np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3))
        self._frames.put(None)

    def buffered(self) -> int:
        """Frames read ahead and waiting."""
        return self._frames.qsize()

    def read(self) -> np.ndarray:
        """The next frame; EOFError after the last one."""
        frame = self._frames.get()
        if frame is None:
            self._frames.put(None)  # stay at the end
            raise EOFError('end of video')
        return frame
