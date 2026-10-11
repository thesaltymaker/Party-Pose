import cv2
import numpy as np

from src.preprocessor import Preprocessor


def _old_full_frame(frame_gpu, w, h):
    """The previous order: colour-convert the whole frame, then resize (kept here as the reference)."""
    fw, fh = frame_gpu.size()
    scale, px, py = Preprocessor.letterbox_params(fw, fh, w, h)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    small = cv2.cuda.resize(cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB), (nw, nh)).download()
    out = np.zeros((h, w, 3), np.uint8)
    out[py:py + nh, px:px + nw] = small
    return out.astype(np.float32)[None]


def test_full_frame_is_unchanged_when_resized_before_colour_conversion():
    # Person detector input: converting the 1080p frame cost 1.3 ms of 2.4 on the Orin; the channel swap gives the
    # same pixels when done on the small image.
    g = cv2.cuda_GpuMat()
    g.upload(np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    new = Preprocessor.full_frame_nhwc(g, 320, 256, normalize=False)
    assert new.shape == (1, 256, 320, 3)
    assert np.array_equal(new, _old_full_frame(g, 320, 256))

