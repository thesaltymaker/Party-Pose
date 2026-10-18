from typing import Optional
from dataclasses import dataclass, field
import argparse
import platform as platform_module

@dataclass(frozen=True)
class Config:
    camera: int = 0
    width: int = 0   # display window width  (0 = match camera)
    height: int = 0  # display window height (0 = match camera)
    mirror: bool = False
    face: bool = True
    body: bool = True
    hands: bool = True
    show_roi: bool = False
    show_fps: bool = False
    face_expressions: bool = False
    black_bg: bool = False
    platform: str = "laptop"
    dump_detections: str = ""  # debug: directory for annotated frames + detection JSON ("" = off)
    dump_every: float = 2.0
    fullscreen: bool = False
    show_ids: bool = False      # draw '#<track id>' on each person
    track_report: float = 0.0   # seconds between [TRACK] stat lines (0 = off)
    require_face: bool = False  # draw a person only once a real face has been seen in their track
    drop_still: bool = False    # don't draw tracks that haven't moved for ~10 s (props; issue #7)
    min_body_height: float = 0.40  # don't draw people whose box is under this fraction of the frame height (0 = off)
    test_label: str = ""        # live tests: text and a seconds counter drawn at the top of the screen
    ghost_frames: int = 5       # frames a missing body, face or hand is redrawn, dimming (issue #13; 0 = off)
    face_hold: float = 0.0      # a face kept last frame stays down to this face logit (0 = off: presence 0.5)
    smooth: bool = True         # smooth landmarks across frames (--no-smooth for A/B tests)
    follow: bool = True         # keep a found person by their pose model, not the detector (--no-follow for A/B tests)
    pipeline: bool = False      # run the next frame's models while this frame is drawn (one frame later on screen)
    video: Optional[str] = None  # read this raw clip (src/raw_video.py) instead of the camera, as fast as possible
    skelly: bool = False        # filled bone-coloured face with eyes, nose and mouth cut out
    santa_hat: bool = False     # Santa hat on each face
    detector: str = ""          # person detector model file in models/ ("" = the default YOLOX nano 320x256)
    detector_bgr: bool = False  # feed the person detector BGR (the order YOLOX was trained on) instead of RGB

def detect_platform() -> str:
    """Detect whether we are running on an NVIDIA Jetson (Orin) board."""
    # Must be aarch64 and model indicates Jetson
    if platform_module.machine() != "aarch64":
        return "laptop"
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
        if "NVIDIA Jetson" in model:
            return "orin"
    except Exception:
        # Ignore any errors (missing file, permission) and fall back to laptop
        pass
    return "laptop"

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Pose detection app configuration")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=0, help="Display window width (0 = camera native)")
    parser.add_argument("--height", type=int, default=0, help="Display window height (0 = camera native)")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera feed")
    parser.add_argument("--no-face", dest="face", action="store_false", help="Disable face detection")
    parser.add_argument("--no-body", dest="body", action="store_false", help="Disable body detection")
    parser.add_argument("--no-hands", dest="hands", action="store_false", help="Disable hands detection")
    parser.add_argument("--show-roi", action="store_true", help="Show region of interest")
    parser.add_argument("--fps", action="store_true", help="Show FPS")
    parser.add_argument("--face-expressions", action="store_true", help="Enable face expressions detection")
    parser.add_argument("--black-bg", action="store_true", help="Black background — show only landmarks and skeleton")
    parser.add_argument("--dump-detections", default="", metavar="DIR", help="Save annotated frames + detection scores as JSON to DIR")
    parser.add_argument("--dump-every", type=float, default=2.0, help="Seconds between dumps (with --dump-detections)")
    parser.add_argument("--fullscreen", action="store_true", help="Show the output full screen (no title bar); Esc or q quits")
    parser.add_argument("--show-ids", action="store_true", help="Draw each person's track number on screen")
    parser.add_argument("--track-report", type=float, default=0.0, metavar="SECONDS",
                        help="Print per-track statistics (text only) every SECONDS (0 = off)")
    parser.add_argument("--require-face", action="store_true",
                        help="Draw a tracked body only after a real face was seen in it (fewer false positives; "
                             "people facing away are not drawn)")
    parser.add_argument("--drop-still", action="store_true",
                        help="Don't draw a person whose box hasn't moved for ~10 s (lamps, chairs; also people "
                             "sitting very still)")
    parser.add_argument("--min-body-height", type=float, default=0.40, metavar="FRACTION",
                        help="Don't draw a person whose box is under FRACTION of the frame height (people far "
                             "back in the room; default 0.40, 0 = off)")
    parser.add_argument("--test-label", default="", metavar="TEXT",
                        help="Live tests: show TEXT and the seconds since the picture appeared at the top of the screen")
    parser.add_argument("--ghost-frames", type=int, default=5, metavar="N",
                        help="When a tracked person's body, face or hand is missing, redraw its last result "
                             "for N-1 frames, dimming, then drop it (default 5 = 1/6 s at 30 FPS; 0 = off)")
    parser.add_argument("--face-hold", type=float, default=0.0, metavar="LOGIT",
                        help="A face kept last frame is kept while its face logit stays at or above LOGIT and the "
                             "pose's head is not turned away (e.g. -8; default 0 = off, presence 0.5 every frame)")
    parser.add_argument("--video", default=None, metavar="CLIP",
                        help="Read a clip recorded with tools/orin_bench/record_video.py instead of the camera, "
                             "as fast as possible, and exit at its end (repeatable benchmarks)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run the next frame's models in a worker thread while this frame is drawn "
                             "(faster; picture one frame later)")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false",
                        help="Don't smooth landmarks across frames (for A/B comparison)")
    parser.add_argument("--no-follow", dest="follow", action="store_false",
                        help="Draw a person only in frames with a detector box, as before (for A/B comparison)")
    parser.add_argument("--skelly", action="store_true", help="Filled skeleton face (eyes, nose and mouth cut out)")
    parser.add_argument("--santa-hat", action="store_true", help="Draw a Santa hat on each face")
    parser.add_argument("--detector", default="", metavar="FILE",
                        help="Person detector model file in models/ (NHWC input; tools/orin_bench/detector_nhwc.py "
                             "makes one from a PINTO model zoo YOLOX-Body-Head-Hand model; default: the YOLOX nano 320x256)")
    parser.add_argument("--detector-bgr", action="store_true",
                        help="Feed the person detector the frame in BGR order, as the model was trained, instead of RGB "
                             "(finds more people)")
    parser.set_defaults(face=True, body=True, hands=True, require_face=False)
    # Platform argument
    parser.add_argument(
        "--platform",
        type=str,
        choices=["orin", "laptop", "auto"],
        default="auto",
        help="Target platform (auto detects), 'orin' for NVIDIA Jetson, 'laptop' otherwise"
    )
    args = parser.parse_args()

    # Resolve the actual platform value
    if args.platform == "auto":
        resolved_platform = detect_platform()
    else:
        resolved_platform = args.platform

    return Config(
        camera=args.camera,
        width=args.width,
        height=args.height,
        mirror=args.mirror,
        face=args.face,
        body=args.body,
        hands=args.hands,
        show_roi=args.show_roi,
        show_fps=args.fps,
        face_expressions=args.face_expressions,
        black_bg=args.black_bg,
        platform=resolved_platform,
        dump_detections=args.dump_detections,
        dump_every=args.dump_every,
        fullscreen=args.fullscreen,
        show_ids=args.show_ids,
        track_report=args.track_report,
        require_face=args.require_face,
        drop_still=args.drop_still,
        min_body_height=args.min_body_height,
        ghost_frames=args.ghost_frames,
        face_hold=args.face_hold,
        test_label=args.test_label,
        smooth=args.smooth,
        follow=args.follow,
        pipeline=args.pipeline,
        video=args.video,
        skelly=args.skelly,
        santa_hat=args.santa_hat,
        detector=args.detector,
        detector_bgr=args.detector_bgr,
    )
