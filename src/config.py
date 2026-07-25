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
    parser.set_defaults(face=True, body=True, hands=True)
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
    )
