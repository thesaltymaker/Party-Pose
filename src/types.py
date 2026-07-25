from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class BoundingBox:
    """Represents a bounding box for object detection.

    Attributes:
        x: The x-coordinate of the top-left corner of the bounding box.
        y: The y-coordinate of the top-left corner of the bounding box.
        w: The width of the bounding box.
        h: The height of the bounding box.
        confidence: The confidence score of the detection (0.0 to 1.0).
    """
    x: float
    y: float
    w: float
    h: float
    confidence: float

@dataclass
class Landmark:
    """Represents a landmark point in 3D space.

    Attributes:
        x: The x-coordinate of the landmark.
        y: The y-coordinate of the landmark.
        z: The z-coordinate of the landmark.
        visibility: The visibility score of the landmark (0.0 to 1.0).
        presence: The presence score of the landmark (0.0 to 1.0).
    """
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0

@dataclass
class FaceResult:
    """Represents the result of a face detection.

    Attributes:
        bbox: The bounding box of the detected face.
        landmarks: A list of landmarks for the detected face.
        blendshapes: Optional array of blendshapes (facial expressions) as numpy array.
        person_id: Index of the person this face belongs to (assigned after detection).
    """
    bbox: BoundingBox
    landmarks: List[Landmark]
    blendshapes: Optional[np.ndarray] = None
    person_id: int = 0

@dataclass
class HandResult:
    """Represents the result of a hand detection.

    Attributes:
        bbox: The bounding box of the detected hand.
        landmarks: A list of landmarks for the detected hand.
        handedness: The handedness of the detected hand ('Right' or 'Left').
        handedness_score: The confidence score of the handedness classification (0.0 to 1.0).
        person_id: Index of the person this hand belongs to (assigned after detection).
    """
    bbox: BoundingBox
    landmarks: List[Landmark]
    handedness: str = 'Right'
    handedness_score: float = 0.0
    person_id: int = 0

@dataclass
class BodyResult:
    """Represents the result of a body pose detection.

    Attributes:
        landmarks: A list of landmarks for the detected body pose.
        presence: The presence score of the body pose (0.0 to 1.0).
        person_id: Index of the person this body belongs to.
    """
    landmarks: List[Landmark]
    presence: float = 0.0
    person_id: int = 0
