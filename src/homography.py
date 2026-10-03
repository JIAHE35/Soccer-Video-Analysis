"""Manual, time-bounded pitch calibration for a single continuous shot."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path

import cv2
import numpy as np


# Coordinates assume a 105 x 68 m pitch, right goal at x=105, far touchline y=0.
ARC_OFFSET = sqrt(9.15**2 - 5.5**2)
LANDMARKS = {
    "penalty_far": (88.5, 13.84),
    "penalty_near": (88.5, 54.16),
    "six_yard_far": (99.5, 24.84),
    "six_yard_near": (99.5, 43.16),
    "arc_far": (88.5, 34.0 - ARC_OFFSET),
    "arc_near": (88.5, 34.0 + ARC_OFFSET),
    "penalty_spot": (94.0, 34.0),
    "goalpost_far": (105.0, 30.34),
    "goalpost_near": (105.0, 37.66),
    "corner_far": (105.0, 0.0),
}


def transform_points(matrix: np.ndarray, points) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(len(values)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Point projects to infinity or has non-finite coordinates")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fit_homography(image_points, pitch_points) -> np.ndarray:
    image = np.asarray(image_points, dtype=np.float64)
    pitch = np.asarray(pitch_points, dtype=np.float64)
    if image.ndim != 2 or image.shape[1:] != (2,) or image.shape != pitch.shape:
        raise ValueError("Image and pitch points must be matching Nx2 arrays")
    if len(image) < 4 or not np.isfinite(image).all() or not np.isfinite(pitch).all():
        raise ValueError("At least four finite corresponding points are required")
    for values in (image, pitch):
        if len(np.unique(values, axis=0)) != len(values):
            raise ValueError("Duplicate landmarks are not allowed")
        if np.linalg.matrix_rank(np.column_stack((values, np.ones(len(values))))) < 3:
            raise ValueError("Landmarks cannot all be collinear")
    matrix, _ = cv2.findHomography(image, pitch, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Could not estimate a non-degenerate homography")
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-9 else matrix


def object_anchor(role: str, bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid detection box")
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0 if role == "ball" else y2)


@dataclass(frozen=True)
class MappingResult:
    point: tuple[float, float] | None
    valid: bool
    reason: str


class PitchCalibration:
    """Interpolate projected reference points, never homography coefficients."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.fps = float(config["fps"])
        self.frame_count = int(config["frame_count"])
        self.image_size = tuple(config["image_size"])
        self.source_start_frame = int(config["source_start_frame"])
        self.shot_id = config["shot_id"]
        self.roi = np.asarray(config["valid_pitch_polygon"], dtype=np.float32)
        if (not np.isfinite(self.fps) or self.fps <= 0 or self.frame_count < 1
                or len(self.image_size) != 2 or min(self.image_size) <= 0
                or self.source_start_frame < 0):
            raise ValueError("Invalid clip metadata")
        if tuple(config.get("pitch_size_m", [105, 68])) != (105, 68):
            raise ValueError("Landmark geometry currently assumes a 105 x 68 m pitch")
        if (self.roi.shape != (4, 2) or not np.isfinite(self.roi).all()
                or not cv2.isContourConvex(self.roi) or cv2.contourArea(self.roi) < 1):
            raise ValueError("Valid region must be a finite convex four-point polygon")
        if np.any(self.roi < 0) or np.any(self.roi > [105, 68]):
            raise ValueError("Valid region must be within the pitch")
        self.keyframes = config["keyframes"]
        self.frames = [int(item["frame"]) for item in self.keyframes]
        if len(self.frames) < 2 or self.frames != sorted(set(self.frames)):
            raise ValueError("Need at least two strictly ordered keyframes")
        if self.frames[0] < 0 or self.frames[-1] >= self.frame_count:
            raise ValueError("Keyframe is outside the clip")
        self.matrices = []
        self.projected_references = []
        self.reprojection_errors = []
        for item in self.keyframes:
            names = list(item["points"])
            if any(name not in LANDMARKS for name in names):
                raise ValueError("Unknown pitch landmark")
            image_points = np.asarray([item["points"][name] for name in names], float)
            if (image_points.shape != (len(names), 2) or not np.isfinite(image_points).all()
                    or np.any(image_points < 0)
                    or np.any(image_points[:, 0] >= self.image_size[0])
                    or np.any(image_points[:, 1] >= self.image_size[1])):
                raise ValueError("Annotated landmarks must be inside the image")
            pitch_points = np.asarray([LANDMARKS[name] for name in names], float)
            matrix = fit_homography(image_points, pitch_points)
            inverse = np.linalg.inv(matrix)
            residuals = np.linalg.norm(transform_points(inverse, pitch_points) - image_points, axis=1)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            if rmse > config.get("max_fit_rmse_px", 12.0):
                raise ValueError(f"Frame {item['frame']}: calibration fit RMSE {rmse:.2f}px is too large")
            self.matrices.append(matrix)
            self.projected_references.append(transform_points(inverse, self.roi))
            self.reprojection_errors.append(rmse)

    @classmethod
    def load(cls, path: str | Path) -> PitchCalibration:
        with Path(path).open(encoding="utf-8") as stream:
            return cls(json.load(stream))

    def matrix_at(self, frame: int) -> np.ndarray | None:
        if not self.frames[0] <= frame <= self.frames[-1]:
            return None
        index = bisect_right(self.frames, frame) - 1
        if self.frames[index] == frame:
            return self.matrices[index].copy()
        ratio = (frame - self.frames[index]) / (self.frames[index + 1] - self.frames[index])
        references = ((1 - ratio) * self.projected_references[index]
                      + ratio * self.projected_references[index + 1])
        return fit_homography(references, self.roi)

    def map_point(self, frame: int, point) -> MappingResult:
        matrix = self.matrix_at(frame)
        if matrix is None:
            return MappingResult(None, False, "uncalibrated_frame")
        x, y = map(float, point)
        if not np.isfinite([x, y]).all() or not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
            return MappingResult(None, False, "anchor_outside_image")
        try:
            mapped = tuple(map(float, transform_points(matrix, [[x, y]])[0]))
        except ValueError:
            return MappingResult(None, False, "invalid_projection")
        if cv2.pointPolygonTest(self.roi, mapped, False) < 0:
            return MappingResult(None, False, "outside_local_region")
        return MappingResult(mapped, True, "inside_local_region")
