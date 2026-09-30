"""Select one football candidate and preserve it across short detection gaps."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BallCandidate:
    bbox: BBox
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class BallTrackResult:
    bbox: BBox
    state: str
    confidence: float | None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class BallTracker:
    """Maintain a single short-term ball track in image coordinates."""

    def __init__(
        self,
        min_confirmed_hits: int = 2,
        max_missing_frames: int = 8,
        max_match_distance: float = 120.0,
        max_box_side: int = 48,
        max_aspect_ratio: float = 1.8,
        velocity_smoothing: float = 0.65,
        confirmation_player_distance: float = 90.0,
        unassisted_confirmation_distance: float = 20.0,
        high_confidence: float = 0.55,
    ) -> None:
        if min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be at least 1")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames cannot be negative")
        if max_match_distance <= 0:
            raise ValueError("max_match_distance must be positive")
        if max_box_side < 1:
            raise ValueError("max_box_side must be positive")
        if max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio must be at least 1")
        if not 0.0 <= velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1)")
        if confirmation_player_distance <= 0:
            raise ValueError("confirmation_player_distance must be positive")
        if unassisted_confirmation_distance <= 0:
            raise ValueError("unassisted_confirmation_distance must be positive")
        if not 0.0 <= high_confidence <= 1.0:
            raise ValueError("high_confidence must be in [0, 1]")

        self.min_confirmed_hits = min_confirmed_hits
        self.max_missing_frames = max_missing_frames
        self.max_match_distance = max_match_distance
        self.max_box_side = max_box_side
        self.max_aspect_ratio = max_aspect_ratio
        self.velocity_smoothing = velocity_smoothing
        self.confirmation_player_distance = confirmation_player_distance
        self.unassisted_confirmation_distance = unassisted_confirmation_distance
        self.high_confidence = high_confidence
        self.reset()

    def reset(self) -> None:
        self._bbox: BBox | None = None
        self._center: tuple[float, float] | None = None
        self._last_detection_center: tuple[float, float] | None = None
        self._start_center: tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._hits = 0
        self._max_confidence = 0.0
        self._missing_frames = 0
        self._confirmed = False

    def update(
        self,
        candidates: list[BallCandidate],
        player_anchors: Sequence[tuple[float, float]] | None = None,
    ) -> BallTrackResult | None:
        valid_candidates = [item for item in candidates if self._is_valid(item)]

        if self._center is None:
            if not valid_candidates:
                return None
            return self._start_track(
                max(valid_candidates, key=lambda item: item.confidence)
            )

        predicted_center = (
            self._center[0] + self._velocity[0],
            self._center[1] + self._velocity[1],
        )
        allowed_distance = self.max_match_distance * (
            1.0 + 0.35 * self._missing_frames
        )
        matched = [
            item
            for item in valid_candidates
            if self._distance(item.center, predicted_center) <= allowed_distance
        ]

        if not matched:
            if self._confirmed:
                return self._predict_or_expire()
            if not valid_candidates:
                self.reset()
                return None
            return self._start_track(
                max(valid_candidates, key=lambda item: item.confidence)
            )

        candidate = min(
            matched,
            key=lambda item: (
                self._distance(item.center, predicted_center) / allowed_distance
                - 0.20 * item.confidence
            ),
        )

        elapsed_frames = self._missing_frames + 1
        previous_detection_center = self._last_detection_center or self._center
        measured_velocity = (
            (candidate.center[0] - previous_detection_center[0]) / elapsed_frames,
            (candidate.center[1] - previous_detection_center[1]) / elapsed_frames,
        )
        old_weight = self.velocity_smoothing
        self._velocity = (
            old_weight * self._velocity[0]
            + (1.0 - old_weight) * measured_velocity[0],
            old_weight * self._velocity[1]
            + (1.0 - old_weight) * measured_velocity[1],
        )

        self._bbox = candidate.bbox
        self._center = candidate.center
        self._last_detection_center = candidate.center
        self._missing_frames = 0
        self._hits += 1
        self._max_confidence = max(self._max_confidence, candidate.confidence)
        if self._hits < self.min_confirmed_hits:
            return None
        if not self._confirmed and not self._has_confirmation_evidence(
            candidate,
            player_anchors,
        ):
            return None

        self._confirmed = True
        return BallTrackResult(candidate.bbox, "detected", candidate.confidence)

    def _start_track(self, candidate: BallCandidate) -> BallTrackResult | None:
        self.reset()
        self._bbox = candidate.bbox
        self._center = candidate.center
        self._last_detection_center = candidate.center
        self._start_center = candidate.center
        self._hits = 1
        self._max_confidence = candidate.confidence
        if self.min_confirmed_hits > 1:
            return None

        self._confirmed = True
        return BallTrackResult(candidate.bbox, "detected", candidate.confidence)

    def _has_confirmation_evidence(
        self,
        candidate: BallCandidate,
        player_anchors: Sequence[tuple[float, float]] | None,
    ) -> bool:
        if player_anchors is None:
            return True
        if self._max_confidence >= self.high_confidence:
            return True
        if any(
            self._distance(candidate.center, anchor)
            <= self.confirmation_player_distance
            for anchor in player_anchors
        ):
            return True
        if self._start_center is None:
            return False
        return (
            self._distance(candidate.center, self._start_center)
            >= self.unassisted_confirmation_distance
        )

    def _predict_or_expire(self) -> BallTrackResult | None:
        self._missing_frames += 1
        if self._missing_frames > self.max_missing_frames:
            self.reset()
            return None

        assert self._center is not None
        assert self._bbox is not None
        predicted_center = (
            self._center[0] + self._velocity[0],
            self._center[1] + self._velocity[1],
        )
        self._bbox = self._move_bbox(self._bbox, predicted_center)
        self._center = predicted_center
        return BallTrackResult(self._bbox, "predicted", None)

    def _is_valid(self, candidate: BallCandidate) -> bool:
        x1, y1, x2, y2 = candidate.bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        if width > self.max_box_side or height > self.max_box_side:
            return False
        aspect_ratio = width / float(height)
        return 0.35 <= aspect_ratio <= self.max_aspect_ratio

    @staticmethod
    def _distance(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _move_bbox(bbox: BBox, center: tuple[float, float]) -> BBox:
        x1, y1, x2, y2 = bbox
        half_width = (x2 - x1) / 2.0
        half_height = (y2 - y1) / 2.0
        return (
            round(center[0] - half_width),
            round(center[1] - half_height),
            round(center[0] + half_width),
            round(center[1] + half_height),
        )
