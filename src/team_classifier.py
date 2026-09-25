"""Classify tracked people by match-specific kit colors."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np


ROLE_NAMES = {
    "team_a",
    "team_b",
    "team_a_goalkeeper",
    "team_b_goalkeeper",
    "referee",
    "unknown",
}
SUPPORTED_KIT_COLORS = {"red", "white", "black", "blue", "yellow"}


@dataclass(frozen=True)
class RolePrediction:
    role: str
    confidence: float


def extract_jersey_crop(
    frame: np.ndarray,
    coordinates: tuple[int, int, int, int],
) -> np.ndarray:
    """Return the central upper-body region of a clipped person box."""

    if frame.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = coordinates
    left = max(0, min(frame_width, x1))
    right = max(0, min(frame_width, x2))
    top = max(0, min(frame_height, y1))
    bottom = max(0, min(frame_height, y2))
    box_width = right - left
    box_height = bottom - top
    if box_width < 4 or box_height < 4:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    crop_left = left + round(box_width * 0.20)
    crop_right = left + round(box_width * 0.80)
    crop_top = top + round(box_height * 0.12)
    crop_bottom = top + round(box_height * 0.55)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    return frame[crop_top:crop_bottom, crop_left:crop_right]


def _color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    if color_name == "red":
        return (
            ((hue <= 10) | (hue >= 170))
            & (saturation >= 80)
            & (value >= 40)
        )
    if color_name == "white":
        return (saturation <= 60) & (value >= 120)
    if color_name == "black":
        return (saturation <= 80) & (value <= 85)
    if color_name == "blue":
        return (
            (hue >= 90)
            & (hue <= 135)
            & (saturation >= 70)
            & (value >= 45)
        )
    if color_name == "yellow":
        return (
            (hue >= 15)
            & (hue <= 40)
            & (saturation >= 70)
            & (value >= 70)
        )
    raise ValueError(f"Unsupported kit color: {color_name}")


class KitColorClassifier:
    """Classify torso pixels against configured match kit colors."""

    def __init__(
        self,
        team_a_color: str = "red",
        team_b_color: str = "white",
        referee_color: str = "black",
        team_a_goalkeeper_color: str | None = None,
        team_b_goalkeeper_color: str | None = "blue",
        min_pixels: int = 20,
        min_color_ratio: float = 0.20,
        min_margin: float = 0.08,
    ) -> None:
        self.role_colors = {
            "team_a": team_a_color.lower(),
            "team_b": team_b_color.lower(),
            "referee": referee_color.lower(),
        }
        if team_a_goalkeeper_color is not None:
            self.role_colors["team_a_goalkeeper"] = (
                team_a_goalkeeper_color.lower()
            )
        if team_b_goalkeeper_color is not None:
            self.role_colors["team_b_goalkeeper"] = (
                team_b_goalkeeper_color.lower()
            )
        unsupported = set(self.role_colors.values()) - SUPPORTED_KIT_COLORS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported kit color(s): {names}")
        if len(set(self.role_colors.values())) != len(self.role_colors):
            raise ValueError(
                "Configured player, goalkeeper, and referee colors must differ"
            )

        self.min_pixels = min_pixels
        self.min_color_ratio = min_color_ratio
        self.min_margin = min_margin

    def predict(
        self,
        frame: np.ndarray,
        coordinates: tuple[int, int, int, int],
    ) -> RolePrediction:
        crop = extract_jersey_crop(frame, coordinates)
        if crop.size == 0 or crop.shape[0] * crop.shape[1] < self.min_pixels:
            return RolePrediction("unknown", 0.0)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixel_count = float(crop.shape[0] * crop.shape[1])
        scores = {
            role: float(np.count_nonzero(_color_mask(hsv, color_name)))
            / pixel_count
            for role, color_name in self.role_colors.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_role, best_score = ranked[0]
        second_score = ranked[1][1]
        if (
            best_score < self.min_color_ratio
            or best_score - second_score < self.min_margin
        ):
            return RolePrediction("unknown", best_score)

        return RolePrediction(best_role, best_score)


class TrackRoleVoter:
    """Turn noisy per-frame kit predictions into stable per-track roles."""

    def __init__(
        self,
        min_votes: int = 3,
        window_size: int = 15,
        stale_after_frames: int = 90,
    ) -> None:
        if min_votes < 1:
            raise ValueError("min_votes must be at least 1")
        if window_size < min_votes:
            raise ValueError("window_size must be at least min_votes")
        if stale_after_frames < 1:
            raise ValueError("stale_after_frames must be at least 1")

        self.min_votes = min_votes
        self.window_size = window_size
        self.stale_after_frames = stale_after_frames
        self._votes: dict[int, deque[str]] = {}
        self._stable_roles: dict[int, str] = {}
        self._last_seen: dict[int, int] = {}

    def update(
        self,
        track_id: int | None,
        prediction: RolePrediction,
        frame_index: int,
    ) -> str:
        if prediction.role not in ROLE_NAMES:
            raise ValueError(f"Unsupported role prediction: {prediction.role}")
        if track_id is None:
            return prediction.role

        last_seen = self._last_seen.get(track_id)
        if (
            last_seen is not None
            and frame_index - last_seen > self.stale_after_frames
        ):
            self._votes.pop(track_id, None)
            self._stable_roles.pop(track_id, None)

        self._last_seen[track_id] = frame_index
        votes = self._votes.setdefault(track_id, deque(maxlen=self.window_size))
        if prediction.role != "unknown":
            votes.append(prediction.role)

        stable_role = self._stable_roles.get(track_id, "unknown")
        if len(votes) < self.min_votes:
            return stable_role

        counts = Counter(votes)
        ranked = counts.most_common()
        top_role, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count >= self.min_votes and top_count > second_count:
            stable_role = top_role
            self._stable_roles[track_id] = top_role

        return stable_role
