"""Draw pitch geometry, calibration overlays, and the V0.6 map panel."""

from __future__ import annotations

import cv2
import numpy as np

if __package__:
    from .homography import transform_points
else:
    from homography import transform_points


ROLE_COLORS = {
    "team_a": (55, 65, 235), "team_b": (240, 225, 60),
    "team_a_goalkeeper": (0, 165, 255), "team_b_goalkeeper": (245, 80, 220),
    "referee": (60, 230, 245), "unknown": (160, 160, 160),
    "player": (60, 220, 60), "ball": (255, 110, 60),
}


def pitch_lines() -> list[np.ndarray]:
    lines = [
        [(52.5, 0), (105, 0), (105, 68), (52.5, 68), (52.5, 0)],
        [(105, 13.84), (88.5, 13.84), (88.5, 54.16), (105, 54.16)],
        [(105, 24.84), (99.5, 24.84), (99.5, 43.16), (105, 43.16)],
    ]
    # Only the segment of the penalty arc outside the penalty box.
    theta = np.linspace(np.arccos(-5.5 / 9.15), 2 * np.pi - np.arccos(-5.5 / 9.15), 60)
    lines.append(np.column_stack((94 + 9.15 * np.cos(theta), 34 + 9.15 * np.sin(theta))))
    theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
    lines.append(np.column_stack((52.5 + 9.15 * np.cos(theta), 34 + 9.15 * np.sin(theta))))
    return [np.asarray(line, dtype=float) for line in lines]


def draw_calibration_overlay(frame: np.ndarray, matrix: np.ndarray, roi) -> np.ndarray:
    canvas = frame.copy()
    inverse = np.linalg.inv(matrix)
    for line in pitch_lines()[1:4]:
        points = transform_points(inverse, line)
        if not np.isfinite(points).all() or np.max(abs(points)) > 100000:
            continue
        cv2.polylines(canvas, [np.round(points).astype(np.int32)], False, (230, 180, 40), 2, cv2.LINE_AA)
    region = transform_points(inverse, roi)
    if np.max(abs(region)) < 100000:
        cv2.polylines(canvas, [np.round(region).astype(np.int32)], True, (70, 230, 230), 1, cv2.LINE_AA)
    return canvas


def draw_pitch_panel(height: int, width: int, mapped_rows: list[dict], roi,
                     clip_time: float, source_time: float) -> np.ndarray:
    panel = np.full((height, width, 3), (28, 33, 31), dtype=np.uint8)
    cv2.putText(panel, "V0.6 | PITCH MAPPING", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 240, 238), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Clip {clip_time:05.2f}s | Source {source_time:05.2f}s", (24, 56), cv2.FONT_HERSHEY_SIMPLEX, .45, (172, 186, 178), 1, cv2.LINE_AA)
    scale = min((height - 174) / 68.0, (width - 64) / 55.0)
    origin_x = int((width - 52.5 * scale) / 2)
    origin_y = 80

    def to_pixels(points):
        points = np.asarray(points, float)
        return np.round((points - [52.5, 0]) * scale + [origin_x, origin_y]).astype(np.int32)

    outer = to_pixels([[52.5, 0], [105, 0], [105, 68], [52.5, 68]])
    cv2.fillConvexPoly(panel, outer, (48, 76, 52))
    cv2.fillConvexPoly(panel, to_pixels(roi), (63, 112, 70))
    for line in pitch_lines():
        cv2.polylines(panel, [to_pixels(line)], False, (190, 208, 193), 1, cv2.LINE_AA)
    cv2.circle(panel, tuple(to_pixels([[94, 34]])[0]), 2, (220, 230, 220), -1)
    cv2.polylines(panel, [to_pixels([[105, 30.34], [107, 30.34], [107, 37.66], [105, 37.66]])], False, (200, 218, 205), 1)
    # Draw the ball last so person markers cannot conceal it.
    valid_rows = [row for row in mapped_rows if row["homography_valid"] == "true"]
    for row in sorted(valid_rows, key=lambda row: row["role"] == "ball"):
        center = tuple(to_pixels([[float(row["pitch_x"]), float(row["pitch_y"])]])[0])
        role = row["role"]
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        radius = 6 if role == "ball" else 4
        fill = 2 if row.get("ball_state") == "predicted" or "goalkeeper" in role else -1
        cv2.circle(panel, center, radius + 1, (20, 25, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, center, radius, color, fill, cv2.LINE_AA)
    y = height - 76
    cv2.putText(panel, f"Mapped {len(valid_rows)} / {len(mapped_rows)} | Right penalty area", (24, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (198, 210, 202), 1, cv2.LINE_AA)
    for i, (role, label) in enumerate((("team_a", "A"), ("team_b", "B"), ("referee", "Ref"), ("team_b_goalkeeper", "GK"), ("ball", "Ball"))):
        x = 30 + i * 85
        cv2.circle(panel, (x, y + 24), 4, ROLE_COLORS[role], -1)
        cv2.putText(panel, label, (x + 12, y + 29), cv2.FONT_HERSHEY_SIMPLEX, .42, (217, 227, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "105 x 68 m assumed | Ball: ground-plane estimate", (24, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (172, 186, 178), 1, cv2.LINE_AA)
    return panel
