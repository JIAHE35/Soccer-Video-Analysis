"""Run a YOLO detection and player-tracking pipeline for the soccer project."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

if __package__:
    from .team_classifier import (
        SUPPORTED_KIT_COLORS,
        KitColorClassifier,
        TrackRoleVoter,
    )
else:
    from team_classifier import (
        SUPPORTED_KIT_COLORS,
        KitColorClassifier,
        TrackRoleVoter,
    )


# OpenCV uses BGR colors. White kits use cyan boxes to stay visible on screen.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (0, 255, 0),
    "team_a": (0, 0, 255),
    "team_b": (255, 255, 0),
    "team_a_goalkeeper": (0, 165, 255),
    "team_b_goalkeeper": (255, 0, 255),
    "referee": (0, 255, 255),
    "unknown": (160, 160, 160),
    "ball": (255, 0, 0),
}

DISPLAY_NAMES = {
    "player": "player",
    "team_a": "Team A",
    "team_b": "Team B",
    "team_a_goalkeeper": "Team A GK",
    "team_b_goalkeeper": "Team B GK",
    "referee": "Referee",
    "unknown": "Unknown",
    "ball": "ball",
}

TRACKED_PERSON_CATEGORIES = {
    "player",
    "team_a",
    "team_b",
    "team_a_goalkeeper",
    "team_b_goalkeeper",
    "referee",
    "unknown",
}

MODEL_LABEL_TO_PROJECT_LABEL = {
    "person": "player",
    "player": "player",
    "referee": "referee",
    "sports ball": "ball",
    "ball": "ball",
}

FIELD_HSV_LOWER = (25, 30, 20)
FIELD_HSV_UPPER = (100, 255, 255)
FIELD_PATCH_RADIUS = 10


def classify_model_label(model_label: str) -> str | None:
    """Map a model label to the current project label."""

    return MODEL_LABEL_TO_PROJECT_LABEL.get(model_label)


def select_model_class_ids(
    model_names: dict[int, str],
) -> tuple[list[int], list[int]]:
    """Split model class IDs into trackable people and detection-only balls."""

    people_class_ids: list[int] = []
    ball_class_ids: list[int] = []
    for class_id, model_label in model_names.items():
        category = classify_model_label(model_label)
        if category in {"player", "referee"}:
            people_class_ids.append(class_id)
        elif category == "ball":
            ball_class_ids.append(class_id)
    return people_class_ids, ball_class_ids


def choose_output_fps(source_fps: float, fallback: float = 30.0) -> float:
    """Keep the source playback speed, with a fallback for missing metadata."""

    return source_fps if source_fps > 0 else fallback


def build_field_mask(frame: Any, cv2: Any) -> Any:
    """Create a rough grass-field mask from the frame's HSV colors."""

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_frame, FIELD_HSV_LOWER, FIELD_HSV_UPPER)


def detection_anchor(
    category: str,
    coordinates: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return the point that should touch the field for a detection."""

    x1, y1, x2, y2 = coordinates
    center_x = (x1 + x2) // 2
    if category in {"player", "referee"}:
        return center_x, y2
    return center_x, (y1 + y2) // 2


def is_detection_on_field(
    field_mask: Any,
    category: str,
    coordinates: tuple[int, int, int, int],
    cv2: Any,
    min_green_ratio: float = 0.25,
) -> bool:
    """Keep detections whose anchor neighborhood contains enough grass."""

    x1, y1, x2, y2 = coordinates
    anchor_x, anchor_y = detection_anchor(category, coordinates)
    height, width = field_mask.shape[:2]
    if not (0 <= anchor_x < width and 0 <= anchor_y < height):
        return False

    left = max(0, anchor_x - FIELD_PATCH_RADIUS)
    right = min(width, anchor_x + FIELD_PATCH_RADIUS + 1)
    top = max(0, anchor_y - FIELD_PATCH_RADIUS)
    bottom = min(height, anchor_y + FIELD_PATCH_RADIUS + 1)
    patch = field_mask[top:bottom, left:right]
    if patch.size == 0:
        return False

    green_ratio = cv2.countNonZero(patch) / float(patch.size)
    if green_ratio >= min_green_ratio:
        return True

    # A close-up player can be cropped at the bottom of the frame, so the
    # exact anchor may still be on the player's body. Use the lower box band
    # as a fallback when there is visible grass around that body.
    if category in {"player", "referee"} and y2 >= height - FIELD_PATCH_RADIUS - 1:
        box_left = max(0, min(width - 1, x1))
        box_right = min(width, max(box_left + 1, x2 + 1))
        band_top = max(0, y2 - 40)
        bottom_band = field_mask[band_top : y2 + 1, box_left:box_right]
        if bottom_band.size:
            bottom_ratio = cv2.countNonZero(bottom_band) / float(bottom_band.size)
            return bottom_ratio >= min_green_ratio

    return False


def _draw_detection(
    frame: Any,
    cv2: Any,
    category: str,
    confidence: float,
    coordinates: tuple[int, int, int, int],
    track_id: int | None = None,
) -> None:
    x1, y1, x2, y2 = coordinates
    color = CLASS_COLORS[category]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = format_detection_label(category, confidence, track_id)
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def format_detection_label(
    category: str,
    confidence: float,
    track_id: int | None = None,
) -> str:
    """Format labels with tracker IDs for people, but never for balls."""

    identity = (
        f" #{track_id}"
        if category in TRACKED_PERSON_CATEGORIES and track_id is not None
        else ""
    )
    return f"{DISPLAY_NAMES[category]}{identity} {confidence:.2f}"


def detect_video(
    source: str | Path,
    output: str | Path,
    csv_output: str | Path | None = None,
    model_path: str = "yolo11n.pt",
    player_confidence: float = 0.5,
    ball_confidence: float = 0.18,
    min_field_green_ratio: float = 0.25,
    image_size: int = 960,
    team_a_color: str = "red",
    team_b_color: str = "white",
    referee_color: str = "black",
    team_a_goalkeeper_color: str | None = None,
    team_b_goalkeeper_color: str | None = "blue",
) -> Path:
    """Track and classify field people, detect balls, and write video and CSV."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    source_path = Path(source)
    output_path = Path(output)
    csv_output_path = (
        Path(csv_output) if csv_output is not None else output_path.with_suffix(".csv")
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    source_fps = choose_output_fps(float(capture.get(cv2.CAP_PROP_FPS)))
    capture.release()

    tracking_model = YOLO(model_path)
    people_class_ids, ball_class_ids = select_model_class_ids(tracking_model.names)
    if not people_class_ids:
        raise RuntimeError("Model has no person, player, or referee class")

    ball_model = YOLO(model_path) if ball_class_ids else None
    kit_classifier = KitColorClassifier(
        team_a_color=team_a_color,
        team_b_color=team_b_color,
        referee_color=referee_color,
        team_a_goalkeeper_color=team_a_goalkeeper_color,
        team_b_goalkeeper_color=team_b_goalkeeper_color,
    )
    role_voter = TrackRoleVoter()
    # ByteTrack uses weak detections to recover existing people; only the
    # user-selected player confidence is used when drawing their boxes.
    results = tracking_model.track(
        source=str(source_path),
        conf=min(player_confidence, 0.1),
        imgsz=image_size,
        classes=people_class_ids,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    writer = None
    csv_file = None
    frames_written = 0
    try:
        csv_file = csv_output_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "time",
                "track_id",
                "role",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ],
        )
        csv_writer.writeheader()

        for frame_index, result in enumerate(results):
            frame = result.orig_img
            if frame is None:
                continue
            analysis_frame = frame.copy()

            field_mask = build_field_mask(analysis_frame, cv2)

            ball_result = None
            if ball_model is not None:
                ball_result = ball_model.predict(
                    source=analysis_frame,
                    conf=ball_confidence,
                    imgsz=image_size,
                    classes=ball_class_ids,
                    verbose=False,
                )[0]

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {output_path}")

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                model_label = str(result.names[class_id])
                category = classify_model_label(model_label)
                if category not in {"player", "referee"}:
                    continue

                confidence_value = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                if confidence_value < player_confidence:
                    continue
                if not is_detection_on_field(
                    field_mask,
                    category,
                    (x1, y1, x2, y2),
                    cv2,
                    min_green_ratio=min_field_green_ratio,
                ):
                    continue

                track_id = None
                if category in {"player", "referee"} and box.id is not None:
                    track_id = int(box.id[0].item())

                role_prediction = kit_classifier.predict(
                    analysis_frame,
                    (x1, y1, x2, y2),
                )
                role = role_voter.update(track_id, role_prediction, frame_index)

                _draw_detection(
                    frame,
                    cv2,
                    role,
                    confidence_value,
                    (x1, y1, x2, y2),
                    track_id=track_id,
                )
                csv_writer.writerow(
                    {
                        "frame": frame_index,
                        "time": f"{frame_index / source_fps:.3f}",
                        "track_id": "" if track_id is None else track_id,
                        "role": role,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": f"{confidence_value:.4f}",
                    }
                )

            if ball_result is not None:
                for box in ball_result.boxes:
                    confidence_value = float(box.conf[0].item())
                    x1, y1, x2, y2 = [
                        int(value) for value in box.xyxy[0].tolist()
                    ]
                    if not is_detection_on_field(
                        field_mask,
                        "ball",
                        (x1, y1, x2, y2),
                        cv2,
                        min_green_ratio=min_field_green_ratio,
                    ):
                        continue

                    _draw_detection(
                        frame,
                        cv2,
                        "ball",
                        confidence_value,
                        (x1, y1, x2, y2),
                    )
                    csv_writer.writerow(
                        {
                            "frame": frame_index,
                            "time": f"{frame_index / source_fps:.3f}",
                            "track_id": "",
                            "role": "ball",
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": f"{confidence_value:.4f}",
                        }
                    )

            writer.write(frame)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    if frames_written == 0:
        raise RuntimeError("No video frames were written")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/soccervideo_cfr.mp4")
    parser.add_argument("--output", default="outputs/v04_team_classification.mp4")
    parser.add_argument("--csv-output", default="outputs/v04_detections.csv")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--player-conf", type=float, default=0.5)
    parser.add_argument("--ball-conf", type=float, default=0.18)
    parser.add_argument("--min-field-green-ratio", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    color_choices = sorted(SUPPORTED_KIT_COLORS)
    parser.add_argument("--team-a-color", choices=color_choices, default="red")
    parser.add_argument("--team-b-color", choices=color_choices, default="white")
    parser.add_argument("--referee-color", choices=color_choices, default="black")
    optional_color_choices = ["none", *color_choices]
    parser.add_argument(
        "--team-a-goalkeeper-color",
        choices=optional_color_choices,
        default="none",
    )
    parser.add_argument(
        "--team-b-goalkeeper-color",
        choices=optional_color_choices,
        default="blue",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = detect_video(
        source=args.source,
        output=args.output,
        csv_output=args.csv_output,
        model_path=args.model,
        player_confidence=args.player_conf,
        ball_confidence=args.ball_conf,
        min_field_green_ratio=args.min_field_green_ratio,
        image_size=args.imgsz,
        team_a_color=args.team_a_color,
        team_b_color=args.team_b_color,
        referee_color=args.referee_color,
        team_a_goalkeeper_color=(
            None
            if args.team_a_goalkeeper_color == "none"
            else args.team_a_goalkeeper_color
        ),
        team_b_goalkeeper_color=(
            None
            if args.team_b_goalkeeper_color == "none"
            else args.team_b_goalkeeper_color
        ),
    )
    print(f"Saved annotated video to {result_path}")
    print(f"Saved detection data to {args.csv_output}")
