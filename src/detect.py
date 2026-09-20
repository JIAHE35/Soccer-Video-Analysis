"""Run a small YOLO video-detection pipeline for the soccer project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# OpenCV uses BGR colors. Referee is reserved for the future custom model.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (0, 255, 0),
    "referee": (0, 0, 255),
    "ball": (255, 0, 0),
}

MODEL_LABEL_TO_PROJECT_LABEL = {
    "person": "player",
    "sports ball": "ball",
}


def classify_model_label(model_label: str) -> str | None:
    """Map a pretrained COCO label to the current project label."""

    return MODEL_LABEL_TO_PROJECT_LABEL.get(model_label)


def choose_output_fps(source_fps: float, fallback: float = 30.0) -> float:
    """Keep the source playback speed, with a fallback for missing metadata."""

    return source_fps if source_fps > 0 else fallback


def _draw_detection(
    frame: Any,
    cv2: Any,
    category: str,
    confidence: float,
    coordinates: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = coordinates
    color = CLASS_COLORS[category]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{category} {confidence:.2f}"
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


def detect_video(
    source: str | Path,
    output: str | Path,
    model_path: str = "yolo11n.pt",
    confidence: float = 0.25,
    image_size: int = 960,
) -> Path:
    """Detect people and balls, then write an annotated MP4 video."""

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
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    source_fps = choose_output_fps(float(capture.get(cv2.CAP_PROP_FPS)))
    capture.release()

    model = YOLO(model_path)
    results = model.predict(
        source=str(source_path),
        conf=confidence,
        imgsz=image_size,
        stream=True,
        verbose=False,
    )

    writer = None
    frames_written = 0
    try:
        for result in results:
            frame = result.orig_img
            if frame is None:
                continue

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
                if category is None:
                    continue

                confidence_value = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                _draw_detection(
                    frame,
                    cv2,
                    category,
                    confidence_value,
                    (x1, y1, x2, y2),
                )

            writer.write(frame)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise RuntimeError("No video frames were written")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/soccervideo_cfr.mp4")
    parser.add_argument("--output", default="outputs/detected_cfr.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = detect_video(
        source=args.source,
        output=args.output,
        model_path=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
    )
    print(f"Saved annotated video to {result_path}")
