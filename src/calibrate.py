"""Review or edit named pitch landmarks on keyframes of the V0.6 clip."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .homography import LANDMARKS, PitchCalibration
    from .pitch import draw_calibration_overlay
else:
    from homography import LANDMARKS, PitchCalibration
    from pitch import draw_calibration_overlay

ROOT = Path(__file__).resolve().parents[1]


def get_frame(capture, number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, number)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot read frame {number}")
    return frame


def review(source: Path, config_path: Path, output: Path) -> None:
    calibration = PitchCalibration.load(config_path)
    capture = cv2.VideoCapture(str(source))
    samples = sorted(set(calibration.frames + [(a + b) // 2 for a, b in zip(calibration.frames, calibration.frames[1:])]))
    tiles = []
    try:
        for number in samples:
            frame = draw_calibration_overlay(get_frame(capture, number), calibration.matrix_at(number), calibration.roi)
            item = next((item for item in calibration.keyframes if item["frame"] == number), None)
            if item:
                for point in item["points"].values():
                    cv2.circle(frame, tuple(map(int, point)), 4, (30, 40, 255), -1)
            tile = cv2.resize(frame, (640, 294))
            label = f"Frame {number} | {number / calibration.fps:.2f}s | {'keyframe' if item else 'interpolated'}"
            cv2.rectangle(tile, (0, 0), (640, 25), (20, 25, 20), -1)
            cv2.putText(tile, label, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(tile)
    finally:
        capture.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError("Cannot save calibration review")
    print(f"Saved {output}; red dots are manual landmarks, cyan lines are projected geometry")


def annotate(source: Path, config_path: Path, number: int, names: list[str]) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if len(names) < 4 or len(set(names)) != len(names) or any(name not in LANDMARKS for name in names):
        raise ValueError("Select at least four unique known landmarks")
    if not 0 <= number < config["frame_count"]:
        raise ValueError("Frame is outside the clip")
    capture = cv2.VideoCapture(str(source))
    try:
        original = get_frame(capture, number)
    finally:
        capture.release()
    window = "V0.6 landmark calibration"
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(names):
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    print("Click landmarks in this order:")
    for name in names:
        print(f"  {name}: nominal pitch coordinate {LANDMARKS[name]}")
    print("Right-click: undo. S: validate and save. Escape: cancel without saving.")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = original.copy()
            for i, point in enumerate(points):
                cv2.circle(canvas, tuple(point), 4, (0, 0, 255), -1)
                cv2.putText(canvas, names[i], (point[0] + 5, max(15, point[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
            next_name = names[len(points)] if len(points) < len(names) else "Complete; S to save"
            cv2.rectangle(canvas, (0, 0), (800, 28), (15, 15, 15), -1)
            cv2.putText(canvas, f"Next: {next_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key == ord("s") and len(points) == len(names):
                updated = deepcopy(config)
                updated["keyframes"] = [item for item in config["keyframes"] if item["frame"] != number]
                updated["keyframes"].append({"frame": number, "points": dict(zip(names, points))})
                updated["keyframes"].sort(key=lambda item: item["frame"])
                try:
                    calibration = PitchCalibration(updated)
                except ValueError as error:
                    print(f"Not saved: {error}")
                    continue
                index = calibration.frames.index(number)
                print(f"Fit RMSE: {calibration.reprojection_errors[index]:.2f}px (not independent accuracy)")
                config_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
                print(f"Saved {config_path}. Run --review before regenerating the mapping.")
                return
    finally:
        cv2.destroyWindow(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/v06_homography_clip.mp4")
    parser.add_argument("--calibration", type=Path, default=ROOT / "config/v06_pitch_calibration.json")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v06_calibration_review.jpg")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--landmarks", nargs="+", choices=LANDMARKS)
    args = parser.parse_args()
    if args.review:
        review(args.source, args.calibration, args.output)
    elif args.frame is not None and args.landmarks:
        annotate(args.source, args.calibration, args.frame, args.landmarks)
    else:
        parser.error("Use --review, or --frame N --landmarks NAME NAME NAME NAME ...")


if __name__ == "__main__":
    main()
