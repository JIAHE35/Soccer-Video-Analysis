"""Map existing V0.5 detections onto a manually calibrated local pitch region."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

if __package__:
    from .homography import PitchCalibration, object_anchor
    from .pitch import ROLE_COLORS, draw_pitch_panel
else:
    from homography import PitchCalibration, object_anchor
    from pitch import ROLE_COLORS, draw_pitch_panel


ROOT = Path(__file__).resolve().parents[1]
DETECTION_FIELDS = ["frame", "time", "track_id", "role", "x1", "y1", "x2", "y2", "confidence", "ball_state"]
OUTPUT_FIELDS = DETECTION_FIELDS + ["source_frame", "source_time", "shot_id", "anchor_x", "anchor_y",
                                    "pitch_x", "pitch_y", "homography_valid", "mapping_reason", "position_basis"]


def map_detection(row: dict, calibration: PitchCalibration, frame: int) -> dict:
    anchor = object_anchor(row["role"], [row[key] for key in ("x1", "y1", "x2", "y2")])
    result = calibration.map_point(frame, anchor)
    source_frame = calibration.source_start_frame + frame
    output = dict(row)
    basis = "person_foot_point"
    if row["role"] == "ball":
        basis = "predicted_ground_plane_assumption" if row["ball_state"] == "predicted" else "ground_plane_assumption"
    output.update(frame=frame, time=f"{frame / calibration.fps:.6f}",
                  source_frame=source_frame, source_time=f"{source_frame / calibration.fps:.6f}",
                  shot_id=calibration.shot_id, anchor_x=f"{anchor[0]:.3f}", anchor_y=f"{anchor[1]:.3f}",
                  pitch_x=f"{result.point[0]:.4f}" if result.valid else "",
                  pitch_y=f"{result.point[1]:.4f}" if result.valid else "",
                  homography_valid=str(result.valid).lower(), mapping_reason=result.reason, position_basis=basis)
    return output


def read_detections(path: Path, calibration: PitchCalibration, offset: int) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != DETECTION_FIELDS:
            raise ValueError("Expected the V0.5 detection CSV schema")
        for row in reader:
            frame = int(row["frame"])
            local = frame - offset
            if not 0 <= local < calibration.frame_count:
                continue
            if abs(float(row["time"]) - frame / calibration.fps) > .002:
                raise ValueError(f"Detection timestamp mismatch at frame {frame}")
            bbox = np.asarray([row[key] for key in ("x1", "y1", "x2", "y2")], dtype=float)
            if not np.isfinite(bbox).all():
                raise ValueError(f"Non-finite detection at frame {frame}")
            if row["role"] not in ROLE_COLORS:
                raise ValueError(f"Unknown role: {row['role']}")
            if row["confidence"] and not 0 <= float(row["confidence"]) <= 1:
                raise ValueError("Detection confidence must be between 0 and 1")
            if row["role"] == "ball" and (row["ball_state"] not in ("detected", "predicted") or row["track_id"]):
                raise ValueError("Ball must have a state and no track ID")
            grouped[local].append(map_detection(row, calibration, local))
    if not grouped:
        raise ValueError("No detections in selected range; check the CSV frame offset")
    return grouped


def validate_video(capture, calibration: PitchCalibration) -> None:
    if not capture.isOpened():
        raise ValueError("Cannot open input video")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (size != calibration.image_size or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != calibration.frame_count
            or abs(capture.get(cv2.CAP_PROP_FPS) - calibration.fps) > .01):
        raise ValueError("Video dimensions, frame count or FPS do not match calibration")


def draw_detections(frame: np.ndarray, rows: list[dict]) -> np.ndarray:
    frame = frame.copy()
    height, width = frame.shape[:2]
    for row in rows:
        x1, y1, x2, y2 = [int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2")]
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        color = ROLE_COLORS[row["role"]]
        if row["ball_state"] == "predicted":
            for start, end in (((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)),
                               ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))):
                length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
                for i in range(0, length, 8):
                    a = np.asarray(start) + (np.asarray(end) - start) * (i / max(length, 1))
                    b = np.asarray(start) + (np.asarray(end) - start) * (min(i + 4, length) / max(length, 1))
                    cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = row["role"].replace("team_a", "A").replace("team_b", "B").replace("goalkeeper", "GK")
        if row["track_id"]:
            label += f" #{row['track_id']}"
        if row["ball_state"] == "predicted":
            label += " predicted"
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0][0]
        cv2.putText(frame, label, (int(min(x1, width - text_width - 2)), int(max(14, y1 - 5))),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "V0.6 | V0.5 detections + local calibration", (130, 575),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(source: Path, detections: Path, config_path: Path, output: Path,
        offset: int | None = None, overwrite: bool = False) -> dict:
    calibration = PitchCalibration.load(config_path)
    csv_path, report_path = output.with_suffix(".csv"), output.with_suffix(".json")
    targets = (output, csv_path, report_path)
    if len({p.resolve() for p in targets + (source, detections, config_path)}) != 6:
        raise ValueError("Output must not replace an input file")
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Output exists; choose a new filename or pass --overwrite")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to encode the result as H.264 MP4")
    grouped = read_detections(detections, calibration,
                              calibration.source_start_frame if offset is None else offset)
    capture = cv2.VideoCapture(str(source))
    try:
        validate_video(capture, calibration)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v06-", dir=output.parent) as temp:
            temp = Path(temp)
            raw_video = temp / "raw.mp4"
            width, height = calibration.image_size
            writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     calibration.fps, (width + 520, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")
            rows = []
            try:
                for frame_number in range(calibration.frame_count):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended early at frame {frame_number}")
                    frame_rows = grouped.get(frame_number, [])
                    rows.extend(frame_rows)
                    panel = draw_pitch_panel(height, 520, frame_rows, calibration.roi,
                                             frame_number / calibration.fps,
                                             (calibration.source_start_frame + frame_number) / calibration.fps)
                    writer.write(np.hstack((draw_detections(frame, frame_rows), panel)))
            finally:
                writer.release()
            encoded = temp / "encoded.mp4"
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", str(raw_video), "-an",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)], check=True)
            result_csv = temp / "coordinates.csv"
            with result_csv.open("w", newline="", encoding="utf-8") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            counts = Counter(row["mapping_reason"] for row in rows)
            report = {
                "version": "0.6", "shot_id": calibration.shot_id,
                "frame_count": calibration.frame_count, "fps": calibration.fps,
                "source_start_frame": calibration.source_start_frame,
                "source_end_frame_exclusive": calibration.source_start_frame + calibration.frame_count,
                "detections_frame_offset": calibration.source_start_frame if offset is None else offset,
                "rows": len(rows), "mapping_counts": dict(counts),
                "ball_states": dict(Counter(row["ball_state"] for row in rows if row["role"] == "ball")),
                "calibration_fit_rmse_px": dict(zip(map(str, calibration.frames), calibration.reprojection_errors)),
                "sha256": {"clip": file_hash(source), "detections": file_hash(detections), "calibration": file_hash(config_path)},
                "limitations": ["Manual local calibration for one continuous shot only",
                                "Nominal 105 x 68 m pitch; fit residual is not ground-truth accuracy",
                                "Linear reference-point interpolation assumes smooth camera motion",
                                "Ball centers, including airborne balls, project to the ground plane; not true ground positions",
                                "Predicted balls inherit V0.5 uncertainty; no new YOLO inference"],
            }
            result_report = temp / "report.json"
            result_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for temporary, target in zip((encoded, result_csv, result_report), targets):
                temporary.replace(target)
    finally:
        capture.release()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/v06_homography_clip.mp4")
    parser.add_argument("--detections", type=Path, default=ROOT / "outputs/v05_detections.csv")
    parser.add_argument("--calibration", type=Path, default=ROOT / "config/v06_pitch_calibration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v06_pitch_mapping.mp4")
    parser.add_argument("--detections-frame-offset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run(args.source, args.detections, args.calibration, args.output,
                 args.detections_frame_offset, args.overwrite)
    print(f"Saved {args.output}\nMapped {report['mapping_counts'].get('inside_local_region', 0)} / {report['rows']} rows")


if __name__ == "__main__":
    main()
