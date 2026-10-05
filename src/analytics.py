"""V0.7 local trajectories, observed-position heatmaps, and data-quality summary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import cv2
import numpy as np

if __package__:
    from .homography import PitchCalibration
    from .map_pitch import OUTPUT_FIELDS, ROOT, draw_detections, file_hash, validate_video
    from .pitch import ROLE_COLORS, draw_pitch_panel, pitch_lines, pitch_to_pixels
else:
    from homography import PitchCalibration
    from map_pitch import OUTPUT_FIELDS, ROOT, draw_detections, file_hash, validate_video
    from pitch import ROLE_COLORS, draw_pitch_panel, pitch_lines, pitch_to_pixels


TEAM_ROLES = {"team_a": "team_a", "team_a_goalkeeper": "team_a",
              "team_b": "team_b", "team_b_goalkeeper": "team_b"}


class LocalAnalytics:
    """Keep contiguous track fragments; never infer identities or fill gaps."""

    def __init__(self, fps: float, frame_count: int, roi, trail_seconds: float = 2,
                 max_person_step_m: float = 1.5, max_ball_step_m: float = 5) -> None:
        if not all(math.isfinite(value) and value > 0 for value in
                   (fps, frame_count, trail_seconds, max_person_step_m, max_ball_step_m)):
            raise ValueError("Analysis settings must be finite and positive")
        self.fps, self.frame_count = fps, frame_count
        self.roi = np.asarray(roi, np.float32)
        self.window = max(1, int(math.ceil(trail_seconds * fps)))
        self.max_person_step_m, self.max_ball_step_m = max_person_step_m, max_ball_step_m
        self.frame = -1
        self.active = {}
        self.segments = defaultdict(list)
        self.heatmaps = {team: np.zeros((68, 105), np.float64) for team in ("team_a", "team_b")}
        self.mapping_counts, self.excluded_counts, self.breaks = Counter(), Counter(), Counter()
        self.ball_frames = Counter({"detected": 0, "predicted": 0, "missing": 0})
        self.team_samples, self.team_frames = Counter(), Counter()
        self.frame_statistics = []
        self.row_count = 0

    def update(self, frame: int, rows: list[dict]) -> None:
        if frame != self.frame + 1 or frame >= self.frame_count:
            raise ValueError("Process every frame sequentially, including empty frames")
        self.frame = frame
        duplicates = Counter(row["track_id"] for row in rows if row["role"] != "ball" and row["track_id"])
        self.mapping_counts.update(row["mapping_reason"] for row in rows)
        self.row_count += len(rows)
        balls = [row for row in rows if row["role"] == "ball"]
        if len(balls) > 1:
            raise ValueError("More than one ball result in a frame")
        self.ball_frames[balls[0]["ball_state"] if balls else "missing"] += 1
        present, teams_present = set(), set()
        frame_teams = Counter()
        duplicate_rows = 0
        for row in rows:
            role, track_id = row["role"], row["track_id"]
            if role != "ball" and track_id and duplicates[track_id] > 1:
                self.excluded_counts["duplicate_track_id_rows"] += 1
                duplicate_rows += 1
                continue
            if row["homography_valid"] != "true":
                continue
            x, y = float(row["pitch_x"]), float(row["pitch_y"])
            if not np.isfinite([x, y]).all() or cv2.pointPolygonTest(self.roi, (x, y), False) < 0:
                raise ValueError("Valid coordinates must lie inside the local region")
            team = TEAM_ROLES.get(role)
            if team:
                self.heatmaps[team][min(67, int(y)), min(104, int(x))] += 1 / self.fps
                self.team_samples[team] += 1
                frame_teams[team] += 1
                teams_present.add(team)
            if role != "ball" and not track_id:
                self.excluded_counts["untracked_person_rows_for_trajectories"] += 1
                continue
            key = "__ball__" if role == "ball" else track_id
            present.add(key)
            sample = {"frame": frame, "x": x, "y": y, "state": row["ball_state"]}
            previous = self.active.get(key)
            reason = "first_observation"
            if previous:
                last = previous["samples"][-1]
                limit = self.max_ball_step_m if role == "ball" else self.max_person_step_m
                if previous["role"] != role:
                    reason = "role_change"
                elif math.hypot(x - last["x"], y - last["y"]) > limit:
                    reason = "position_jump"
                else:
                    previous["samples"].append(sample)
                    continue
                self.breaks[reason] += 1
            elif self.segments[key]:
                reason = "after_gap_or_invalid"
            segment = {"track_id": track_id, "role": role, "start_reason": reason, "samples": [sample]}
            self.segments[key].append(segment)
            self.active[key] = segment
        for key in set(self.active) - present:
            self.active.pop(key)
            self.breaks["gap_invalid_or_duplicate"] += 1
        self.team_frames.update(teams_present)
        self.frame_statistics.append({"frame": frame, "valid_mapped_rows": sum(row["homography_valid"] == "true" for row in rows),
                                      "team_a_accepted": frame_teams["team_a"], "team_b_accepted": frame_teams["team_b"],
                                      "duplicate_id_rows": duplicate_rows,
                                      "ball_state": balls[0]["ball_state"] if balls else "missing"})

    def trails(self) -> list[dict]:
        return [{"track_id": item["track_id"], "role": item["role"],
                 "points": [point for point in item["samples"] if point["frame"] > self.frame - self.window]}
                for item in self.active.values()]

    def summary(self) -> dict:
        def describe(item):
            samples = item["samples"]
            return {"role": item["role"], "start_frame": samples[0]["frame"], "end_frame": samples[-1]["frame"],
                    "sample_count": len(samples), "observation_seconds": len(samples) / self.fps,
                    "start_reason": item["start_reason"]}
        return {"version": "0.7", "fps": self.fps, "expected_frames": self.frame_count,
                "processed_frames": self.frame + 1, "duration_seconds": (self.frame + 1) / self.fps,
                "input_rows": self.row_count, "mapping_counts": dict(self.mapping_counts),
                "excluded_counts": dict(self.excluded_counts), "trajectory_breaks": dict(self.breaks),
                "ball_frames": dict(self.ball_frames),
                "ball_frame_fractions": {state: count / max(1, self.frame + 1) for state, count in self.ball_frames.items()},
                "tracks": {key: [describe(item) for item in items] for key, items in self.segments.items() if key != "__ball__"},
                "ball_segments": [{**describe(item), "samples": item["samples"]} for item in self.segments["__ball__"]],
                "teams": {team: {"accepted_samples": self.team_samples[team],
                                  "person_observation_seconds": self.team_samples[team] / self.fps,
                                  "frames_with_member_seconds": self.team_frames[team] / self.fps,
                                  "max_cell_person_seconds": float(grid.max())}
                          for team, grid in self.heatmaps.items()},
                "frame_statistics": self.frame_statistics}


def read_mapping(path: Path, calibration: PitchCalibration) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError("Expected the V0.6 mapping CSV schema")
        for row in reader:
            frame = int(row["frame"])
            if not 0 <= frame < calibration.frame_count or row["shot_id"] != calibration.shot_id:
                raise ValueError("Mapping frame or shot does not match calibration")
            source_frame = frame + calibration.source_start_frame
            times = [float(row["time"]), float(row["source_time"])]
            if (not np.isfinite(times).all() or int(row["source_frame"]) != source_frame
                    or abs(times[0] - frame / calibration.fps) > .002
                    or abs(times[1] - source_frame / calibration.fps) > .002):
                raise ValueError("Mapping timestamp or source frame mismatch")
            bbox = np.asarray([row[key] for key in ("x1", "y1", "x2", "y2")], float)
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Invalid bounding box")
            if row["role"] not in ROLE_COLORS or row["homography_valid"] not in ("true", "false"):
                raise ValueError("Invalid role or mapping validity flag")
            if row["confidence"] and not 0 <= float(row["confidence"]) <= 1:
                raise ValueError("Invalid detection confidence")
            if row["role"] == "ball":
                if row["track_id"] or row["ball_state"] not in ("detected", "predicted"):
                    raise ValueError("Ball must have a state but no ID")
                if row["ball_state"] == "predicted" and row["confidence"]:
                    raise ValueError("Predicted ball cannot have a detection confidence")
                if any(item["role"] == "ball" for item in grouped[frame]):
                    raise ValueError("Multiple ball rows in one frame")
            elif row["ball_state"]:
                raise ValueError("Person cannot have a ball state")
            if row["homography_valid"] == "true":
                point = tuple(map(float, (row["pitch_x"], row["pitch_y"])))
                if (not np.isfinite(point).all() or cv2.pointPolygonTest(calibration.roi, point, False) < 0
                        or row["mapping_reason"] != "inside_local_region"):
                    raise ValueError("Valid mapping must be within the local region")
            elif row["pitch_x"] or row["pitch_y"]:
                raise ValueError("Invalid mapping must have empty pitch coordinates")
            grouped[frame].append(row)
    if not grouped:
        raise ValueError("Mapping CSV contains no detections")
    return grouped


def draw_heatmaps(analysis: LocalAnalytics) -> np.ndarray:
    height, width = 760, 600
    scale = min((height - 174) / 68, (width - 64) / 55)
    origin_x = int((width - 52.5 * scale) / 2)
    image_y, image_x = np.indices((height, width))
    pitch_x = (image_x - origin_x) / scale + 52.5
    pitch_y = (image_y - 80) / scale
    x_bins = np.clip(np.floor(pitch_x).astype(int), 0, 104)
    y_bins = np.clip(np.floor(pitch_y).astype(int), 0, 67)
    roi_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(roi_mask, pitch_to_pixels(analysis.roi, height, width), 255)
    maximum = max(float(grid.max()) for grid in analysis.heatmaps.values())
    panels = []
    for team, grid in analysis.heatmaps.items():
        name = "TEAM A" if team == "team_a" else "TEAM B"
        panel = draw_pitch_panel(height, width, [], analysis.roi, 0, 0, title=f"V0.7 | {name} OBSERVED POSITIONS",
                                 subtitle="Right penalty area | Same clip and shared color scale")
        values = grid[y_bins, x_bins]
        intensities = np.round(values / max(maximum, 1e-9) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(intensities, cv2.COLORMAP_TURBO)
        mask = (roi_mask > 0) & (values > 0)
        panel[mask] = (.2 * panel[mask] + .8 * colors[mask]).astype(np.uint8)
        for line in pitch_lines():
            cv2.polylines(panel, [pitch_to_pixels(line, height, width)], False, (190, 208, 193), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (0, height - 88), (width, height), (28, 33, 31), -1)
        seconds = analysis.team_samples[team] / analysis.fps
        cv2.putText(panel, f"{seconds:.2f} person-s | Goalkeepers included", (24, 686), cv2.FONT_HERSHEY_SIMPLEX, .48, (215, 228, 219), 1, cv2.LINE_AA)
        cv2.putText(panel, "Person-s per nominal 1 m cell; not possession", (24, 708), cv2.FONT_HERSHEY_SIMPLEX, .43, (185, 200, 190), 1, cv2.LINE_AA)
        gradient = cv2.applyColorMap(np.tile(np.arange(256, dtype=np.uint8), (10, 1)), cv2.COLORMAP_TURBO)
        panel[721:731, 150:490] = cv2.resize(gradient, (340, 10))
        cv2.putText(panel, "0", (125, 731), cv2.FONT_HERSHEY_SIMPLEX, .42, (215, 228, 219), 1)
        cv2.putText(panel, f"{maximum:.2f}", (501, 731), cv2.FONT_HERSHEY_SIMPLEX, .42, (215, 228, 219), 1)
        cv2.putText(panel, "Blank outside ROI / no accepted sample; no smoothing", (24, 752), cv2.FONT_HERSHEY_SIMPLEX, .4, (172, 186, 178), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def check_targets(targets, overwrite: bool) -> None:
    for target in targets:
        if target.is_symlink():
            raise ValueError(f"Refusing to replace output symlink: {target}")
        if target.exists() and not target.is_file():
            raise IsADirectoryError(f"Output is not a regular file: {target}")
        if not overwrite and target.exists():
            raise FileExistsError("V0.7 output exists; choose another --output-dir or use --overwrite")


def publish_outputs(sources, targets, overwrite: bool = False) -> None:
    """No-clobber links by default; restore our replacements on publication failure."""
    check_targets(targets, overwrite)
    backups, installed, recovery_problems = {}, [], []
    try:
        if overwrite:
            for target in targets:
                if target.exists():
                    backup = target.with_name(f".v07-recovery-{uuid.uuid4().hex}-{target.name}")
                    os.link(target, backup)
                    backups[target] = backup
        for source, target in zip(sources, targets):
            identity = source.stat()
            if overwrite:
                os.replace(source, target)
            else:
                # Same-filesystem link creation atomically fails if a target appeared meanwhile.
                os.link(source, target)
            installed.append((target, (identity.st_dev, identity.st_ino)))
    except BaseException:
        for target, identity in reversed(installed):
            try:
                current = target.lstat() if target.exists() or target.is_symlink() else None
                if current is None or (current.st_dev, current.st_ino) != identity:
                    if target in backups:
                        recovery_problems.append(target)
                    continue
                if target in backups:
                    os.replace(backups[target], target)
                else:
                    target.unlink()
            except OSError:
                recovery_problems.append(target)
        if recovery_problems:
            paths = [str(path) for path in backups.values() if path.exists()]
            raise RuntimeError(f"Output restoration incomplete; recovery files preserved: {paths}")
        raise
    finally:
        if not recovery_problems:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def run(source: Path, csv_path: Path, config_path: Path, output_dir: Path, *,
        mapping_report: Path | None = None, trail_seconds: float = 2,
        max_person_step_m: float = 1.5, max_ball_step_m: float = 5, overwrite: bool = False) -> dict:
    calibration = PitchCalibration.load(config_path)
    mapping_report = mapping_report or csv_path.with_suffix(".json")
    report = json.loads(mapping_report.read_text(encoding="utf-8"))
    hashes = {"clip": file_hash(source), "calibration": file_hash(config_path),
              "mapping_csv": file_hash(csv_path), "mapping_report": file_hash(mapping_report)}
    if any(report["sha256"][key] != hashes[key] for key in ("clip", "calibration")):
        raise ValueError("Input hash does not match the V0.6 report; regenerate V0.6 mapping first")
    if (report["frame_count"] != calibration.frame_count or report["fps"] != calibration.fps
            or report["source_start_frame"] != calibration.source_start_frame or report["shot_id"] != calibration.shot_id):
        raise ValueError("Mapping report metadata does not match calibration")
    targets = tuple(output_dir / name for name in ("v07_local_trajectories.mp4", "v07_team_heatmaps.png", "v07_summary.json"))
    if any(target.resolve() in {p.resolve() for p in (source, csv_path, config_path, mapping_report)} for target in targets):
        raise ValueError("Output must not overwrite an input file")
    check_targets(targets, overwrite)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required")
    grouped = read_mapping(csv_path, calibration)
    analysis = LocalAnalytics(calibration.fps, calibration.frame_count, calibration.roi,
                              trail_seconds, max_person_step_m, max_ball_step_m)
    capture = cv2.VideoCapture(str(source))
    try:
        validate_video(capture, calibration)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v07-", dir=output_dir) as temp:
            temp = Path(temp)
            width, height = calibration.image_size
            raw_video = temp / "raw.mp4"
            writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), calibration.fps, (width + 520, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create trajectory video")
            try:
                for number in range(calibration.frame_count):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended early at frame {number}")
                    rows = grouped.get(number, [])
                    analysis.update(number, rows)
                    panel = draw_pitch_panel(height, 520, rows, calibration.roi, number / calibration.fps,
                                             (number + calibration.source_start_frame) / calibration.fps,
                                             trails=analysis.trails(), title="V0.7 | LOCAL TRAJECTORIES")
                    video_frame = draw_detections(frame, rows, "V0.7 | Local tracks; ball is a ground-plane projection")
                    writer.write(np.hstack((video_frame, panel)))
            finally:
                writer.release()
            encoded = temp / "encoded.mp4"
            subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-i", str(raw_video), "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)], check=True)
            heatmap_path = temp / "heatmaps.png"
            if not cv2.imwrite(str(heatmap_path), draw_heatmaps(analysis)):
                raise RuntimeError("Cannot save heatmaps")
            summary = analysis.summary()
            summary.update(shot_id=calibration.shot_id, source_start_frame=calibration.source_start_frame,
                           sha256=hashes, settings={"trail_seconds": trail_seconds, "max_person_step_m": max_person_step_m,
                                                    "max_ball_step_m": max_ball_step_m, "heatmap_cell_size_m": 1,
                                                    "include_goalkeepers": True, "include_untracked_team_people_in_heatmap": True,
                                                    "exclude_duplicate_ids": True},
                           valid_pitch_polygon=calibration.roi.tolist(), limitations=[
                               "Exploratory local position distribution, not ground-truth tactical statistics",
                               "Nominal pitch coordinates; no reliable physical distance, speed, or possession estimates",
                               "Temporary IDs can switch even without a large position jump; no identity recovery",
                               "Missing, invalid, duplicate-ID and role-change intervals break trajectories",
                               "Heatmap totals are accepted person-seconds, not unique-player time or team possession",
                               "Heatmaps include assigned goalkeepers but exclude referees, unknown roles and balls",
                               "All ball positions, including airborne balls, are ground-plane projections; predictions stay distinct",
                               "Camera/calibration and V0.5 detection errors remain; no automatic airborne or accuracy validation"])
            summary_path = temp / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            publish_outputs((encoded, heatmap_path, summary_path), targets, overwrite)
    finally:
        capture.release()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/v06_homography_clip.mp4")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs/v06_pitch_mapping.csv")
    parser.add_argument("--calibration", type=Path, default=ROOT / "config/v06_pitch_calibration.json")
    parser.add_argument("--mapping-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--trail-seconds", type=float, default=2)
    parser.add_argument("--max-person-step-m", type=float, default=1.5)
    parser.add_argument("--max-ball-step-m", type=float, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run(args.source, args.csv, args.calibration, args.output_dir, mapping_report=args.mapping_report,
                  trail_seconds=args.trail_seconds, max_person_step_m=args.max_person_step_m,
                  max_ball_step_m=args.max_ball_step_m, overwrite=args.overwrite)
    print(f"Saved V0.7 results to {args.output_dir}\nProcessed {summary['processed_frames']} frames; ball coverage: {summary['ball_frames']}")


if __name__ == "__main__":
    main()
