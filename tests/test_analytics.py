import csv
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import os

import cv2
import numpy as np

from src.homography import LANDMARKS
from src.map_pitch import OUTPUT_FIELDS, file_hash


ROI = [[88.5, 13.84], [105, 13.84], [105, 54.16], [88.5, 54.16]]


def row(frame=0, track_id="7", role="team_a", point=(94, 34), state="", valid=True):
    values = {key: "" for key in OUTPUT_FIELDS}
    values.update(frame=str(frame), time=f"{frame/10:.6f}", source_frame=str(100 + frame),
                  source_time=f"{(100+frame)/10:.6f}", shot_id="synthetic",
                  track_id=track_id, role=role, x1="370", y1="120", x2="382", y2="136",
                  confidence="" if state == "predicted" else "0.8", ball_state=state,
                  anchor_x="376", anchor_y="128" if role == "ball" else "136",
                  pitch_x=str(point[0]) if valid else "", pitch_y=str(point[1]) if valid else "",
                  homography_valid="true" if valid else "false",
                  mapping_reason="inside_local_region" if valid else "outside_local_region",
                  position_basis="person_foot_point" if role != "ball" else
                  ("predicted_ground_plane_assumption" if state == "predicted" else "ground_plane_assumption"))
    return values


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("src.analytics"), "V0.7 analysis module is not implemented")
        self.module = importlib.import_module("src.analytics")
        self.analysis = self.module.LocalAnalytics(10, 10, ROI)

    def test_consecutive_same_id_points_form_one_segment(self):
        for frame in range(3):
            self.analysis.update(frame, [row(frame, point=(94 + frame * .1, 34))])
        tracks = self.analysis.summary()["tracks"]["7"]
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["sample_count"], 3)
        self.assertAlmostEqual(tracks[0]["observation_seconds"], .3)
        self.assertEqual(len(self.analysis.trails()[0]["points"]), 3)

    def test_gap_does_not_join_segments(self):
        self.analysis.update(0, [row()])
        self.analysis.update(1, [])
        self.assertEqual(self.analysis.trails(), [])
        self.analysis.update(2, [row(2)])
        self.assertEqual(len(self.analysis.summary()["tracks"]["7"]), 2)
        self.assertEqual(len(self.analysis.trails()[0]["points"]), 1)

    def test_invalid_mapping_breaks_and_never_adds_heat(self):
        self.analysis.update(0, [row()])
        self.analysis.update(1, [row(1, valid=False)])
        self.analysis.update(2, [row(2)])
        self.assertEqual(len(self.analysis.summary()["tracks"]["7"]), 2)
        self.assertAlmostEqual(self.analysis.heatmaps["team_a"].sum(), .2)
        self.assertEqual(self.analysis.summary()["mapping_counts"]["outside_local_region"], 1)

    def test_role_change_and_new_id_never_connect(self):
        self.analysis.update(0, [row()])
        self.analysis.update(1, [row(1, role="team_b")])
        self.analysis.update(2, [row(2, track_id="8")])
        self.assertEqual(len(self.analysis.summary()["tracks"]["7"]), 2)
        self.assertEqual(self.analysis.trails()[0]["track_id"], "8")
        self.assertEqual(len(self.analysis.trails()[0]["points"]), 1)

    def test_large_jump_starts_new_segment_instead_of_a_long_line(self):
        self.analysis.update(0, [row()])
        self.analysis.update(1, [row(1, point=(99, 34))])
        self.assertEqual(len(self.analysis.summary()["tracks"]["7"]), 2)
        self.assertEqual(self.analysis.summary()["trajectory_breaks"]["position_jump"], 1)

    def test_duplicate_id_rows_are_excluded_and_break_existing_track(self):
        self.analysis.update(0, [row()])
        self.analysis.update(1, [row(1), row(1, point=(95, 34))])
        self.assertEqual(self.analysis.trails(), [])
        self.assertAlmostEqual(self.analysis.heatmaps["team_a"].sum(), .1)
        self.assertEqual(self.analysis.summary()["excluded_counts"]["duplicate_track_id_rows"], 2)
        self.analysis.update(2, [row(2)])
        self.assertEqual(len(self.analysis.summary()["tracks"]["7"]), 2)

    def test_untracked_and_goalkeeper_rows_count_heat_but_referees_do_not(self):
        self.analysis.update(0, [row(track_id=""), row(track_id="9", role="team_b_goalkeeper"),
                                 row(track_id="10", role="referee")])
        self.assertAlmostEqual(self.analysis.heatmaps["team_a"].sum(), .1)
        self.assertAlmostEqual(self.analysis.heatmaps["team_b"].sum(), .1)
        self.assertNotIn("", self.analysis.summary()["tracks"])
        self.assertEqual(len(self.analysis.trails()), 2)

    def test_team_frame_time_is_not_person_seconds(self):
        self.analysis.update(0, [row(), row(track_id="8")])
        self.analysis.update(1, [])
        team = self.analysis.summary()["teams"]["team_a"]
        self.assertAlmostEqual(team["person_observation_seconds"], .2)
        self.assertAlmostEqual(team["frames_with_member_seconds"], .1)

    def test_ball_states_count_every_frame_and_keep_predictions_distinct(self):
        self.analysis.update(0, [row(track_id="", role="ball", state="detected")])
        self.analysis.update(1, [row(1, track_id="", role="ball", point=(94.1, 34), state="predicted")])
        self.analysis.update(2, [])
        summary = self.analysis.summary()
        self.assertEqual(summary["ball_frames"], {"detected": 1, "predicted": 1, "missing": 1})
        self.assertEqual(summary["ball_segments"][0]["samples"][1]["state"], "predicted")
        self.assertAlmostEqual(sum(grid.sum() for grid in self.analysis.heatmaps.values()), 0)

    def test_trail_window_retains_only_recent_points(self):
        self.analysis = self.module.LocalAnalytics(10, 30, ROI, trail_seconds=.2)
        for frame in range(5):
            self.analysis.update(frame, [row(frame)])
        self.assertEqual([p["frame"] for p in self.analysis.trails()[0]["points"]], [3, 4])

    def test_update_requires_sequential_frames(self):
        with self.assertRaises(ValueError):
            self.analysis.update(1, [])

    def test_heatmap_preserves_boundary_points(self):
        self.analysis.update(0, [row(point=(105, 54.16))])
        self.assertAlmostEqual(self.analysis.heatmaps["team_a"].sum(), .1)

    def test_empty_heatmap_renders_without_fabricating_density(self):
        image = self.module.draw_heatmaps(self.analysis)
        self.assertEqual(image.shape, (760, 1200, 3))
        self.assertGreater(np.std(image), 5)
        self.assertEqual(self.analysis.heatmaps["team_a"].sum(), 0)


class AnalyticsPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("src.analytics"), "V0.7 analysis module is not implemented")
        self.module = importlib.import_module("src.analytics")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source, self.csv_path = self.root / "clip.mp4", self.root / "mapping.csv"
        self.config_path = self.root / "calibration.json"
        names = ("penalty_far", "penalty_near", "six_yard_far", "six_yard_near", "penalty_spot")
        points = {name: [4 * LANDMARKS[name][0], 4 * LANDMARKS[name][1]] for name in names}
        config = {"fps": 10, "frame_count": 4, "source_start_frame": 100, "image_size": [600, 600],
                  "shot_id": "synthetic", "valid_pitch_polygon": ROI,
                  "keyframes": [{"frame": 0, "points": points}, {"frame": 3, "points": points}]}
        self.config_path.write_text(json.dumps(config))
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (600, 600))
        for _ in range(4):
            writer.write(np.full((600, 600, 3), (25, 100, 30), np.uint8))
        writer.release()
        self.write_rows([row(0), row(1), row(3)])
        report = {"sha256": {"clip": file_hash(self.source), "calibration": file_hash(self.config_path)},
                  "frame_count": 4, "fps": 10, "source_start_frame": 100, "shot_id": "synthetic"}
        self.csv_path.with_suffix(".json").write_text(json.dumps(report))

    def write_rows(self, rows):
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_bad_timestamp_and_outside_valid_coordinate_rejected(self):
        from src.homography import PitchCalibration
        calibration = PitchCalibration.load(self.config_path)
        bad = row()
        bad["source_time"] = "0"
        self.write_rows([bad])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.module.read_mapping(self.csv_path, calibration)
        self.write_rows([row(point=(80, 34))])
        with self.assertRaisesRegex(ValueError, "region"):
            self.module.read_mapping(self.csv_path, calibration)

    def test_duplicate_ball_rows_rejected(self):
        from src.homography import PitchCalibration
        self.write_rows([row(role="ball", track_id="", state="detected")] * 2)
        with self.assertRaisesRegex(ValueError, "ball"):
            self.module.read_mapping(self.csv_path, PitchCalibration.load(self.config_path))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_pipeline_outputs_synced_video_heatmap_and_summary_without_overwriting(self):
        original_hash = file_hash(self.csv_path)
        output_dir = self.root / "outputs"
        summary = self.module.run(self.source, self.csv_path, self.config_path, output_dir)
        self.assertEqual(summary["processed_frames"], 4)
        self.assertEqual(summary["ball_frames"]["missing"], 4)
        self.assertEqual(len(summary["tracks"]["7"]), 2)
        video = cv2.VideoCapture(str(output_dir / "v07_local_trajectories.mp4"))
        count = 0
        while True:
            ok, image = video.read()
            if not ok:
                break
            count += 1
            self.assertEqual(image.shape, (600, 1120, 3))
        video.release()
        self.assertEqual(count, 4)
        self.assertIsNotNone(cv2.imread(str(output_dir / "v07_team_heatmaps.png")))
        exported = json.loads((output_dir / "v07_summary.json").read_text())
        self.assertEqual(exported["sha256"]["mapping_csv"], original_hash)
        self.assertEqual(file_hash(self.csv_path), original_hash)
        with self.assertRaises(FileExistsError):
            self.module.run(self.source, self.csv_path, self.config_path, output_dir)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_mismatched_calibration_provenance_rejected_before_outputs(self):
        report_path = self.csv_path.with_suffix(".json")
        report = json.loads(report_path.read_text())
        report["sha256"]["calibration"] = "wrong"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.module.run(self.source, self.csv_path, self.config_path, self.root / "outputs")
        self.assertFalse((self.root / "outputs").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_file_created_during_encoding_is_never_overwritten(self):
        output_dir = self.root / "outputs"
        raced = output_dir / "v07_local_trajectories.mp4"
        real_run = subprocess.run
        def encode_then_create(*args, **kwargs):
            result = real_run(*args, **kwargs)
            raced.write_bytes(b"another process created this file")
            return result
        with patch.object(self.module.subprocess, "run", side_effect=encode_then_create):
            with self.assertRaises(FileExistsError):
                self.module.run(self.source, self.csv_path, self.config_path, output_dir)
        self.assertEqual(raced.read_bytes(), b"another process created this file")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_directory_target_does_not_partially_replace_existing_outputs(self):
        output_dir = self.root / "outputs"
        output_dir.mkdir()
        video, summary = output_dir / "v07_local_trajectories.mp4", output_dir / "v07_summary.json"
        video.write_bytes(b"old video")
        summary.write_bytes(b"old summary")
        (output_dir / "v07_team_heatmaps.png").mkdir()
        with self.assertRaises(IsADirectoryError):
            self.module.run(self.source, self.csv_path, self.config_path, output_dir, overwrite=True)
        self.assertEqual(video.read_bytes(), b"old video")
        self.assertEqual(summary.read_bytes(), b"old summary")

    def test_mid_publication_failure_restores_the_previous_group(self):
        self.assertTrue(hasattr(self.module, "publish_outputs"), "Transactional publication is missing")
        staging, output = self.root / "staging", self.root / "outputs"
        staging.mkdir(); output.mkdir()
        sources = tuple(staging / name for name in ("video", "heatmap", "summary"))
        targets = tuple(output / name for name in ("video", "heatmap", "summary"))
        for source, target in zip(sources, targets):
            source.write_bytes(b"new " + source.name.encode())
            target.write_bytes(b"old " + target.name.encode())
        real_replace = os.replace
        def fail_heatmap(source, target):
            if Path(source) == sources[1]:
                raise OSError("simulated write failure")
            return real_replace(source, target)
        with patch.object(self.module.os, "replace", side_effect=fail_heatmap):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.module.publish_outputs(sources, targets, overwrite=True)
        for target in targets:
            self.assertEqual(target.read_bytes(), b"old " + target.name.encode())


if __name__ == "__main__":
    unittest.main()
