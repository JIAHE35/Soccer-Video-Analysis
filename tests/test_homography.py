import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.homography import LANDMARKS, PitchCalibration, fit_homography, object_anchor, transform_points
from src.map_pitch import DETECTION_FIELDS, map_detection, read_detections, run, validate_video


ROOT = Path(__file__).resolve().parents[1]


def make_config(frame_count=5, source_start=100, image_size=(600, 600)):
    names = ("penalty_far", "penalty_near", "six_yard_far", "six_yard_near", "penalty_spot")
    # A known affine projection lets tests check absolute coordinates independently.
    points = {name: [4 * LANDMARKS[name][0], 4 * LANDMARKS[name][1]] for name in names}
    moved = {name: [point[0] + 8, point[1]] for name, point in points.items()}
    return {"fps": 30, "frame_count": frame_count, "image_size": list(image_size),
            "source_start_frame": source_start, "shot_id": "synthetic",
            "valid_pitch_polygon": [[88.5, 13.84], [105, 13.84], [105, 54.16], [88.5, 54.16]],
            "keyframes": [{"frame": 0, "points": points}, {"frame": frame_count - 1, "points": moved}]}


def detection(frame=100, role="team_a", state=""):
    return dict(zip(DETECTION_FIELDS, [str(frame), f"{frame/30:.6f}", "7" if role != "ball" else "", role,
                                       "370", "125", "382", "136", "" if state == "predicted" else "0.8", state]))


class HomographyTests(unittest.TestCase):
    def test_known_perspective_mapping(self):
        image = np.array([[0, 0], [100, 0], [95, 80], [5, 80]], float)
        pitch = np.array([[0, 0], [16, 0], [16, 40], [0, 40]], float)
        np.testing.assert_allclose(transform_points(fit_homography(image, pitch), image), pitch, atol=1e-5)

    def test_reject_degenerate_landmarks(self):
        for points in ([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [0, 0], [1, 1], [0, 1]]):
            with self.assertRaises(ValueError):
                fit_homography(points, points)
        with self.assertRaises(ValueError):
            fit_homography([[0, 0]] * 3, [[0, 0]] * 3)

    def test_point_at_infinity_rejected(self):
        matrix = np.eye(3)
        matrix[2] = [1, 0, 0]
        with self.assertRaises(ValueError):
            transform_points(matrix, [[0, 4]])

    def test_all_people_use_feet_balls_use_center(self):
        for role in ("team_a", "team_b", "team_a_goalkeeper", "team_b_goalkeeper", "referee", "unknown", "player"):
            self.assertEqual(object_anchor(role, (10, 20, 30, 60)), (20, 60))
        self.assertEqual(object_anchor("ball", (10, 20, 30, 60)), (20, 40))
        for box in ((1, 2, 1, 3), (1, 2, 3, float("nan"))):
            with self.assertRaises(ValueError):
                object_anchor("ball", box)

    def test_interpolation_compensates_camera_translation(self):
        calibration = PitchCalibration(make_config())
        result = calibration.map_point(2, [4 * 94 + 4, 4 * 34])
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.point, (94, 34), atol=1e-5)

    def test_outside_region_and_no_extrapolation(self):
        calibration = PitchCalibration(make_config())
        self.assertEqual(calibration.map_point(0, (200, 100)).reason, "outside_local_region")
        self.assertEqual(calibration.map_point(-1, (380, 136)).reason, "uncalibrated_frame")
        self.assertEqual(calibration.map_point(5, (380, 136)).reason, "uncalibrated_frame")
        self.assertEqual(calibration.map_point(0, (-1, 20)).reason, "anchor_outside_image")
        self.assertIsNone(calibration.map_point(0, (float("nan"), 20)).point)

    def test_invalid_calibration_metadata_and_order(self):
        for key, value in (("fps", 0), ("pitch_size_m", [110, 70]), ("source_start_frame", -1)):
            config = make_config()
            config[key] = value
            with self.assertRaises(ValueError):
                PitchCalibration(config)
        config = make_config()
        config["keyframes"].reverse()
        with self.assertRaises(ValueError):
            PitchCalibration(config)

    def test_bad_fit_is_not_silently_accepted(self):
        config = make_config()
        config["max_fit_rmse_px"] = 1
        config["keyframes"][0]["points"]["penalty_spot"] = [300, 180]
        with self.assertRaisesRegex(ValueError, "RMSE"):
            PitchCalibration(config)

    def test_real_configuration_is_complete_and_finite(self):
        calibration = PitchCalibration.load(ROOT / "config/v06_pitch_calibration.json")
        self.assertEqual((calibration.frames[0], calibration.frames[-1]), (0, 356))
        self.assertLess(max(calibration.reprojection_errors), 12)
        for number in range(calibration.frame_count):
            self.assertTrue(np.isfinite(calibration.matrix_at(number)).all())


class MappingPipelineTests(unittest.TestCase):
    def test_ball_prediction_and_detection_metadata_preserved(self):
        calibration = PitchCalibration(make_config())
        row = detection(role="ball", state="predicted")
        mapped = map_detection(row, calibration, 0)
        self.assertEqual(mapped["track_id"], "")
        self.assertEqual(mapped["confidence"], "")
        self.assertEqual(mapped["ball_state"], "predicted")
        self.assertEqual(mapped["position_basis"], "predicted_ground_plane_assumption")
        self.assertEqual(mapped["source_frame"], 100)
        self.assertEqual(mapped["frame"], 0)
        outside = detection()
        outside.update(x1="20", x2="30")
        mapped = map_detection(outside, calibration, 0)
        self.assertEqual(mapped["homography_valid"], "false")
        self.assertEqual((mapped["pitch_x"], mapped["pitch_y"]), ("", ""))

    def test_csv_offsets_and_timestamp_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            def write(rows):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=DETECTION_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
            calibration = PitchCalibration(make_config())
            write([detection(99), detection(100), detection(104), detection(105)])
            self.assertEqual(set(read_detections(path, calibration, 100)), {0, 4})
            with self.assertRaisesRegex(ValueError, "No detections"):
                read_detections(path, calibration, 1000)
            wrong = detection()
            wrong["time"] = "0"
            write([wrong])
            with self.assertRaisesRegex(ValueError, "timestamp"):
                read_detections(path, calibration, 100)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_video_pipeline_preserves_all_frames_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, csv_path = folder / "input.mp4", folder / "input.csv"
            config_path, output = folder / "config.json", folder / "mapped.mp4"
            config = make_config()
            config_path.write_text(json.dumps(config))
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (600, 600))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.full((600, 600, 3), (25, 100, 30), np.uint8))
            writer.release()
            with csv_path.open("w", newline="") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=DETECTION_FIELDS)
                csv_writer.writeheader()
                csv_writer.writerows([detection(100), detection(102, "ball", "predicted")])
            report = run(source, csv_path, config_path, output)
            self.assertEqual(report["frame_count"], 5)
            self.assertEqual(report["rows"], 2)
            result = cv2.VideoCapture(str(output))
            frames = 0
            try:
                while True:
                    ok, frame = result.read()
                    if not ok:
                        break
                    frames += 1
                    self.assertEqual(frame.shape, (600, 1120, 3))
                    self.assertGreater(np.std(frame[:, 600:]), 10)
            finally:
                result.release()
            self.assertEqual(frames, 5)
            with output.with_suffix(".csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["frame"] for row in rows], ["0", "2"])
            self.assertEqual(rows[1]["confidence"], "")
            with self.assertRaises(FileExistsError):
                run(source, csv_path, config_path, output)
            with self.assertRaisesRegex(ValueError, "input file"):
                run(source, csv_path, config_path, source, overwrite=True)
            capture = cv2.VideoCapture(str(source))
            try:
                wrong = copy.deepcopy(config)
                wrong["frame_count"] = 6
                with self.assertRaisesRegex(ValueError, "do not match"):
                    validate_video(capture, PitchCalibration(wrong))
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
