import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

import src.detect as detect
from src.detect import (
    CLASS_COLORS,
    build_field_mask,
    choose_output_fps,
    classify_model_label,
    detection_anchor,
    is_detection_on_field,
    parse_args,
)


class DetectionLabelTests(unittest.TestCase):
    def test_person_maps_to_player_color(self):
        self.assertEqual(classify_model_label("person"), "player")
        self.assertEqual(CLASS_COLORS["player"], (0, 255, 0))

    def test_sports_ball_maps_to_ball_color(self):
        self.assertEqual(classify_model_label("sports ball"), "ball")
        self.assertEqual(CLASS_COLORS["ball"], (255, 0, 0))

    def test_custom_model_labels_map_to_project_labels(self):
        self.assertEqual(classify_model_label("player"), "player")
        self.assertEqual(classify_model_label("referee"), "referee")
        self.assertEqual(classify_model_label("ball"), "ball")

    def test_unknown_model_label_is_ignored(self):
        self.assertIsNone(classify_model_label("car"))

    def test_referee_color_is_reserved_for_custom_model(self):
        self.assertEqual(CLASS_COLORS["referee"], (0, 0, 255))

    def test_output_fps_preserves_source_fps(self):
        self.assertEqual(choose_output_fps(32.652), 32.652)
        self.assertEqual(choose_output_fps(0), 30.0)

    def test_cli_defaults_use_constant_fps_video(self):
        with patch("sys.argv", ["detect.py"]):
            args = parse_args()

        self.assertEqual(args.source, "data/soccervideo_cfr.mp4")
        self.assertEqual(args.output, "outputs/v03_player_tracking.mp4")
        self.assertEqual(args.player_conf, 0.5)
        self.assertEqual(args.ball_conf, 0.18)
        self.assertEqual(args.min_field_green_ratio, 0.25)

    def test_player_label_includes_track_id(self):
        self.assertTrue(hasattr(detect, "format_detection_label"))
        self.assertEqual(
            detect.format_detection_label("player", 0.834, track_id=12),
            "player #12 0.83",
        )

    def test_untracked_player_label_omits_track_id(self):
        self.assertTrue(hasattr(detect, "format_detection_label"))
        self.assertEqual(
            detect.format_detection_label("player", 0.834, track_id=None),
            "player 0.83",
        )

    def test_ball_label_ignores_track_id(self):
        self.assertTrue(hasattr(detect, "format_detection_label"))
        self.assertEqual(
            detect.format_detection_label("ball", 0.456, track_id=7),
            "ball 0.46",
        )

    def test_model_class_ids_separate_people_from_balls(self):
        self.assertTrue(hasattr(detect, "select_model_class_ids"))
        self.assertEqual(
            detect.select_model_class_ids(
                {0: "person", 1: "car", 2: "referee", 3: "sports ball"}
            ),
            ([0, 2], [3]),
        )

    def test_player_anchor_uses_bottom_center_and_ball_uses_center(self):
        coordinates = (10, 20, 30, 60)

        self.assertEqual(detection_anchor("player", coordinates), (20, 60))
        self.assertEqual(detection_anchor("ball", coordinates), (20, 40))

    def test_green_field_support_accepts_player_and_ball(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (30, 120, 30)
        field_mask = build_field_mask(frame, cv2)

        self.assertTrue(
            is_detection_on_field(
                field_mask,
                "player",
                (40, 20, 60, 80),
                cv2,
                min_green_ratio=0.25,
            )
        )
        self.assertTrue(
            is_detection_on_field(
                field_mask,
                "ball",
                (40, 45, 50, 55),
                cv2,
                min_green_ratio=0.25,
            )
        )

    def test_non_green_anchor_is_rejected(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (120, 120, 120)
        field_mask = build_field_mask(frame, cv2)

        self.assertFalse(
            is_detection_on_field(
                field_mask,
                "player",
                (40, 20, 60, 80),
                cv2,
                min_green_ratio=0.25,
            )
        )

    def test_bottom_cropped_player_uses_grass_below_the_body(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (120, 120, 120)
        frame[70:, :] = (30, 120, 30)
        frame[80:, 40:61] = (30, 30, 120)
        field_mask = build_field_mask(frame, cv2)

        self.assertTrue(
            is_detection_on_field(
                field_mask,
                "player",
                (10, 10, 90, 89),
                cv2,
                min_green_ratio=0.25,
            )
        )


class VideoPipelineTests(unittest.TestCase):
    def test_low_confidence_ball_bypasses_tracking_and_empty_frames_survive(self):
        from ultralytics.engine.results import Results

        names = {0: "person", 32: "sports ball"}
        frame = np.full((128, 200, 3), (30, 120, 30), dtype=np.uint8)
        results = [
            Results(frame.copy(), "input.mp4", names,
                    boxes=np.array([[20, 30, 60, 100, 7, 0.8, 0]], dtype=np.float32)),
            Results(frame.copy(), "input.mp4", names,
                    boxes=np.array([[20, 30, 60, 100, 0.65, 0]], dtype=np.float32)),
            Results(frame.copy(), "input.mp4", names,
                    boxes=np.empty((0, 6), dtype=np.float32)),
        ]
        test_case = self

        class PeopleModel:
            def __init__(self):
                self.names = names

            def track(self, **kwargs):
                test_case.assertEqual(kwargs["classes"], [0])
                test_case.assertEqual(kwargs["tracker"], "bytetrack.yaml")
                test_case.assertTrue(kwargs["persist"])
                test_case.assertTrue(kwargs["stream"])
                test_case.assertLessEqual(kwargs["conf"], 0.1)
                return iter(results)

        class BallModel:
            def predict(self, **kwargs):
                test_case.assertEqual(kwargs["classes"], [32])
                test_case.assertEqual(kwargs["conf"], 0.18)
                np.testing.assert_array_equal(kwargs["source"], frame)
                return [Results(
                    kwargs["source"], "input.mp4", names,
                    boxes=np.array([[130, 90, 138, 98, 0.21, 32]], dtype=np.float32),
                )]

        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            output = Path(directory) / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (200, 128)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(frame)
            writer.release()

            with patch("ultralytics.YOLO", side_effect=[PeopleModel(), BallModel()]), \
                    patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
                self.assertEqual(detect.detect_video(source, output), output)

            labels = [call.args[1] for call in put_text.call_args_list]
            self.assertEqual(labels, [
                "player #7 0.80", "ball 0.21",
                "player 0.65", "ball 0.21", "ball 0.21",
            ])
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30)
                decoded = []
                while True:
                    ok, decoded_frame = capture.read()
                    if not ok:
                        break
                    decoded.append(decoded_frame)
                self.assertEqual(len(decoded), 3)
                for decoded_frame in decoded:
                    blue, green, red = map(int, decoded_frame[90, 130])
                    self.assertGreater(blue, green + 60)
                    self.assertGreater(blue, red + 60)
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
