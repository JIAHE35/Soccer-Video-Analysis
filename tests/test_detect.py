import unittest
from unittest.mock import patch

import cv2
import numpy as np

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
        self.assertEqual(args.output, "outputs/detected_cfr.mp4")
        self.assertEqual(args.player_conf, 0.45)
        self.assertEqual(args.ball_conf, 0.15)
        self.assertEqual(args.min_field_green_ratio, 0.25)

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


if __name__ == "__main__":
    unittest.main()
