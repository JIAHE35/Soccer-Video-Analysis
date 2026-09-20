import unittest
from unittest.mock import patch

from src.detect import (
    CLASS_COLORS,
    choose_output_fps,
    classify_model_label,
    parse_args,
)


class DetectionLabelTests(unittest.TestCase):
    def test_person_maps_to_player_color(self):
        self.assertEqual(classify_model_label("person"), "player")
        self.assertEqual(CLASS_COLORS["player"], (0, 255, 0))

    def test_sports_ball_maps_to_ball_color(self):
        self.assertEqual(classify_model_label("sports ball"), "ball")
        self.assertEqual(CLASS_COLORS["ball"], (255, 0, 0))

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


if __name__ == "__main__":
    unittest.main()
