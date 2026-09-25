import unittest

import numpy as np

from src.team_classifier import (
    KitColorClassifier,
    RolePrediction,
    TrackRoleVoter,
    extract_jersey_crop,
)


class JerseyCropTests(unittest.TestCase):
    def test_crop_is_clipped_to_frame_bounds(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        crop = extract_jersey_crop(frame, (-20, 10, 140, 90))

        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)
        self.assertLessEqual(crop.shape[0], frame.shape[0])
        self.assertLessEqual(crop.shape[1], frame.shape[1])

    def test_tiny_box_returns_empty_crop(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)

        crop = extract_jersey_crop(frame, (10, 10, 11, 11))

        self.assertEqual(crop.size, 0)


class KitColorClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = KitColorClassifier(
            team_a_color="red",
            team_b_color="white",
            referee_color="black",
            team_b_goalkeeper_color="blue",
        )
        self.coordinates = (20, 10, 80, 90)

    def solid_person_frame(self, color):
        frame = np.full((100, 100, 3), (30, 120, 30), dtype=np.uint8)
        frame[10:91, 20:81] = color
        return frame

    def test_red_jersey_maps_to_team_a(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((0, 0, 180)), self.coordinates
        )

        self.assertEqual(prediction.role, "team_a")
        self.assertGreater(prediction.confidence, 0.8)

    def test_white_jersey_maps_to_team_b(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((220, 220, 220)), self.coordinates
        )

        self.assertEqual(prediction.role, "team_b")
        self.assertGreater(prediction.confidence, 0.8)

    def test_black_jersey_maps_to_referee(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((25, 25, 25)), self.coordinates
        )

        self.assertEqual(prediction.role, "referee")
        self.assertGreater(prediction.confidence, 0.8)

    def test_dark_saturated_red_is_not_referee(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((0, 0, 55)), self.coordinates
        )

        self.assertEqual(prediction.role, "team_a")

    def test_blue_goalkeeper_maps_to_team_b_goalkeeper(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((180, 60, 20)), self.coordinates
        )

        self.assertEqual(prediction.role, "team_b_goalkeeper")
        self.assertGreater(prediction.confidence, 0.8)

    def test_optional_yellow_goalkeeper_maps_to_team_a_goalkeeper(self):
        classifier = KitColorClassifier(
            team_a_color="red",
            team_b_color="white",
            referee_color="black",
            team_a_goalkeeper_color="yellow",
            team_b_goalkeeper_color="blue",
        )

        prediction = classifier.predict(
            self.solid_person_frame((0, 220, 220)), self.coordinates
        )

        self.assertEqual(prediction.role, "team_a_goalkeeper")

    def test_unconfigured_goalkeeper_color_is_unknown(self):
        prediction = self.classifier.predict(
            self.solid_person_frame((0, 220, 220)), self.coordinates
        )

        self.assertEqual(prediction, RolePrediction("unknown", 0.0))

    def test_green_or_tiny_crop_is_unknown(self):
        green_frame = self.solid_person_frame((30, 120, 30))

        self.assertEqual(
            self.classifier.predict(green_frame, self.coordinates).role,
            "unknown",
        )
        self.assertEqual(
            self.classifier.predict(green_frame, (10, 10, 11, 11)).role,
            "unknown",
        )

    def test_ambiguous_red_and_white_crop_is_unknown(self):
        frame = self.solid_person_frame((0, 0, 180))
        frame[10:91, 50:81] = (220, 220, 220)

        prediction = self.classifier.predict(frame, self.coordinates)

        self.assertEqual(prediction.role, "unknown")


class TrackRoleVoterTests(unittest.TestCase):
    def test_three_votes_establish_role_and_unknown_does_not_erase_it(self):
        voter = TrackRoleVoter(min_votes=3, window_size=5, stale_after_frames=90)
        team_a = RolePrediction("team_a", 0.8)

        self.assertEqual(voter.update(7, team_a, 0), "unknown")
        self.assertEqual(voter.update(7, team_a, 1), "unknown")
        self.assertEqual(voter.update(7, team_a, 2), "team_a")
        self.assertEqual(
            voter.update(7, RolePrediction("unknown", 0.0), 3), "team_a"
        )

    def test_one_conflicting_vote_does_not_switch_established_role(self):
        voter = TrackRoleVoter(min_votes=3, window_size=5, stale_after_frames=90)
        for frame_index in range(3):
            voter.update(4, RolePrediction("team_a", 0.8), frame_index)

        role = voter.update(4, RolePrediction("team_b", 0.9), 3)

        self.assertEqual(role, "team_a")

    def test_later_majority_can_switch_role(self):
        voter = TrackRoleVoter(min_votes=3, window_size=5, stale_after_frames=90)
        for frame_index in range(3):
            voter.update(4, RolePrediction("team_a", 0.8), frame_index)
        voter.update(4, RolePrediction("team_b", 0.9), 3)
        voter.update(4, RolePrediction("team_b", 0.9), 4)

        role = voter.update(4, RolePrediction("team_b", 0.9), 5)

        self.assertEqual(role, "team_b")

    def test_untracked_person_uses_current_prediction(self):
        voter = TrackRoleVoter()

        role = voter.update(None, RolePrediction("team_b", 0.7), 0)

        self.assertEqual(role, "team_b")

    def test_stale_track_restarts_voting_history(self):
        voter = TrackRoleVoter(min_votes=3, window_size=5, stale_after_frames=2)
        for frame_index in range(3):
            voter.update(9, RolePrediction("team_a", 0.8), frame_index)

        role = voter.update(9, RolePrediction("team_b", 0.9), 5)

        self.assertEqual(role, "unknown")

    def test_goalkeeper_role_uses_same_temporal_voting(self):
        voter = TrackRoleVoter(min_votes=3, window_size=5, stale_after_frames=90)
        prediction = RolePrediction("team_b_goalkeeper", 0.9)

        self.assertEqual(voter.update(12, prediction, 0), "unknown")
        self.assertEqual(voter.update(12, prediction, 1), "unknown")
        self.assertEqual(voter.update(12, prediction, 2), "team_b_goalkeeper")


if __name__ == "__main__":
    unittest.main()
