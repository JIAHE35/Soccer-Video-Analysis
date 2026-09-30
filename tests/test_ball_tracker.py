import unittest

from src.ball_tracker import BallCandidate, BallTracker


class BallTrackerTests(unittest.TestCase):
    def test_two_nearby_detections_confirm_track(self):
        tracker = BallTracker(min_confirmed_hits=2)

        first = tracker.update([BallCandidate((10, 10, 18, 18), 0.30)])
        second = tracker.update([BallCandidate((14, 11, 22, 19), 0.35)])

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.state, "detected")
        self.assertEqual(second.bbox, (14, 11, 22, 19))
        self.assertEqual(second.confidence, 0.35)

    def test_confirmed_track_prefers_nearby_candidate_over_distant_confident_one(self):
        tracker = BallTracker(min_confirmed_hits=2, max_match_distance=60)
        tracker.update([BallCandidate((100, 100, 108, 108), 0.30)])
        tracker.update([BallCandidate((104, 100, 112, 108), 0.32)])

        result = tracker.update(
            [
                BallCandidate((109, 101, 117, 109), 0.22),
                BallCandidate((300, 220, 308, 228), 0.95),
            ]
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.bbox, (109, 101, 117, 109))

    def test_short_gap_returns_prediction_without_detection_confidence(self):
        tracker = BallTracker(
            min_confirmed_hits=2,
            max_missing_frames=2,
            velocity_smoothing=0.0,
        )
        tracker.update([BallCandidate((10, 10, 18, 18), 0.30)])
        tracker.update([BallCandidate((14, 10, 22, 18), 0.35)])

        predicted = tracker.update([])

        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.state, "predicted")
        self.assertEqual(predicted.bbox, (18, 10, 26, 18))
        self.assertIsNone(predicted.confidence)

    def test_track_expires_after_maximum_missing_frames(self):
        tracker = BallTracker(
            min_confirmed_hits=2,
            max_missing_frames=2,
            velocity_smoothing=0.0,
        )
        tracker.update([BallCandidate((10, 10, 18, 18), 0.30)])
        tracker.update([BallCandidate((14, 10, 22, 18), 0.35)])

        self.assertIsNotNone(tracker.update([]))
        self.assertIsNotNone(tracker.update([]))
        self.assertIsNone(tracker.update([]))
        self.assertIsNone(
            tracker.update([BallCandidate((30, 10, 38, 18), 0.40)])
        )

    def test_distant_candidate_does_not_replace_confirmed_track(self):
        tracker = BallTracker(
            min_confirmed_hits=2,
            max_missing_frames=2,
            max_match_distance=40,
            velocity_smoothing=0.0,
        )
        tracker.update([BallCandidate((10, 10, 18, 18), 0.30)])
        tracker.update([BallCandidate((14, 10, 22, 18), 0.35)])

        result = tracker.update([BallCandidate((400, 250, 408, 258), 0.99)])

        self.assertIsNotNone(result)
        self.assertEqual(result.state, "predicted")
        self.assertEqual(result.bbox, (18, 10, 26, 18))

    def test_oversized_candidates_are_not_confirmed(self):
        tracker = BallTracker(min_confirmed_hits=2, max_box_side=40)
        oversized = BallCandidate((20, 20, 100, 100), 0.90)

        self.assertIsNone(tracker.update([oversized]))
        self.assertIsNone(tracker.update([oversized]))

    def test_flattened_pitch_mark_candidate_is_not_confirmed(self):
        tracker = BallTracker(min_confirmed_hits=2, max_aspect_ratio=1.8)
        flattened = BallCandidate((100, 100, 111, 105), 0.53)

        self.assertIsNone(tracker.update([flattened]))
        self.assertIsNone(tracker.update([flattened]))

    def test_reset_requires_a_new_track_to_be_confirmed(self):
        tracker = BallTracker(min_confirmed_hits=2)
        tracker.update([BallCandidate((10, 10, 18, 18), 0.30)])
        self.assertIsNotNone(
            tracker.update([BallCandidate((14, 10, 22, 18), 0.35)])
        )

        tracker.reset()

        self.assertIsNone(
            tracker.update([BallCandidate((14, 10, 22, 18), 0.35)])
        )

    def test_static_low_confidence_candidate_far_from_players_stays_tentative(self):
        tracker = BallTracker(min_confirmed_hits=2)
        candidate = BallCandidate((100, 100, 108, 108), 0.26)
        distant_player = [(300.0, 300.0)]

        self.assertIsNone(tracker.update([candidate], distant_player))
        self.assertIsNone(tracker.update([candidate], distant_player))
        self.assertIsNone(tracker.update([candidate], distant_player))

    def test_low_confidence_candidate_near_player_can_be_confirmed(self):
        tracker = BallTracker(min_confirmed_hits=2)
        candidate = BallCandidate((100, 100, 108, 108), 0.26)
        nearby_player = [(112.0, 116.0)]

        self.assertIsNone(tracker.update([candidate], nearby_player))
        result = tracker.update([candidate], nearby_player)

        self.assertIsNotNone(result)
        self.assertEqual(result.state, "detected")


if __name__ == "__main__":
    unittest.main()
