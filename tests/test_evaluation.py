import importlib.util
import math
import unittest

from watchdog_cctv import evaluation
from watchdog_cctv.tracking import KIND_SWITCH_OFFSET, UltralyticsTracker
from watchdog_cctv.types import Frame, Observation


class TrackerIdentityTests(unittest.TestCase):
    def test_kind_change_continues_under_a_separate_id(self):
        tracker = UltralyticsTracker.__new__(UltralyticsTracker)
        tracker.first_kinds = {}
        self.assertEqual(tracker._stable_id(7, "person"), 7)
        self.assertEqual(tracker._stable_id(7, "item"), 7 + KIND_SWITCH_OFFSET)
        self.assertEqual(tracker._stable_id(7, "person"), 7)
        self.assertEqual(tracker._stable_id(8, "item"), 8)


class EvaluationTests(unittest.TestCase):
    def test_iou_distances_reject_weak_overlap(self):
        distances = evaluation.iou_distances(
            [(0.0, 0.0, 0.2, 0.2)], [(0.0, 0.0, 0.2, 0.2), (0.1, 0.1, 0.3, 0.3)]
        )
        self.assertAlmostEqual(distances[0, 0], 0.0)
        self.assertTrue(math.isnan(distances[0, 1]))
        self.assertEqual(evaluation.iou_distances([], []).shape, (0, 0))

    def test_owner_pairs_and_binding_scores(self):
        kinds = {1: "person", 2: "person", 10: "item", 11: "item"}
        activities = [
            {"label": "PickUp", "actor_track_ids": [1, 10]},
            {"label": "activity_walking", "actor_track_ids": [2, 11]},
            {"label": "Object_Transfer", "actor_track_ids": [1, 2, 11]},
        ]
        owners = evaluation.gt_owner_pairs(activities, kinds)
        self.assertEqual(owners, {10: {1}, 11: {1, 2}})

        # Predicted IDs 100/200/300 map to ground truth 1/10/11.
        mapping = {100: 1, 200: 10, 300: 11, 400: 2}
        bindings = {(200, 400): 3.0, (200, 100): 1.0, (999, 100): 2.0}
        score = evaluation.score_bindings(bindings, owners, {10, 11}, mapping)
        self.assertEqual(
            score,
            {"gt_owned_items": 2, "covered_items": 1, "correct_items": 1, "unjudged_bindings": 1},
        )

    @unittest.skipUnless(importlib.util.find_spec("motmetrics"), "motmetrics is required")
    def test_tracking_metrics_map_predicted_ids(self):
        truth = [Frame(i / 10, (Observation(1, "person", (0.1, 0.1, 0.3, 0.5)),)) for i in range(5)]
        predicted = [Frame(i / 10, (Observation(42, "person", (0.1, 0.1, 0.3, 0.52)),)) for i in range(5)]
        accumulators, mapping = evaluation.tracking_metrics(truth, predicted, 10.0)
        self.assertEqual(mapping, {42: 1})
        person = evaluation.summarize_tracking([accumulators["person"]])
        self.assertEqual(person["idf1"], 1)
        self.assertEqual(person["num_switches"], 0)

    def test_score_alerts_lead_time_and_false_alerts(self):
        events, false_alerts = evaluation.score_alerts(
            [5.0, 9.5, 12.0, 30.0], [(10.0, 20.0), (40.0, 50.0)], tolerance=1.0
        )
        self.assertEqual(
            [(e["detected"], e["first_alert"], e["lead_seconds"], e["delay_seconds"]) for e in events],
            [(True, 9.5, 10.5, -0.5), (False, None, None, None)],
        )
        self.assertEqual(false_alerts, [5.0, 30.0])


if __name__ == "__main__":
    unittest.main()
