import unittest

import numpy as np

from watchdog_cctv import pairwise
from watchdog_cctv.types import Frame, Observation

FPS = 15.0
COLUMN = {name: index for index, name in enumerate(pairwise.FEATURE_NAMES)}


def carry_frames(count=90, item_missing=(), stranger=False):
    """A person walking right at 0.075 width/s with a bag at their side."""
    frames = []
    for index in range(count):
        x = 0.1 + 0.005 * index
        observations = [Observation(1, "person", (x, 0.4, x + 0.05, 0.6))]
        if index not in item_missing:
            observations.append(Observation(2, "item", (x + 0.05, 0.5, x + 0.07, 0.55)))
        if stranger:
            observations.append(Observation(3, "person", (0.9, 0.0, 0.95, 0.2)))
        frames.append(Frame(index / FPS, tuple(observations)))
    return frames


class PairFeatureTests(unittest.TestCase):
    def test_carried_item_moves_with_person(self):
        times, tracks = pairwise.track_arrays(carry_frames())
        features, masks = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        self.assertEqual(features.shape, (90, 14))
        self.assertEqual(masks.shape, (90, 3))
        later = features[1:]
        # Height 0.2: the bag's center sits 0.035 right and 0.025 below the person's,
        # i.e. 0.175 and 0.125 person heights.
        np.testing.assert_allclose(later[:, COLUMN["dx"]], 0.175, atol=1e-5)
        np.testing.assert_allclose(later[:, COLUMN["dy"]], 0.125, atol=1e-5)
        np.testing.assert_allclose(later[:, COLUMN["box_gap"]], 0, atol=1e-6)
        np.testing.assert_allclose(later[:, COLUMN["velocity_cosine"]], 1, atol=1e-5)
        np.testing.assert_allclose(later[:, COLUMN["relative_speed"]], 0, atol=1e-4)
        np.testing.assert_allclose(later[:, COLUMN["person_speed"]], 0.375, atol=1e-4)
        self.assertEqual(features[0, COLUMN["person_speed"]], 0)  # no previous sample
        # 89 frames after first contact = 5.93 s, scaled by the 10 s cap.
        self.assertAlmostEqual(float(features[-1, COLUMN["copresence_seconds"]]), 89 / FPS / 10, places=5)

    def test_wrist_distance_is_a_masked_placeholder(self):
        times, tracks = pairwise.track_arrays(carry_frames())
        features, masks = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        self.assertTrue((features[:, COLUMN["wrist_distance"]] == 0).all())
        self.assertTrue((masks[:, pairwise.MASK_NAMES.index("pose_available")] == 0).all())

    def test_missing_item_is_masked_and_zeroed(self):
        times, tracks = pairwise.track_arrays(carry_frames(item_missing={10, 11}))
        features, masks = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        self.assertEqual(masks[10].tolist(), [1, 0, 0])
        self.assertEqual(features[10, COLUMN["box_gap"]], 0)
        self.assertGreater(features[10, COLUMN["person_speed"]], 0)
        self.assertEqual(features[10, COLUMN["item_speed"]], 0)
        # After the gap the item velocity spans 3 frames, still under 0.5 s.
        self.assertAlmostEqual(float(features[12, COLUMN["velocity_cosine"]]), 1.0, places=4)


    def test_comotion_history_persists_after_the_pair_stops(self):
        # Walk with the bag for 30 frames, then stand still beside it for 60.
        frames = carry_frames(count=30)
        x = 0.1 + 0.005 * 29
        for index in range(30, 90):
            frames.append(Frame(index / FPS, (
                Observation(1, "person", (x, 0.4, x + 0.05, 0.6)),
                Observation(2, "item", (x + 0.05, 0.5, x + 0.07, 0.55)),
            )))
        times, tracks = pairwise.track_arrays(frames)
        features, _ = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        comotion = features[:, COLUMN["comotion_seconds"]]
        since = features[:, COLUMN["since_comotion"]]
        # Frames 1..29 co-move: 29/15 s of co-motion, kept while standing still.
        self.assertAlmostEqual(float(comotion[29]), 29 / FPS / 10, places=5)
        self.assertAlmostEqual(float(comotion[-1]), float(comotion[29]), places=6)
        self.assertEqual(float(since[0]), 1.0)  # no co-motion yet: capped
        self.assertEqual(float(since[29]), 0.0)
        self.assertAlmostEqual(float(since[-1]), (89 - 29) / FPS / 30, places=5)
        self.assertEqual(float(features[-1, COLUMN["velocity_cosine"]]), 0.0)

    def test_features_are_causal(self):
        frames = carry_frames(count=90, item_missing={40, 41})
        times, tracks = pairwise.track_arrays(frames)
        full, _ = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        times, tracks = pairwise.track_arrays(frames[:50])
        prefix, _ = pairwise.pair_features(times, tracks[1][1], tracks[2][1])
        np.testing.assert_array_equal(full[:50], prefix)


class WindowTests(unittest.TestCase):
    def test_windows_labels_and_candidates(self):
        frames = carry_frames(count=150, stranger=True)
        owners = {2: {1}}
        activities = [{
            "label": "Transport_HeavyCarry",
            "start_timestamp": 0.0,
            "end_timestamp": 5.0,
            "actor_track_ids": [1, 2],
        }]
        features, masks, ownership, interaction, meta = pairwise.build_windows(frames, owners, activities)
        # The stranger (track 3) never comes near the bag, so only (1, 2) is a pair.
        self.assertEqual({(m["person"], m["item"]) for m in meta}, {(1, 2)})
        # 150 frames, window 60, hop 30: starts 0, 30, 60, 90.
        self.assertEqual([m["start_timestamp"] for m in meta], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(features.shape, (4, 60, 14))
        self.assertEqual(masks.shape, (4, 60, 3))
        self.assertEqual(ownership.tolist(), [1, 1, 1, 1])
        carry = pairwise.INTERACTION_CLASSES.index("carry")
        # Window ends at 3.93 s and 5.93 s ... only the first is inside the activity.
        self.assertEqual(interaction.tolist(), [carry, 0, 0, 0])

    def test_unmatched_and_unowned_items_are_ignored(self):
        frames = carry_frames(count=60)
        _, _, ownership, interaction, _ = pairwise.build_windows(frames, {}, [])
        self.assertEqual(ownership.tolist(), [-1])
        self.assertEqual(interaction.tolist(), [0])
        _, _, ownership, interaction, _ = pairwise.build_windows(frames, {7: {8}}, [], mapping={1: 8})
        self.assertEqual(ownership.tolist(), [-1])
        self.assertEqual(interaction.tolist(), [-1])

    def test_short_clip_gives_no_windows(self):
        features, _, ownership, _, meta = pairwise.build_windows(carry_frames(count=30), {2: {1}}, [])
        self.assertEqual(features.shape, (0, 60, 14))
        self.assertEqual(len(ownership), 0)
        self.assertEqual(meta, [])


if __name__ == "__main__":
    unittest.main()
