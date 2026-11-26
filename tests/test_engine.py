import unittest

from watchdog_cctv.config import Rules
from watchdog_cctv.engine import OwnershipEngine
from watchdog_cctv.scenarios import SCENARIOS, scenario_frames
from watchdog_cctv.types import Frame, Observation


class OwnershipTests(unittest.TestCase):
    def test_expected_scenarios(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                engine = OwnershipEngine()
                alerts = []
                for frame in scenario_frames(name):
                    alerts.extend(engine.update(frame))
                self.assertEqual(engine.items[4].owner_id, 17)
                self.assertEqual(len(alerts), int(name == "suspicious_pickup"))
                if alerts:
                    event = alerts[0]
                    self.assertEqual(event.actor_track_id, 29)
                    self.assertLess(event.decision_timestamp, 7.0)
                    self.assertAlmostEqual(
                        event.decision_timestamp - event.cue_timestamp, 1.2
                    )

    def test_brief_occlusions_preserve_binding(self):
        engine = OwnershipEngine()
        for frame in scenario_frames("short_occlusion"):
            engine.update(frame)
            if 1.2 <= frame.timestamp < 3:
                self.assertEqual(engine.items[4].owner_id, 17)

    def test_stationary_item_starts_with_unknown_owner(self):
        engine = OwnershipEngine()
        observations = (
            Observation(1, "person", (0.3, 0.3, 0.4, 0.7)),
            Observation(2, "item", (0.35, 0.6, 0.4, 0.65)),
        )
        for index in range(100):
            self.assertEqual(engine.update(Frame(index / 15, observations)), [])
        self.assertIsNone(engine.items[2].owner_id)

    def test_ambiguous_comoving_people_do_not_bind(self):
        engine = OwnershipEngine()
        for frame in scenario_frames("owner_retrieval"):
            if frame.timestamp >= 2:
                break
            owner = next(item for item in frame.observations if item.track_id == 17)
            duplicate = Observation(18, "person", owner.bbox)
            engine.update(Frame(frame.timestamp, frame.observations + (duplicate,)))
        self.assertIsNone(engine.items[4].owner_id)

    def test_gap_cannot_trigger_immediate_alert(self):
        engine = OwnershipEngine()
        for frame in scenario_frames("suspicious_pickup"):
            if frame.timestamp > 5:
                break
            engine.update(frame)
        observed = frame.observations
        self.assertEqual(engine.update(Frame(8.0, observed)), [])
        self.assertEqual(engine.items[4].state, "guarded")

    def test_missing_item_frames_break_interaction_timer(self):
        engine = OwnershipEngine()
        alerts = []
        for frame in scenario_frames("suspicious_pickup"):
            if 5.5 <= frame.timestamp < 5.9:
                frame = Frame(
                    frame.timestamp,
                    tuple(item for item in frame.observations if item.kind == "person"),
                )
            alerts.extend(engine.update(frame))
        self.assertEqual(len(alerts), 1)
        self.assertGreaterEqual(alerts[0].cue_timestamp, 5.9)

    def test_new_contact_after_clear_can_raise_new_event(self):
        engine = OwnershipEngine()
        alerts = []
        for frame in scenario_frames("suspicious_pickup"):
            alerts.extend(engine.update(frame))
        last = frame
        distant_actor = Observation(29, "person", (0.85, 0.32, 0.93, 0.7))
        for index in range(1, 25):
            observations = tuple(
                distant_actor if item.track_id == 29 else item
                for item in last.observations
            )
            engine.update(Frame(last.timestamp + index / 15, observations))
        for index in range(25, 65):
            alerts.extend(
                engine.update(Frame(last.timestamp + index / 15, last.observations))
            )
        self.assertEqual(len(alerts), 2)
        self.assertNotEqual(alerts[0].event_id, alerts[1].event_id)

    def test_expired_item_requires_new_binding(self):
        engine = OwnershipEngine()
        for frame in scenario_frames("owner_retrieval"):
            engine.update(frame)
        engine.update(Frame(30, ()))
        self.assertEqual(engine.items, {})
        engine.update(Frame(31, (Observation(4, "item", (0.3, 0.3, 0.4, 0.4)),)))
        self.assertIsNone(engine.items[4].owner_id)

    def test_timestamp_order_and_duplicate_ids(self):
        engine = OwnershipEngine()
        engine.update(Frame(1.0, ()))
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            engine.update(Frame(1.0, ()))
        item = Observation(1, "item", (0.1, 0.1, 0.2, 0.2))
        with self.assertRaisesRegex(ValueError, "unique"):
            Frame(2.0, (item, item))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Observation(1, "item", (0.4, 0.1, 0.2, 0.3))
        with self.assertRaises(ValueError):
            Frame(float("nan"), ())
        with self.assertRaises(ValueError):
            Rules(interaction_persistence_s=-1)
        with self.assertRaises(ValueError):
            Rules(velocity_cosine_min=1.1)


if __name__ == "__main__":
    unittest.main()
