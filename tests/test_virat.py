import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from watchdog_cctv import virat
from watchdog_cctv.pipeline import read_tracks

GEOM = """\
- { meta: "clip geometry"}
- { geom: { id1: 1, id0: 0, ts0: 10, ts1: 0.33, g0: 100 200 140 300 , src: truth, } }
- { geom: { id1: 2, id0: 1, ts0: 10, ts1: 0.33, g0: 150 280 170 310 , src: truth, } }
- { geom: { id1: 3, id0: 2, ts0: 10, ts1: 0.33, g0: 0 0 300 200 , src: truth, } }
- { geom: { id1: 1, id0: 3, ts0: 13, ts1: 0.43, g0: -5 200 30 300 , src: truth, } }
"""
TYPES = """\
- { meta: "clip types"}
- { types: { id1: 1 , cset3: { Person: 1.0 } } }
- { types: { id1: 2 , cset3: { Prop: 1.0 } } }
- { types: { id1: 3 , cset3: { Vehicle: 1.0 } } }
"""
ACTIVITIES = """\
- { meta: "clip events"}
- { act: { act2: { PickUp: 1.0 } , id2: 7, timespan: [{tsr0: [10, 13]}], src: truth, actors: [{ id1: 1, timespan: [{tsr0: [10, 13]}]} , { id1: 2, timespan: [{tsr0: [10, 13]}]} ,  ]} }
"""


class ViratTests(unittest.TestCase):
    def write_diva(self, root):
        folder = root / "train"
        folder.mkdir()
        (folder / "clip.geom.yml").write_text(GEOM)
        (folder / "clip.types.yml").write_text(TYPES)
        (folder / "clip.activities.yml").write_text(ACTIVITIES)

    def test_diva_keeps_people_and_items_and_fills_gaps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_diva(root)
            self.assertEqual(virat.diva_names(root), ["clip"])
            detections = virat.read_diva(root, "clip")
            frames = list(virat.frames_from_detections(detections, 400, 400, 10.0))

            self.assertEqual([frame.timestamp for frame in frames], [1.0, 1.1, 1.2, 1.3])
            first = {item.track_id: item for item in frames[0].observations}
            self.assertEqual(set(first), {1, 2})
            self.assertEqual(first[2].kind, "item")
            self.assertEqual(first[1].bbox, (0.25, 0.5, 0.35, 0.75))
            self.assertEqual(frames[1].observations, ())
            self.assertEqual(frames[3].observations[0].bbox[0], 0.0)

            path = root / "clip.tracks.jsonl"
            self.assertEqual(virat.write_tracks(frames, path), 4)
            replayed = [
                (frame.timestamp, [(o.track_id, o.kind, tuple(o.bbox)) for o in frame.observations])
                for frame in read_tracks(path)
            ]
            expected = [
                (frame.timestamp, [(o.track_id, o.kind, o.bbox) for o in frame.observations])
                for frame in frames
            ]
            self.assertEqual(replayed, expected)

    def test_diva_activities_use_source_seconds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_diva(root)
            [activity] = virat.read_diva_activities(root, "clip", 10.0)
            self.assertEqual(activity["label"], "PickUp")
            self.assertEqual(activity["actor_track_ids"], [1, 2])
            self.assertEqual(
                (activity["start_timestamp"], activity["end_timestamp"]), (1.0, 1.3)
            )

    def test_release2_maps_object_types_and_strides(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.viratdata.objects.txt").write_text(
                "1 3 0 10 20 30 40 1\n5 3 0 50 60 10 10 4\n9 3 0 0 0 99 99 2\n"
                "1 3 1 12 20 30 40 1\n1 3 2 14 20 30 40 1\n"
            )
            self.assertEqual(virat.release2_names(root), ["clip"])
            detections = virat.read_release2(root, "clip")
            frames = list(virat.frames_from_detections(detections, 100, 100, 2.0, stride=2))

            self.assertEqual([frame.timestamp for frame in frames], [0.0, 1.0])
            kinds = {item.track_id: item.kind for item in frames[0].observations}
            self.assertEqual(kinds, {1: "person", 5: "item"})
            self.assertEqual(frames[1].observations[0].bbox, (0.14, 0.2, 0.44, 0.6))


if __name__ == "__main__":
    unittest.main()
