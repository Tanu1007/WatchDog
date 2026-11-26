import importlib.util
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from watchdog_cctv.cli import main
from watchdog_cctv.config import Rules
from watchdog_cctv.pipeline import read_tracks, run_frames
from watchdog_cctv.scenarios import render_scene, scenario_frames


class PipelineTests(unittest.TestCase):
    def test_cli_demo_and_replay_preserve_event_timing(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            self.assertEqual(
                main(
                    ["demo", "--scenario", "suspicious_pickup", "--output", directory]
                ),
                0,
            )
            demo = next(root.glob("demo-*")) / "suspicious_pickup"
            self.assertEqual(
                main(
                    [
                        "replay",
                        "--tracks",
                        str(demo / "tracks.jsonl"),
                        "--output",
                        directory,
                    ]
                ),
                0,
            )
            replay = next(root.glob("replay-*"))
            original_events = json.loads((demo / "events.json").read_text())
            replayed_events = json.loads((replay / "events.json").read_text())
            self.assertEqual(len(original_events), 1)
            self.assertEqual(len(replayed_events), 1)
            self.assertEqual(
                original_events[0]["decision_timestamp"],
                replayed_events[0]["decision_timestamp"],
            )

    def test_malformed_tracks_report_line_number(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.jsonl"
            path.write_text('{"timestamp":0,"observations":[]}\n{"timestamp":1}\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                list(read_tracks(path))

    def test_empty_source_is_failed_run(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "empty"
            with self.assertRaisesRegex(ValueError, "no readable frames"):
                run_frames([], output, Rules(), "empty")
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["status"], "failed"
            )

    def test_invalid_config_and_source_fail_cleanly(self):
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            config = Path(directory) / "bad.json"
            config.write_text('{"typo": 1}')
            self.assertEqual(main(["demo", "--config", str(config)]), 1)
            self.assertEqual(
                main(["replay", "--tracks", str(Path(directory) / "absent.jsonl")]), 1
            )

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "video extra is required")
    def test_demo_saves_playable_evidence_and_overlay(self):
        import cv2

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "demo",
                        "--video",
                        "--scenario",
                        "suspicious_pickup",
                        "--output",
                        directory,
                    ]
                ),
                0,
            )
            output = next(Path(directory).glob("demo-*")) / "suspicious_pickup"
            event = json.loads((output / "events.json").read_text())[0]
            self.assertEqual(event["evidence_status"], "ready")
            self.assertLessEqual(
                event["clip_start_timestamp"], event["decision_timestamp"] - 1.9
            )
            self.assertGreaterEqual(
                event["clip_end_timestamp"], event["decision_timestamp"] + 0.9
            )
            for path in (
                output / "annotated.mp4",
                output / event["evidence_clip_path"],
            ):
                capture = cv2.VideoCapture(str(path))
                frames = 0
                try:
                    while capture.read()[0]:
                        frames += 1
                finally:
                    capture.release()
                self.assertGreater(frames, 20)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "video extra is required")
    def test_clip_at_end_of_source_is_finalized_as_truncated(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "truncated"
            frames = (
                (frame, render_scene(frame, "suspicious_pickup"))
                for frame in scenario_frames("suspicious_pickup")
                if frame.timestamp < 6.5
            )
            run_frames(frames, output, Rules(), "short-video")
            event = json.loads((output / "events.json").read_text())[0]
            self.assertEqual(event["evidence_status"], "truncated")
            self.assertTrue((output / event["evidence_clip_path"]).is_file())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "video extra is required")
    def test_evidence_finalize_failure_is_recorded(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "failed"
            frames = (
                (frame, render_scene(frame, "owner_retrieval"))
                for frame in scenario_frames("owner_retrieval")
            )
            with patch(
                "watchdog_cctv.video.EvidenceRecorder.close",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    run_frames(frames, output, Rules(), "failed-video")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error"], "disk failure")

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "video extra is required")
    def test_one_clip_failure_preserves_other_completed_clip(self):
        import numpy as np

        from watchdog_cctv.types import Alert
        from watchdog_cctv.video import EvidenceRecorder

        with TemporaryDirectory() as directory:
            recorder = EvidenceRecorder(directory, 15)
            events = [
                Alert(name, "test", 4, 17, 29, 0, 0, "", 1)
                for name in ("first", "second")
            ]
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            recorder.observe(0, image, events)
            finish = recorder._finish

            def fail_second_clip(clip, status):
                if clip.event.event_id == "second":
                    clip.writer.release()
                    clip.event.evidence_status = "failed"
                    raise OSError("disk failure")
                finish(clip, status)

            with patch.object(recorder, "_finish", side_effect=fail_second_clip):
                with self.assertRaises(OSError):
                    recorder.observe(1, image, [])
            recorder.close(failed=True)
            self.assertEqual(events[0].evidence_status, "ready")
            self.assertTrue((Path(directory) / events[0].evidence_clip_path).exists())
            self.assertEqual(events[1].evidence_status, "failed")


if __name__ == "__main__":
    unittest.main()
