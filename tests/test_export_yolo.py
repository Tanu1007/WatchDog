import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_yolo_dataset.py"
spec = importlib.util.spec_from_file_location("export_yolo_dataset", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class LabelLinesTest(unittest.TestCase):
    def test_converts_clips_and_drops_tiny_boxes(self):
        detections = [
            (1, "person", (100, 50, 200, 250)),
            (2, "item", (-20, 90, 20, 110)),  # clipped to x 0..20
            (3, "item", (300, 300, 302, 340)),  # 2 px wide: dropped
        ]
        lines = export.label_lines(detections, 400, 400, 4)
        self.assertEqual(
            lines,
            [
                "0 0.375000 0.375000 0.250000 0.500000",
                "1 0.025000 0.250000 0.050000 0.050000",
            ],
        )


if __name__ == "__main__":
    unittest.main()
