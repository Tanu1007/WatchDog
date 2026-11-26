"""Ultralytics adapter. Create one instance for each independent video."""

from .types import Observation

# Offset for a track ID re-emitted under a second kind (see _stable_id).
KIND_SWITCH_OFFSET = 1_000_000
# Classes treated as items when no item_classes are given: the COCO bag
# classes, plus "item" from detectors fine-tuned with scripts/train_detector.py.
DEFAULT_ITEM_CLASSES = ("backpack", "handbag", "suitcase", "item")


class UltralyticsTracker:
    def __init__(
        self,
        model="yolo11n.pt",
        tracker="botsort.yaml",
        device=None,
        confidence=0.25,
        imgsz=None,
        item_classes=None,
    ):
        """item_classes: text prompts for an open-vocabulary model such as
        yolov8l-worldv2.pt; "person" is always added as the first class."""
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Install video inference with: python -m pip install -e '.[vision]'"
            ) from error
        self.model = YOLO(model)
        if item_classes:
            self.model.set_classes(["person", *item_classes])
        item_names = set(item_classes or DEFAULT_ITEM_CLASSES)
        self.tracker = tracker
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.first_kinds = {}
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.kinds = {
            identifier: "person" if name == "person" else "item"
            for identifier, name in names.items()
            if name == "person" or name in item_names
        }
        if "person" not in self.kinds.values() or "item" not in self.kinds.values():
            raise ValueError(
                "The model must include person and at least one item class: "
                + ", ".join(sorted(item_names))
            )

    def _stable_id(self, track_id, kind):
        """Keep each emitted track ID to one kind.

        BoT-SORT and ByteTrack can relabel a track's class mid-sequence, for
        example person to handbag. The engine treats a kind change as invalid,
        so the second kind continues under its own ID, like a new track.
        """
        first = self.first_kinds.setdefault(track_id, kind)
        return track_id if kind == first else track_id + KIND_SWITCH_OFFSET

    def track(self, image):
        options = {"imgsz": self.imgsz} if self.imgsz else {}
        result = self.model.track(
            image,
            persist=True,
            tracker=self.tracker,
            classes=list(self.kinds),
            device=self.device,
            conf=self.confidence,
            verbose=False,
            **options,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return ()
        observations = []
        for track_id, class_id, bbox, confidence in zip(
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.xyxyn.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            coordinates = tuple(max(0.0, min(1.0, value)) for value in bbox)
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                continue
            kind = self.kinds[int(class_id)]
            observations.append(
                Observation(
                    self._stable_id(int(track_id), kind),
                    kind,
                    coordinates,
                    float(confidence),
                )
            )
        return tuple(observations)
