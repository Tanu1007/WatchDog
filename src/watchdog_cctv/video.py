"""Annotated playback and evidence clips with a bounded pre-event frame buffer."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .scenarios import pixel_box
from .types import Alert


def require_video():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install video support with: python -m pip install -e '.[video]'"
        ) from error
    return cv2


def open_writer(path, fps, shape):
    cv2 = require_video()
    height, width = shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create MP4 video at {path}")
    return writer


def annotate(image, frame, engine):
    cv2 = require_video()
    image = image.copy()
    observations = {item.track_id: item for item in frame.observations}
    for observation in frame.observations:
        x1, y1, x2, y2 = pixel_box(observation, image.shape)
        color = (200, 170, 60)
        label = f"P{observation.track_id}"
        if observation.kind == "item":
            memory = engine.items[observation.track_id]
            color = (70, 80, 245) if memory.state == "alert" else (80, 210, 210)
            owner_label = (
                "unknown" if memory.owner_id is None else f"P{memory.owner_id}"
            )
            label = (
                f"I{observation.track_id} owner {owner_label} {memory.state.upper()}"
            )
            owner = observations.get(memory.owner_id)
            if owner is not None:
                ox1, _, ox2, oy2 = pixel_box(owner, image.shape)
                cv2.line(image, ((x1 + x2) // 2, y2), ((ox1 + ox2) // 2, oy2), color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (max(x1, 4), max(y1 - 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    cv2.putText(
        image,
        f"Source time {frame.timestamp:.2f}s",
        (24, image.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
    )
    return image


@dataclass
class PendingClip:
    event: Alert
    writer: object
    temporary_path: Path
    final_path: Path
    end_timestamp: float


class EvidenceRecorder:
    def __init__(self, output, fps, pre_seconds=2.0, post_seconds=1.0):
        self.output = Path(output)
        self.directory = self.output / "clips"
        self.directory.mkdir(exist_ok=True)
        self.fps = fps
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.buffer = deque()
        self.pending: list[PendingClip] = []
        self.last_timestamp = 0.0

    def observe(self, timestamp, image, events):
        self.last_timestamp = timestamp
        self.buffer.append((timestamp, image.copy()))
        while self.buffer and self.buffer[0][0] < timestamp - self.pre_seconds:
            self.buffer.popleft()

        for clip in tuple(self.pending):
            if timestamp <= clip.end_timestamp + 1e-9:
                clip.writer.write(image)
                clip.event.clip_end_timestamp = timestamp
            if timestamp >= clip.end_timestamp - 1e-9:
                self.pending.remove(clip)
                self._finish(clip, "ready")

        for event in events:
            final_path = self.directory / f"{event.event_id}.mp4"
            temporary_path = final_path.with_suffix(".partial.mp4")
            writer = open_writer(temporary_path, self.fps, image.shape)
            clip = PendingClip(
                event, writer, temporary_path, final_path, timestamp + self.post_seconds
            )
            self.pending.append(clip)
            for _, buffered in self.buffer:
                writer.write(buffered)
            event.evidence_clip_path = str(final_path.relative_to(self.output))
            event.evidence_status = "recording"
            event.clip_start_timestamp = self.buffer[0][0]
            event.clip_end_timestamp = timestamp

    def _finish(self, clip, status):
        clip.writer.release()
        cv2 = require_video()
        capture = cv2.VideoCapture(str(clip.temporary_path))
        try:
            readable = capture.isOpened() and capture.read()[0]
        finally:
            capture.release()
        if not readable:
            clip.event.evidence_status = "failed"
            raise RuntimeError(
                f"Evidence video could not be saved for {clip.event.event_id}"
            )
        try:
            clip.temporary_path.replace(clip.final_path)
        except OSError:
            clip.event.evidence_status = "failed"
            raise
        clip.event.evidence_status = status

    def close(self, failed=False):
        pending, self.pending = self.pending, []
        first_error = None
        for clip in pending:
            complete = self.last_timestamp + 1e-9 >= clip.end_timestamp
            status = "ready" if complete and not failed else "truncated"
            try:
                self._finish(clip, status)
            except Exception as error:
                first_error = first_error or error
        self.buffer.clear()
        if first_error is not None:
            raise first_error
