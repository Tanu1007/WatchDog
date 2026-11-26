"""Synthetic trajectories for exercising ownership rules without model downloads."""

from .types import Frame, Observation

SCENARIOS = (
    "owner_retrieval",
    "benign_interaction",
    "short_occlusion",
    "suspicious_pickup",
)


def scenario_frames(name: str, fps: float = 15.0):
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}")
    for index in range(int(10 * fps)):
        timestamp = index / fps
        owner_x = 0.2 + 0.06 * min(timestamp, 2)
        item_x = owner_x + 0.05
        actor_x = 0.78
        if timestamp >= 2:
            item_x = 0.37
            owner_x = max(0.06, 0.32 - 0.16 * (timestamp - 2))

        if name in {"owner_retrieval", "short_occlusion"} and timestamp >= 5:
            owner_x = min(0.32, 0.06 + 0.26 * (timestamp - 5))
            if timestamp >= 7:
                owner_x += 0.06 * (timestamp - 7)
                item_x += 0.06 * (timestamp - 7)

        if name == "benign_interaction" and 5 <= timestamp < 5.6:
            actor_x = 0.4
        if name == "suspicious_pickup":
            actor_x = max(0.4, 0.78 - 0.38 * max(0, timestamp - 4))
            if timestamp >= 7:
                actor_x += 0.06 * (timestamp - 7)
                item_x += 0.06 * (timestamp - 7)

        people = [
            Observation(17, "person", (owner_x - 0.04, 0.32, owner_x + 0.04, 0.7)),
            Observation(29, "person", (actor_x - 0.04, 0.32, actor_x + 0.04, 0.7)),
        ]
        items = [Observation(4, "item", (item_x - 0.025, 0.59, item_x + 0.025, 0.65))]
        if name == "short_occlusion":
            if 1.2 <= timestamp < 1.6:
                people = [person for person in people if person.track_id != 17]
            if 2.2 <= timestamp < 2.6:
                items = []
        yield Frame(timestamp, tuple(people + items))


def render_scene(frame: Frame, scenario: str):
    import cv2
    import numpy as np

    image = np.full((540, 960, 3), (29, 25, 21), dtype=np.uint8)
    cv2.putText(
        image,
        "WatchDog | synthetic trajectory demo",
        (24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
    )
    cv2.putText(
        image,
        scenario.replace("_", " "),
        (24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )
    cv2.line(image, (0, 380), (959, 380), (60, 60, 60), 2)
    for observation in frame.observations:
        x1, y1, x2, y2 = pixel_box(observation, image.shape)
        color = (208, 167, 76) if observation.kind == "person" else (79, 192, 237)
        if observation.kind == "person":
            center_x = (x1 + x2) // 2
            cv2.circle(image, (center_x, y1 + 20), 17, color, -1)
            cv2.line(image, (center_x, y1 + 38), (center_x, y2 - 40), color, 5)
            cv2.line(image, (center_x, y1 + 60), (x1, y1 + 105), color, 5)
            cv2.line(image, (center_x, y1 + 60), (x2, y1 + 105), color, 5)
            cv2.line(image, (center_x, y2 - 40), (x1 + 5, y2), color, 5)
            cv2.line(image, (center_x, y2 - 40), (x2 - 5, y2), color, 5)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def pixel_box(observation, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = observation.bbox
    return (
        int(x1 * (width - 1)),
        int(y1 * (height - 1)),
        int(x2 * (width - 1)),
        int(y2 * (height - 1)),
    )
