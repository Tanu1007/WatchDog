"""Build the clip-level split manifest and UCF-Crime temporal labels.

Splits are assigned per whole clip so overlapping training windows never cross
splits. VIRAT keeps the DIVA train/validate folders. UCF-Crime keeps the
official Anomaly_Train/Anomaly_Test lists, read straight from UCF_Crimes.zip.

Writes <data>/splits.csv and <data>/annotations/ucf_crime_temporal.json.
With --verify, also checks every video against its expected size and adds
FPS and second-based windows to the UCF-Crime labels.

Usage:
    python scripts/build_manifest.py --data D:/watchdog-data [--verify]
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

FIELDS = (
    "dataset",
    "clip_id",
    "split",
    "category",
    "video_path",
    "video_bytes",
    "annotation_path",
)


def virat_rows(raw):
    annotations = raw / "virat" / "diva_annotations"
    videos = raw / "virat" / "videos"
    rows = []
    for geometry in sorted(annotations.glob("*/*.geom.yml")):
        clip = geometry.name.split(".")[0]
        video = videos / f"{clip}.mp4"
        rows.append(
            {
                "dataset": "virat",
                "clip_id": clip,
                "split": "train" if geometry.parent.name == "train" else "validation",
                "category": "",
                "video_path": video.relative_to(raw.parent).as_posix() if video.exists() else "",
                "video_bytes": video.stat().st_size if video.exists() else "",
                "annotation_path": geometry.parent.relative_to(raw.parent).as_posix(),
            }
        )
    return rows


def ucf_rows(raw):
    root = raw / "ucf_crime"
    with zipfile.ZipFile(root / "UCF_Crimes.zip") as archive:
        sizes = {info.filename: info.file_size for info in archive.infolist()}
        rows = []
        for split, listing in (("train", "Anomaly_Train.txt"), ("test", "Anomaly_Test.txt")):
            text = archive.read(f"UCF_Crimes/Anomaly_Detection_splits/{listing}").decode()
            for entry in text.split():
                member = f"UCF_Crimes/Videos/{entry}"
                folder, filename = entry.split("/")
                rows.append(
                    {
                        "dataset": "ucf_crime",
                        "clip_id": filename.removesuffix(".mp4"),
                        "split": split,
                        "category": "Normal" if "Normal" in folder else folder,
                        "video_path": (root / member).relative_to(raw.parent).as_posix(),
                        "video_bytes": sizes[member],
                        "annotation_path": "annotations/ucf_crime_temporal.json" if split == "test" else "",
                    }
                )
    return rows


def ucf_temporal(raw):
    """Parse test-video anomaly windows; frames use the video's own frame index."""
    path = next((raw / "ucf_crime").glob("Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/*.txt"))
    labels = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        name, category, *bounds = fields
        bounds = [int(value) for value in bounds]
        windows = [
            {"start_frame": start, "end_frame": end}
            for start, end in (bounds[:2], bounds[2:])
            if start >= 0 and end >= 0
        ]
        labels[name.removesuffix(".mp4")] = {"category": category, "anomaly_windows": windows}
    return labels


def verify(rows, data, labels):
    import cv2

    problems = []
    for row in rows:
        if not row["video_path"]:
            continue
        video = data / row["video_path"]
        if not video.exists():
            problems.append(f"missing: {row['video_path']}")
        elif video.stat().st_size != int(row["video_bytes"]):
            problems.append(f"size mismatch: {row['video_path']}")
        elif row["clip_id"] in labels:
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if not fps or fps <= 0:
                problems.append(f"unreadable FPS: {row['video_path']}")
                continue
            label = labels[row["clip_id"]]
            label.update(fps=fps, frame_count=frames, duration_seconds=frames / fps)
            for window in label["anomaly_windows"]:
                window["start_timestamp"] = window["start_frame"] / fps
                window["end_timestamp"] = window["end_frame"] / fps
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="root holding raw/")
    parser.add_argument("--verify", action="store_true", help="check videos and add FPS-based times")
    args = parser.parse_args()
    raw = args.data / "raw"

    rows = virat_rows(raw) + ucf_rows(raw)
    labels = ucf_temporal(raw)
    test_clips = {row["clip_id"] for row in rows if row["dataset"] == "ucf_crime" and row["split"] == "test"}
    unmatched = sorted(set(labels) ^ test_clips)

    problems = verify(rows, args.data, labels) if args.verify else []

    with (args.data / "splits.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.data / "annotations").mkdir(exist_ok=True)
    (args.data / "annotations" / "ucf_crime_temporal.json").write_text(
        json.dumps(labels, indent=1) + "\n", encoding="utf-8"
    )

    for dataset in ("virat", "ucf_crime"):
        subset = [row for row in rows if row["dataset"] == dataset]
        counts = {split: sum(row["split"] == split for row in subset) for split in sorted({r["split"] for r in subset})}
        print(f"{dataset}: {len(subset)} clips {counts}, {sum(bool(r['video_path']) for r in subset)} with video")
    print(f"ucf_crime temporal labels: {len(labels)}, {sum(bool(l['anomaly_windows']) for l in labels.values())} with anomaly windows")
    if unmatched:
        problems.append(f"labels and test split disagree on {len(unmatched)} clips: {unmatched[:5]}")
    for problem in problems[:20]:
        print(problem)
    if args.verify:
        print(f"verify: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
