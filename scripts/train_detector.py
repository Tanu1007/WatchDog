"""Fine-tune a YOLO detector on the dataset from scripts/export_yolo_dataset.py.

The trained classes are person and item, which UltralyticsTracker maps to the
engine's kinds directly. Checkpoints are chosen on the YOLO val clips (held out
from the train split), never on the VIRAT validation split. Pass --resume with
the run's last.pt to continue an interrupted run.

Usage:
    python scripts/train_detector.py --dataset D:/watchdog-data/yolo/virat_person_item/dataset.yaml
        --model yolo11m.pt --imgsz 1280 --epochs 30 --batch 4
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="dataset.yaml")
    parser.add_argument("--model", default="yolo11m.pt", help="starting weights (COCO-pretrained)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, help="default: <dataset root>/../../runs/detect")
    parser.add_argument("--name", default="virat_person_item")
    parser.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return 0
    project = args.project or args.dataset.resolve().parents[2] / "runs" / "detect"
    YOLO(args.model).train(
        data=str(args.dataset),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=False,
        seed=0,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
