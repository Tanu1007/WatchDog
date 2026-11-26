# WatchDog

WatchDog is a campus CCTV research prototype for tracking people and portable items, retaining owner–item associations, and producing reviewable theft-risk warnings with evidence clips.

The first milestone implements an interpretable baseline. It binds an item to a consistently co-moving person, remembers the owner after the item is deposited, and raises one alert for a sustained nearby interaction by another person. The temporal model and dataset evaluation are upcoming milestones.

## Quick start

Use Python 3.10 or newer. The trajectory demo and replay engine run with the Python standard library.

```bash
python -m venv .venv
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -e .
watchdog demo
python -m unittest discover -s tests -v
```

The demo runs four ten-second scenarios. The expected alert counts are zero for owner retrieval, brief benign interaction, and short occlusion, and one for sustained suspicious interaction followed by pickup. The fixture starts item displacement at 7 seconds, allowing the alert's lead time to be checked against a known event.

## Video demo

```bash
python -m pip install -e '.[video]'
watchdog demo --video --output runs/demo
```

Every invocation creates a new directory beneath the selected output location. Open `annotated.mp4` in a scenario directory to see track IDs, owner associations, and state changes. The suspicious-pickup scenario also saves an evidence clip containing two seconds before the decision and one second after it.

These animations use synthetic tracks supplied directly to the rules engine. They exercise the decision and evidence pipeline. Detection accuracy and performance on campus footage require separate evaluation.

## Process a recorded video

```bash
python -m pip install -e '.[vision]'
watchdog video --source data/sample.mp4 --model yolo11n.pt --tracker botsort.yaml --device cpu
```

The adapter uses YOLO11 and BoT-SORT. Pass `--tracker bytetrack.yaml` to compare trackers. The default object classes are person, backpack, handbag, and suitcase. Custom checkpoints need these class names. Ultralytics downloads the pretrained model when the specified checkpoint is absent. GPU environments can select `--device 0` and install the PyTorch build appropriate for their hardware.

This initial runner supports recorded, constant-frame-rate footage. Timestamps use the decoded frame index and declared source FPS. Use `--fps` when a source has incorrect FPS metadata. A separate tracker instance is created for each run. Live capture, variable-frame-rate timestamp handling, homography, and wrist keypoints are future work.

The adapter follows Ultralytics' documented persistent-tracking interface. [Tracking documentation](https://docs.ultralytics.com/modes/track/) · [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)

## Output files

| Output | Contents |
| --- | --- |
| `tracks.jsonl` | Source timestamps and normalized tracked boxes for deterministic replay |
| `states.jsonl` | Per-frame item state, owner and actor track IDs, and heuristic score |
| `events.json` | Final event records with cue time, decision time, dispatch time, and evidence status |
| `summary.json` | Run status, frame count, settings, source metadata, and elapsed processing time |
| `annotated.mp4` | Overlay video when image frames are available |
| `clips/<event_id>.mp4` | Evidence for each alert when image frames are available |

The CLI prints each event when it is emitted. Evidence status becomes `ready` after the requested post-event frames are available. A clip cut short by end-of-file has status `truncated` and remains playable. A failed run records its error in `summary.json`. An incomplete full-video recording retains the `.partial.mp4` filename.

Cue and decision times use source-video seconds. Dispatch time uses UTC wall-clock time. For unpaced replay, evaluate cue-to-decision delay on the source timeline. The heuristic risk score measures persistence toward the configured threshold and has no probability calibration.

## Replay and settings

```bash
watchdog replay --tracks runs/<run>/<scenario>/tracks.jsonl
watchdog config > rules.json
watchdog demo --config rules.json
```

Each input line contains one frame. Track IDs must be unique within the source, and timestamps must strictly increase.

```json
{"timestamp": 0.0, "observations": [{"track_id": 17, "kind": "person", "bbox": [0.1, 0.2, 0.3, 0.8], "confidence": 0.95}]}
```

Boxes use `[x1, y1, x2, y2]`, normalized by frame width and height. Distances are minimum box gaps in that coordinate system. Speeds are changes in normalized box-center coordinates per second. The initial thresholds are configurable research defaults that require validation for each camera geometry.

| Rule | Default |
| --- | --- |
| Sustained co-motion before binding | 0.8 seconds |
| Sustained non-owner interaction before alert | 1.2 seconds |
| Brief occlusion grace | 0.75 seconds |
| Track retention | 10 seconds |
| Clear interval before another alert | 1 second |

Items begin with an unknown owner. Binding requires observed co-motion and an unambiguous candidate. Brief missing observations preserve the binding while interrupting accumulation of interaction evidence. A returning owner clears the active warning state. IDs are anonymous and scoped to one source. A track-ID switch requires a new association in this baseline.

The current proximity rule can confuse friendly handoffs with theft and depends on bounding-box quality. Owner reassignment, calibrated ground distance, pose cues, and learned interactions need labeled examples and further evaluation.

## Development plan

The project brief and targets are in [WatchDog_Project_Context.md](WatchDog_Project_Context.md).

1. Collect a small set of staged or authorized campus clips and annotate tracks, owner intervals, benign interactions, cue times, and displacement times.
2. Run the recorded-video pipeline and measure tracking continuity, binding coverage, false alerts, and warning lead time. Calibrate thresholds using a validation split.
3. Define the 12-dimensional pairwise feature schema, missing-data masks, and 60-frame rolling windows.
4. Add the dual-head pairwise ST-LSTM, dataset adapters, training, and baseline/ablation evaluation.
5. Add live-stream ingestion and benchmark end-to-end dispatch on the intended hardware.

The deck's IDF1, owner accuracy, latency, FPS, and false-alarm goals remain evaluation targets. Current automated checks cover the synthetic baseline and output integrity.

Ankita Dan leads data preparation and label QA. Tanvi Patel leads the ST-LSTM and metric analysis. Dushyant Betala leads tracking, heuristics, integration, and the live demo.
