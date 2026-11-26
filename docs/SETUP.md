# Setup guide: reproduce the WatchDog environment and datasets

Follow these steps to reproduce the environment and data used so far. They were tested on Windows 11 with an RTX 5060 Laptop GPU. Commands are for **PowerShell**, run from the repository root (the folder containing `pyproject.toml`). The work log is in [WORKLOG.md](WORKLOG.md).

## 0. What you need

| Requirement | Notes |
| --- | --- |
| Python 3.10+ | Tested with 3.11.9 |
| NVIDIA GPU and recent driver | Optional; CPU works but is slow for YOLO. The driver must support CUDA 12.8 for the cu128 build. |
| FFmpeg | `winget install Gyan.FFmpeg`, then open a new terminal |
| About 260 GB of free disk | VIRAT 17 GB, UCF-Crime zip 103 GB, extracted UCF-Crime 103 GB, plus tracks and runs. An external NTFS drive works. |
| A fast connection | Observed: UCF server about 38 MB/s, Kitware (VIRAT) about 5 MB/s per stream |

## 1. Python environment (≈10 min)

```powershell
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1

# Install PyTorch FIRST. On Windows, the PyPI wheel is CPU-only.
#   RTX 50xx (Blackwell): cu128 is required.
#   Older GPUs: cu126 also works.  No GPU: plain `pip install torch torchvision`.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e ".[vision]"
```

The exact versions used are in [`requirements-lock.txt`](requirements-lock.txt).

> ⚠️ Don't run two `pip` commands in the same venv at the same time. It can break pip itself. If that happens, run `python -m ensurepip --upgrade`, then delete any `~ip*` folders in `.venv\Lib\site-packages`.

### Check it works (≈2 min)

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
watchdog demo                                   # expect alert counts 0, 0, 0, 1
python -m unittest discover -s tests -v         # expect all tests OK
ffmpeg -version
```

## 2. Dataset terms (do this before downloading)

- **VIRAT:** every person with access must accept the [VIRAT Video Dataset Usage Agreement](https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf). In short: research use is allowed, protect personally identifiable information and don't try to identify people, and don't redistribute except to others who have agreed (a few frames in papers or slides are fine).
- **UCF-Crime:** the project page states no licence; cite Sultani, Chen and Shah, *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018.
- **PETS 2006** is not available: every official host is offline as of November 2026.

## 3. Data layout

Choose a data root. The commands below use `D:\watchdog-data`; change `$DATA` if your drive letter differs.

```text
$DATA/
├── raw/
│   ├── virat/
│   │   ├── videos/                  79 .mp4 + missing_videos.txt
│   │   ├── diva_annotations/        train/ validate/ (KPF .yml)
│   │   └── release2_annotations/    annotations/ docs/ homographies/ software/
│   └── ucf_crime/
│       ├── UCF_Crimes.zip           103 GB (can be deleted after verification)
│       ├── UCF_Crimes/              Videos/ Anomaly_Detection_splits/ ...
│       └── Temporal_Anomaly_Annotation_For_Testing_Videos/
├── tracks/virat_diva/               <clip>.tracks.jsonl + <clip>.activities.json
├── annotations/ucf_crime_temporal.json
├── splits.csv
└── runs/                            watchdog replay or video outputs
```

```powershell
$DATA = "D:\watchdog-data"
New-Item -ItemType Directory -Force "$DATA\raw\virat", "$DATA\raw\ucf_crime" | Out-Null
```

## 4. VIRAT (≈15 min)

```powershell
# DIVA annotations (activity labels incl. PickUp / SetDown / Object_Transfer), about 194 MB
curl.exe -L -o "$DATA\raw\virat\diva.zip" https://gitlab.kitware.com/viratdata/viratannotations/-/archive/master/viratannotations-master.zip
Expand-Archive "$DATA\raw\virat\diva.zip" "$DATA\raw\virat\_diva"
Move-Item "$DATA\raw\virat\_diva\viratannotations-master" "$DATA\raw\virat\diva_annotations"

# Original Release 2.0 annotations and docs, 41 MB (optional; needed for --format release2)
curl.exe -L -o "$DATA\raw\virat\annotations-and-docs.zip" https://data.kitware.com/api/v1/item/67914e81f07d3010406a7d21/download
Expand-Archive "$DATA\raw\virat\annotations-and-docs.zip" "$DATA\raw\virat\release2_annotations"

# The 79 published videos that have DIVA annotations (17.2 GB). Resumable; re-run if interrupted.
python scripts\download_virat.py --annotations "$DATA\raw\virat\diva_annotations" --output "$DATA\raw\virat\videos"
```

Some entries in the Release 2 zip are empty folders where annotation files should be. The scripts skip them.

## 5. UCF-Crime (≈45 min download + ≈60 min extraction)

```powershell
$UCF = "$DATA\raw\ucf_crime"
curl.exe -L -o "$UCF\Temporal_Anomaly_Annotation_For_Testing_Videos.zip" https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip
curl.exe -L -o "$UCF\UCF_Crimes_Instruction.docx" https://www.crcv.ucf.edu/projects/real-world/UCF_Crimes_Instruction.docx

# 102,957,372,377 bytes. -C - resumes; re-run the same line if it stops.
curl.exe -L -C - --retry 10 --retry-all-errors -o "$UCF\UCF_Crimes.zip" https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip
(Get-Item "$UCF\UCF_Crimes.zip").Length          # must print 102957372377

# Resumable extraction with checksum verification
python scripts\extract_ucf.py --zip "$UCF\UCF_Crimes.zip"
```

If the extraction stops (for example because memory runs low), just run it again. Finished files are skipped.

## 6. Convert, build splits, verify (≈5 min)

```powershell
# VIRAT ground truth -> replayable tracks at 15 FPS
python scripts\virat_to_tracks.py --format diva --annotations "$DATA\raw\virat\diva_annotations" `
    --videos "$DATA\raw\virat\videos" --output "$DATA\tracks\virat_diva" --stride 2

# Clip-level splits and UCF-Crime temporal labels; --verify checks every video's size and adds FPS-based times
python scripts\build_manifest.py --data $DATA --verify

# Sanity check: replay one VIRAT clip through the rules engine
watchdog replay --tracks "$DATA\tracks\virat_diva\VIRAT_S_040001_01_000448_001101.tracks.jsonl" --output "$DATA\runs"
```

Expected results: 79 clips converted, 24 of them with item tracks. `build_manifest` reports VIRAT 119 clips (79 with video) and UCF-Crime 1,900 clips (1,610 train / 290 test), with `verify: 0 problems`.

## 7. Part 2: tracking and evaluation

```powershell
python -m pip install -e ".[vision,eval]"     # adds motmetrics

# Detection and tracking at 15 FPS; tracks only, resumable. yolo11m@1280 was best in the pilot.
python scripts\track_videos.py --data $DATA --dataset virat --split validation --model yolo11m.pt --imgsz 1280 --device 0

# Score tracking (IDF1/MOTA) and owner binding against VIRAT ground truth
python scripts\evaluate_virat.py --data $DATA --split validation `
    --predictions "$DATA\tracks\virat_yolo11m_1280_botsort" --config configs\rules_virat_moderate.json `
    --report "$DATA\eval\validation_yolo11m_1280_moderate.json"

# Omit --predictions to score the rules on ground-truth tracks only (oracle)
# Tune thresholds on train, check on validation:
python scripts\tune_rules.py --data $DATA --tune train --check validation --output "$DATA\eval\tuning.csv"

# Open-vocabulary items (YOLO-World). First use auto-installs the `clip` package and downloads 338 MB of weights.
python scripts\track_videos.py --data $DATA --dataset virat --split validation --model yolov8l-worldv2.pt --imgsz 1280 `
    --confidence 0.1 --item-classes backpack handbag suitcase bag box cart `
    --output "$DATA\tracks\virat_yolov8l-worldv2_1280_c0.1_botsort"

# Fine-tuned person + item detector (≈5 min export; ≈12.5 min per epoch on an 8 GB laptop GPU).
# Batch 4 at 1280 overflows 8 GB of VRAM and runs about 3× slower; use batch 2.
python scripts\export_yolo_dataset.py --data $DATA
python scripts\train_detector.py --dataset "$DATA\yolo\virat_person_item\dataset.yaml" --model yolo11m.pt --imgsz 1280 --epochs 15 --batch 2 --workers 1
# If training stops, resume it:
python scripts\train_detector.py --dataset "$DATA\yolo\virat_person_item\dataset.yaml" --resume "$DATA\runs\detect\virat_person_item\weights\last.pt"
```

Tuned rule sets are in `configs/`. Model weights download automatically on first use (run from `weights\` to keep them there).

## 8. UCF-Crime alert evaluation

```powershell
# Theft-type and Normal test videos (320x240, so imgsz 640 is enough)
python scripts\track_videos.py --data $DATA --dataset ucf_crime --split test `
    --categories Stealing Shoplifting Robbery Burglary Normal --model yolo11m.pt --imgsz 640 --device 0

# Alerts vs labelled windows: recall, lead time, false alerts per hour
python scripts\evaluate_ucf.py --data $DATA --predictions "$DATA\tracks\ucf_crime_yolo11m_640_botsort" `
    --config configs\rules_virat_moderate.json --report "$DATA\eval\ucf_yolo11m_640_moderate.json"
```

## 9. ST-LSTM windows (CPU, ≈30 s)

```powershell
# 60-frame pairwise windows + ownership / interaction labels from DIVA ground truth
python scripts\build_windows.py --data $DATA --split train      --output "$DATA\windows\virat_gt_train"
python scripts\build_windows.py --data $DATA --split validation --output "$DATA\windows\virat_gt_validation"
# From detector tracks instead (labels via the predicted -> ground-truth ID mapping):
python scripts\build_windows.py --data $DATA --split validation `
    --predictions "$DATA\tracks\virat_yolo11m-ft-e15_1280_botsort" --output "$DATA\windows\virat_ft_validation"
```

The feature schema is in [WORKLOG.md](WORKLOG.md#st-lstm-input-schema-agreed-25-nov-1912) and `src/watchdog_cctv/pairwise.py`.

## 10. Train the ST-LSTM (CPU, ≈15 s)

```powershell
python -m pip install -e ".[stlstm]"     # adds scikit-learn
python scripts\train_stlstm.py --train "$DATA\windows\virat_gt_train" --test "$DATA\windows\virat_gt_validation" `
    --select-from "$DATA\yolo\virat_person_item\summary.json" --output "$DATA\runs\stlstm\baseline_gt"
```

Writes `model.pt`, `history.csv`, `metrics.json` (selection and test metrics plus the box-gap rule and majority baselines) and `test_predictions.npz`. `--select-from` reuses the detector's held-out train clips for early stopping; without it every 5th train clip is used.

```powershell
# Several seeds, then mean ± std on ground-truth and detector-track windows, with the gap rule for reference
foreach ($s in 0..4) { python scripts\train_stlstm.py --train "$DATA\windows\virat_gt_train" --test "$DATA\windows\virat_gt_validation" `
    --select-from "$DATA\yolo\virat_person_item\summary.json" --seed $s --output "$DATA\runs\stlstm\gt_seed$s" }
python scripts\evaluate_stlstm.py --runs (0..4 | % { "$DATA\runs\stlstm\gt_seed$_" }) --train "$DATA\windows\virat_gt_train" `
    --windows gt="$DATA\windows\virat_gt_validation" ft="$DATA\windows\virat_ft_validation" --report "$DATA\eval\stlstm_gt_seeds_eval.json"
```

The windows above were built before the two history features were added. With the current code, `build_windows.py` writes 14 features; the current runs use `virat_*_v2` windows and `runs\stlstm\hist_seed*`. Older 12-feature models still load and read the first 12 columns.

```powershell
# Owner binding: rules engine vs ST-LSTM per-pair binding, same metric, ground-truth or detector tracks
python scripts\evaluate_binding.py --data $DATA --split validation --runs (0..4 | % { "$DATA\runs\stlstm\hist_seed$_" }) `
    --config configs\rules_virat_moderate.json --predictions "$DATA\tracks\virat_yolo11m-ft-e15_1280_botsort" `
    --report "$DATA\eval\binding_ft_hist.json"
```

```powershell
# Same alert engine, ST-LSTM binding: --binding learned (LSTM only) or hybrid (LSTM, then rules)
python scripts\evaluate_virat.py --data $DATA --split validation --predictions "$DATA\tracks\virat_yolo11m-ft-e15_1280_botsort" `
    --config configs\rules_virat_moderate.json --binding hybrid --stlstm "$DATA\runs\stlstm\hist_seed0" `
    --report "$DATA\eval\fsm_virat_hybrid_seed0.json"
# evaluate_ucf.py takes the same --binding / --stlstm options
```

## 11. Git

The repository is on `main`. `data/`, `weights/`, `runs/` and `*.pt` are ignored: never commit VIRAT data or weights trained on it (usage agreement). `.gitattributes` stores text with LF endings, so Windows and macOS/Linux clones match.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is False | You installed the CPU wheel. Reinstall from the cu128 (or cu126) index **before** installing the project. |
| `No module named pip` in the venv | `python -m ensurepip --upgrade`, then delete the `~ip*` folders. |
| Extraction or download stops | Re-run the same command; all download and extraction steps resume. |
| The drive letter changed | Set `$DATA` to the new letter. `splits.csv` paths are relative to the data root. |
| `cannot open ... .mp4` during conversion | That clip's video isn't downloaded; check `missing_videos.txt`. |
| `module 'numpy' has no attribute 'asfarray'` | You called `motmetrics.distances.iou_matrix` directly. Use `watchdog_cctv.evaluation.iou_distances`. |
| `track N changed kind within the same source` | You're on code from before the `tracking.py` fix; update. |
| Training is killed, or the PC runs out of RAM | On Windows each data-loader worker holds its own copy of PyTorch. Use `--workers 1` (or 0) and close other programs. |
