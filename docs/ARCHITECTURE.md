# WatchDog architecture

Diagrams of the planned system (from the project deck) and of what the repository implements today. In the diagrams, **solid green** nodes are implemented and **dashed grey** nodes are planned. GitHub, VS Code (with a Mermaid extension) and most Markdown viewers render these diagrams.

## 1. End-to-end system

```mermaid
flowchart LR
    classDef done fill:#d9f2e3,stroke:#2e7d4f,color:#123;
    classDef todo fill:#f2f2f2,stroke:#999,stroke-dasharray:5 4,color:#555;

    CAM[/"CCTV stream<br/>or recorded video"/]:::done
    DET["YOLO11 detector<br/>COCO: person + bags, or<br/>fine-tuned on VIRAT: person + item"]:::done
    TRK["BoT-SORT / ByteTrack<br/>persistent track IDs"]:::done
    HOM["Homography<br/>ground-plane distance"]:::todo
    POSE["Pose / wrist keypoints"]:::todo
    FEAT["Pairwise features<br/>14-D per person–item pair"]:::done
    LSTM["Pairwise ST-LSTM<br/>ownership head + interaction head"]:::done
    RULES["Rule-based scorer<br/>co-motion binding, proximity persistence"]:::done
    FSM["Alert FSM<br/>owned → deposited → guarded → alert"]:::done
    EVT["Event record<br/>events.json"]:::done
    CLIP["Evidence clip<br/>2 s before, 1 s after"]:::done
    OPS(["Security operator"]):::todo

    CAM --> DET --> TRK
    TRK --> RULES --> FSM
    TRK -.-> HOM -.-> FEAT
    TRK -.-> POSE -.-> FEAT
    TRK --> FEAT --> LSTM -->|learned / hybrid binding| FSM
    FSM --> EVT --> OPS
    FSM --> CLIP --> OPS
```

The rule-based scorer is the baseline. When the ST-LSTM is ready, it feeds the same FSM, so both can be compared under one alert policy.

## 2. Alert state machine (as implemented in `engine.py`)

```mermaid
stateDiagram-v2
    [*] --> unknown: item track appears
    unknown --> owned: one nearby person co-moves with the item<br/>for ≥ 0.8 s (unambiguous)
    owned --> deposited: item stationary and<br/>owner ≥ 0.2 away (or gone)
    deposited --> guarded: non-owner within contact distance
    guarded --> deposited: contact ends for ≥ 1 s
    guarded --> alert: contact persists ≥ 1.2 s<br/>(emit one Alert + evidence clip)
    alert --> deposited: contact ends for ≥ 1 s<br/>(re-armed for a new event)
    deposited --> owned: owner returns within bind distance
    guarded --> owned: owner returns
    alert --> owned: owner returns
    note right of deposited
        Owner hidden < 0.75 s (occlusion grace):
        binding kept, interaction timer paused
    end note
```

Distances are minimum box gaps in normalized image coordinates. All thresholds live in `config.Rules` and can be overridden with `watchdog config > rules.json`.

## 3. Code structure

```mermaid
flowchart TB
    classDef done fill:#d9f2e3,stroke:#2e7d4f,color:#123;
    classDef new fill:#dbe9ff,stroke:#2b5fb4,color:#123;

    CLI["cli.py<br/>demo · replay · video · config"]:::done
    SCN["scenarios.py<br/>synthetic fixtures"]:::done
    VID["video.py / tracking.py<br/>Ultralytics adapter"]:::done
    PIPE["pipeline.py<br/>read_tracks · run_frames · outputs"]:::done
    ENG["engine.py<br/>OwnershipEngine FSM"]:::done
    CFG["config.py<br/>Rules"]:::done
    TYP["types.py<br/>Observation · Frame · Alert"]:::done
    VIR["virat.py<br/>VIRAT → Frames"]:::new
    EVA["evaluation.py<br/>IDF1 · binding · alert scoring"]:::new
    PW["pairwise.py<br/>14-D features · windows · labels"]:::new
    S9["scripts/build_windows.py"]:::new
    ST["stlstm.py<br/>PairwiseSTLSTM · metrics"]:::new
    S10["scripts/train_stlstm.py<br/>scripts/evaluate_stlstm.py"]:::new
    S11["scripts/evaluate_binding.py<br/>rules vs per-pair ST-LSTM"]:::new
    S1["scripts/download_virat.py"]:::new
    S2["scripts/virat_to_tracks.py"]:::new
    S3["scripts/extract_ucf.py"]:::new
    S4["scripts/build_manifest.py"]:::new
    S5["scripts/track_videos.py<br/>batch YOLO + tracker"]:::new
    S6["scripts/evaluate_virat.py<br/>scripts/tune_rules.py"]:::new
    S7["scripts/evaluate_ucf.py"]:::new
    S8["scripts/export_yolo_dataset.py<br/>scripts/train_detector.py<br/>scripts/training_log.py"]:::new

    CLI --> SCN & VID & PIPE
    PIPE --> ENG --> CFG
    ENG --> TYP
    VID --> TYP
    S2 --> VIR --> TYP
    S2 -->|tracks.jsonl| CLI
    S5 --> VID
    S6 --> EVA
    S7 --> EVA
    EVA --> ENG & VIR
    S8 --> VIR
    S8 -->|fine-tuned weights| S5
    S9 --> PW --> TYP
    S9 --> EVA
    S10 --> ST --> PW
    S11 --> ST
    S11 --> EVA
```

Blue nodes were added in this session (see [WORKLOG.md](WORKLOG.md)). `tracking.py` also gained `--imgsz`, open-vocabulary `--item-classes` (YOLO-World) and a fix for tracks that change class mid-sequence.

## 4. Data pipeline

```mermaid
flowchart LR
    classDef src fill:#fff4d6,stroke:#b38600,color:#123;
    classDef done fill:#d9f2e3,stroke:#2e7d4f,color:#123;
    classDef todo fill:#f2f2f2,stroke:#999,stroke-dasharray:5 4,color:#555;

    K[("Kitware Girder<br/>VIRAT 2.0 videos")]:::src
    G[("Kitware GitLab<br/>DIVA annotations")]:::src
    U[("UCF CRCV<br/>UCF_Crimes.zip 103 GB")]:::src

    K -->|download_virat.py| VV["79 videos<br/>17.2 GB"]:::done
    G --> VA["KPF geom / types / activities"]:::done
    U -->|curl resume| UZ["zip, size verified"]:::done
    UZ -->|extract_ucf.py| UV["1,950 videos<br/>+ splits"]:::done

    VA & VV -->|virat_to_tracks.py| TJ["tracks.jsonl<br/>+ activities.json"]:::done
    TJ -->|watchdog replay| RUN["states · events · summary"]:::done
    VA & UV & UZ -->|build_manifest.py| SPL["splits.csv<br/>ucf_crime_temporal.json"]:::done

    VA & VV -->|export_yolo_dataset.py<br/>train split only| YD["YOLO dataset<br/>7,260 + 1,329 images"]:::done
    YD -->|train_detector.py| FT["yolo11m fine-tuned<br/>person + item"]:::done
    VV & FT -->|track_videos.py| DT["YOLO + BoT-SORT tracks"]:::done
    DT -->|evaluate_virat.py| EV["IDF1 · owner binding<br/>vs VIRAT ground truth"]:::done
    TJ -->|tune_rules.py| RS["tuned rule sets<br/>configs/*.json"]:::done
    RS --> EV
    UV -->|track_videos.py| UT["UCF tracks"]:::done
    UT -->|evaluate_ucf.py| UA["alerts vs anomaly windows<br/>false alarms/hour, lead time"]:::todo
    TJ -->|build_windows.py| WIN["60-frame pairwise windows<br/>14 features + 3 masks"]:::done
    WIN -->|train_stlstm.py| TR["ST-LSTM<br/>ownership + interaction"]:::done
    TR -->|evaluate_binding.py| PB["per-pair binding<br/>best score so far"]:::done
    PB -->|ScheduledBinder| FSMX["alert FSM with<br/>rules / learned / hybrid binding"]:::done
```

UCF tracking is running; the UCF alert scoring is built and tested but not yet run on real tracks.

## 5. Datasets and what each provides

| Dataset | Used for | Labels available | What it doesn't have |
| --- | --- | --- | --- |
| VIRAT Ground 2.0 + DIVA | Owner binding, interaction learning, tracking evaluation | Per-frame person and object boxes with IDs; 40 activity types incl. PickUp, SetDown, Object_Transfer, carrying | Theft events; bags are small and far from the camera |
| UCF-Crime | Anomaly pretraining; alert timing and false-alarm evaluation | Video-level class (13 anomalies + normal); frame windows for the 290 test videos | Boxes or tracks (we generate them with YOLO) |
| PETS 2006 | Planned luggage benchmark | — | Unavailable (hosts offline) |
| Campus footage | Planned fine-tuning | — | Out of scope for now |
