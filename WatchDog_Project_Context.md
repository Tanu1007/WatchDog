# WatchDog project context

WatchDog is a campus CCTV project that aims to warn security operators of possible item theft while there is still time to respond. The system tracks people and portable items such as bags, remembers who carried or deposited an item, and flags a suspicious interaction by another person. A warning should include a saved evidence clip for human review. The central challenge is that an owner retrieving a bag and someone else picking it up can look identical in a single frame.

**Source.** The eight-slide deck `WatchDog CCTV Theft and Anomaly Detection.pptx`. It describes a proposed system, experimental plan, and target metrics. Repository contents, footage, checkpoints, and measured results need to be confirmed.

## Where the coding agent should start

Inspect the repository, run its documented entry point if available, and find the first incomplete milestone. Reuse the existing stack and interfaces where they serve the design below. If the project is greenfield, start with the recorded-video baseline described here. Build an observable path from tracks to a reviewable alert before training the temporal model.

## Architecture from the deck

1. **Perception and multi-object tracking.** Detect people and items and assign stable track IDs. Preserve associations through short occlusions where the tracker can recover the track. With camera calibration, project locations onto the ground plane via homography to estimate metric distance. The architecture slide names YOLOv11 with BoT-SORT. The experimental slides name YOLO or RT-DETR with ByteTrack. Reuse the repository's tracker if present. For a new video prototype, start with YOLOv11 and BoT-SORT to follow the architecture slide and keep the tracker interface replaceable for experiments.
2. **Pairwise relationship modeling.** Keep a trailing history for each relevant person–item pair. The planned pairwise spatio-temporal LSTM has two heads for ownership affinity and interaction state. The architecture slide proposes a bidirectional LSTM. At inference, both directions can run over the already-observed window ending at the score timestamp. Use anonymous track IDs scoped to the camera or video.
3. **Alert decision.** Use a deterministic finite-state machine (FSM) to combine binding history, temporal persistence, and interaction cues. The rule-based score also serves as the initial baseline. Once the LSTM is available, evaluate its contribution under the same alert policy and evaluation protocol. Emit a potential-theft event with the associated tracks and evidence clip.

**Windows.** The experimental setup specifies 60 frames sampled at 15 FPS, or about four seconds, with 50% overlap for training. Keep a rolling window in live mode and configure scoring cadence independently from the training stride. Capture source timestamps for cues and decisions and wall-clock timestamps for dispatch. Evaluate warm-up separately from steady-state latency, and finalize any post-alert clip footage asynchronously if needed.

**Features.** The architecture calls for a 12-dimensional vector and explicitly names ground-plane distance, cosine similarity of person and item velocities, and the minimum wrist-to-item distance. Other slides mention normalized boxes, relative position and speed, box IoU, duration, pose, owner distance, and item motion. Confirm the 12 fields, order, units, and missing-value handling before training. Tag uncalibrated distances as image-space measurements and record missing pose or track observations.

## Suggested first milestone

Build a recorded-video demo that runs the rule-based pipeline and can replay fixture tracks for deterministic tests. Show person and item track IDs, the current binding, and the alert state. Keep model weights, thresholds, and video inputs configurable.

| State | Proposed baseline behavior |
| --- | --- |
| Owned | Bind the item after a nearby person and the item move together consistently. Retain the binding through brief detection gaps. |
| Deposited | Keep the binding after the item becomes stationary and the owner moves away. |
| Guarded | Monitor the deposited item for another person's sustained approach, reach, or contact. Handle brief nearby activity and owner retrieval without raising a warning. |
| Alert | Emit one warning after a sustained interaction crosses the configured risk threshold. Record the trigger and allow a later interaction to generate a new event once the current one ends. |

Use fixture tracks to check owner retrieval, a brief benign interaction, a short occlusion, and a sustained non-owner reach or pickup. The first three should complete without alerts. The sustained interaction should produce one event and, in video mode, a playable evidence clip. Include source timestamps so the demo can show whether the warning preceded displacement.

Suggested event fields are `event_id`, `camera_id` or `video_id`, `item_track_id`, `owner_track_id`, `actor_track_id`, `decision_timestamp`, `dispatch_timestamp`, `risk_score`, `trigger_reason`, and `evidence_clip_path`. Record an unknown owner explicitly when a track cannot be associated reliably. Provide a reproducible command, example configuration, and observed test results with the milestone.

## Data and training plan

| Dataset | Role in the deck |
| --- | --- |
| UCF-Crime | Pretrain anomaly-aware temporal features. If using the theft-related subset for comparison, reserve those videos before pretraining. |
| VIRAT | Pretrain trajectory and human–object interaction features. Its activity labels supervise interactions such as carrying or loading. Use campus or benchmark labels for theft-specific supervision. |
| PETS 2006 | Held-out luggage abandonment and theft benchmark in the experimental plan. Verify the available sequences and event annotations before scoring. |
| Campus footage | Fine-tune and evaluate owner binding and early warnings using authorized or staged clips with campus-specific labels. |

For campus annotations, propose person and item track IDs, owner binding intervals, interaction label, earliest observable cue, and item displacement time when present. Include ordinary pickups and benign approaches. Split whole clips or scenes before making overlapping windows so sequences from the same incident stay together. Adopt the deck's initial training settings where applicable. It proposes AdamW, learning rate `1e-3`, cosine annealing, and a weighted loss. The hardware plan lists A100 training and RTX 4090 or Jetson edge tests.

## Evaluation

| Metric | Deck target | Suggested measurement |
| --- | --- | --- |
| Stream ingestion | ≥15 FPS | Report input and processed FPS, frame drops, video resolution, and hardware. The related-work slide mentions 25+ FPS as an unverified figure. |
| Tracking IDF1 | ≥80% | Measure on held-out videos with annotated person and item tracks. |
| Owner-binding accuracy | ≥90% | Score predicted associations against annotated person–item bindings. Report coverage and how unknown owners are counted. |
| Alert latency | <2.5 s | Measure from the annotated actionable cue to dispatch on a shared clock in live or real-time playback. For unpaced replay, report cue-to-decision time on the video timeline and dispatch overhead separately. |
| False alarms | ≤0.5 per hour | Count unmatched alert events per evaluated camera-hour under a documented event-matching rule. |

Report the latency distribution and number of matched and missed events. Report lead time as displacement time minus alert time on the same timeline. Positive values mean the alert was early. The deck also calls for detection F1, PR-AUC, HOTA, MOTA, ID switches, and FPS.

**Comparisons.** Use the rule-based heuristic, spatio-temporal autoencoder, person–item ST-GCN, LSTM without binding, and the proposed pairwise ST-LSTM. Planned ablations remove VIRAT pretraining, UCF-Crime pretraining, tracking, velocity features, and the LSTM one at a time where applicable.

## Project milestones and owners

| Proposal schedule | Milestone | Lead responsibilities |
| --- | --- | --- |
| Weeks 1–2 | Data and labels | Ankita Dan handles data preparation, label mapping, and dataset QA. |
| Weeks 3–4 | Tracking and heuristics | Dushyant Betala handles tracking, heuristics, and integration. |
| Week 5 | ST-LSTM training | Tanvi Patel handles the pairwise ST-LSTM, training, and metric analysis. |
| Week 6 | Evaluation and demo | Tanvi handles metric analysis and Dushyant handles the live demo. |

Final deliverables are the detection, tracking, and trained ST-LSTM pipeline, reproducible preprocessing and evaluation scripts, a baseline and ablation report, and a live demo that shows owner–item pairs and warning clips.

## Decisions to confirm as work progresses

- Which repository, sample clips, annotations, and checkpoints are available.
- Which item classes and camera inputs the demo should support, and whether homography calibration and wrist keypoints are available.
- What defines the actionable cue, a benign handoff, and a persistent interaction. Set thresholds using validation clips and document the event-matching rule.
- Which hardware and stream count govern the runtime targets, and how operators should receive and retain evidence in a live deployment.
