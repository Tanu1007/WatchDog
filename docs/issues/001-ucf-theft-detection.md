# UCF-Crime: the alert rule misses real thefts, and ST-LSTM binding adds false alarms there

**Status:** open (pending). Found on 25 Nov 2026 in the UCF-Crime test evaluation.

## What happens

On the UCF-Crime test split (52 labelled theft events in 44 videos, plus 6 hours of normal footage), WatchDog almost never raises an alert for a real theft:

| Binding mode | Thefts detected | False alarms on 6 h of normal video |
| --- | --- | --- |
| Rules | 0 of 52 | 0 |
| Hybrid ST-LSTM (5 seeds) | 0–1 of 52 | 2–6 |

On VIRAT the same system works well (owners found for 12 of 13 items, no false alarms), so the problem is specific to the kind of theft UCF-Crime shows.

## Why it fails, in plain language

WatchDog raises an alert for one specific story:

1. A person carries an item, so the system learns **"this bag belongs to this person"**.
2. The owner puts the item down and walks away.
3. **Someone else** stays at the item for more than about a second.

If the video doesn't show step 1, the system never knows who owns the item, so step 3 never looks suspicious.

### Scenario A: what WatchDog catches (the campus case)

> A student walks into the library carrying a backpack and puts it on a chair, then goes to the printer. A stranger walks up and picks up the backpack.

WatchDog sees the student carry the backpack (owner = the student), sees it put down while the student walks away, then sees a different person handling it. **Alert.**

### Scenario B: shoplifting, which WatchDog misses

> In a convenience store, a man takes a bottle off the shelf, slips it into his jacket and walks out without paying.

Nobody carried the bottle into the scene. It has been on the shelf the whole time, so WatchDog never assigns it an owner. The man is the *first* person to touch it, which looks exactly like an ordinary customer picking something up. **No alert.** The same applies to burglary: the TV in an empty house has no owner visible on camera.

### Scenario C: the item is never seen

> A grainy 320×240 camera shows someone grabbing a phone from a counter.

At this resolution the phone is a few pixels wide and is not one of the detector's item classes (it was trained on backpacks, handbags, suitcases and VIRAT's bags, boxes and carts). Items were detected in only 106 of the 194 UCF videos. No item, no owner, no alert.

### Scenario D: a false alarm from hybrid binding

> In a normal video, two friends stand next to a bag on a bench and talk. One of them leans over it for a couple of seconds.

The ST-LSTM was trained on clear, high-resolution VIRAT footage. On UCF's noisy video it is more willing to decide "the bag belongs to friend 1" just because friend 1 stood next to it. When friend 2 leans over the bag, the alert rule sees "a non-owner at the owner's item" and fires. The rules alone don't bind here, so they stay silent.

## Why it matters

The system as built is an **owner-protection** alarm (someone takes *your* bag). Most UCF-Crime thefts are **property thefts** with no owner on screen. We can't claim theft detection on UCF-Crime yet, and hybrid binding should not be used on unfamiliar, low-resolution cameras without tuning.

## Possible fixes

- [ ] A theft cue that doesn't need a visible owner: a person removing an item from where it rested (shelf, counter, floor) and leaving, using the ST-LSTM interaction head and item displacement.
- [ ] A stricter hybrid binding threshold (or rules-only binding) on unfamiliar or low-resolution cameras, to remove the extra false alarms.
- [ ] UCF-Crime anomaly pretraining, as planned in the project deck.
- [ ] Labelled campus clips (currently out of scope) to evaluate the real deployment case.

## Evidence

- Results and analysis: `docs/WORKLOG.md`, section "UCF-Crime alert results".
- Reports: `docs/logs/ucf_rules_default.json`, `docs/logs/ucf_rules_moderate.json`, `docs/logs/ucf_hybrid_moderate_seed{0..4}.json`.
- Reproduce: `docs/SETUP.md`, section 8 (`scripts/evaluate_ucf.py`, with `--binding hybrid --stlstm ...` for the ST-LSTM).
