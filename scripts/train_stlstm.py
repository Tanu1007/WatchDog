"""Train the pairwise ST-LSTM on windows from scripts/build_windows.py.

Model selection uses held-out clips of the training windows (--select-from the
detector's export summary, so the same clips are held out throughout, or every
--select-every-th clip); the --test windows (VIRAT validation split) are scored
once, with the selected checkpoint and the ownership threshold that maximizes
F1 on the selection clips (stored in model.pt). Settings follow the deck: AdamW,
learning rate 1e-3, cosine annealing, class-weighted losses.

Two references are scored on the same test windows: a box-gap rule for
ownership (mean gap over the window, threshold tuned on the training clips) and
the majority class for interaction.

Usage:
    python scripts/train_stlstm.py --train D:/watchdog-data/windows/virat_gt_train
        --test D:/watchdog-data/windows/virat_gt_validation
        --select-from D:/watchdog-data/yolo/virat_person_item/summary.json
        --output D:/watchdog-data/runs/stlstm/baseline
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from watchdog_cctv import stlstm
from watchdog_cctv.pairwise import FEATURE_NAMES, INTERACTION_CLASSES

GAP = FEATURE_NAMES.index("box_gap")


def load_windows(stem):
    arrays = np.load(Path(stem).with_suffix(".npz"))
    info = json.loads(Path(stem).with_suffix(".json").read_text(encoding="utf-8"))
    return {key: arrays[key] for key in arrays.files}, info


def subset(arrays, keep):
    return {key: value[keep] for key, value in arrays.items()}


def mean_gap(arrays):
    """Mean box gap over the frames where both boxes are visible."""
    both = (arrays["masks"][..., 0] & arrays["masks"][..., 1]).astype(bool)
    return np.where(both, arrays["features"][..., GAP], 0).sum(1) / np.maximum(both.sum(1), 1)


def gap_rule(train, evaluate):
    """Ownership baseline: owner if the mean gap is at most a threshold tuned on train."""
    known = train["ownership"] >= 0
    gaps, labels = mean_gap(train)[known], train["ownership"][known] == 1
    threshold = max(
        np.unique(gaps), key=lambda t: stlstm.binary_metrics(labels, (gaps <= t).astype(float))["f1"]
    )
    known = evaluate["ownership"] >= 0
    gaps = mean_gap(evaluate)[known]
    metrics = stlstm.binary_metrics(evaluate["ownership"][known] == 1, -gaps, threshold=-threshold)
    return {**metrics, "threshold": float(threshold)}


def predict(model, arrays, device, batch=256):
    model.eval()
    ownership, interaction = [np.zeros(0)], [np.zeros((0, len(INTERACTION_CLASSES)))]
    with torch.no_grad():
        for start in range(0, len(arrays["ownership"]), batch):
            features = torch.as_tensor(arrays["features"][start : start + batch], device=device)
            masks = torch.as_tensor(arrays["masks"][start : start + batch], device=device)
            own, inter = model(features, masks)
            ownership.append(torch.sigmoid(own).cpu().numpy())
            interaction.append(torch.softmax(inter, -1).cpu().numpy())
    return np.concatenate(ownership), np.concatenate(interaction)


def score(arrays, ownership, interaction, threshold=0.5):
    own_known = arrays["ownership"] >= 0
    inter_known = arrays["interaction"] >= 0
    return {
        "ownership": stlstm.binary_metrics(arrays["ownership"][own_known] == 1, ownership[own_known], threshold),
        "interaction": stlstm.multiclass_metrics(
            arrays["interaction"][inter_known], interaction[inter_known].argmax(1)
        ),
    }


def selection_score(metrics):
    """Mean of ownership AUROC and interaction macro-F1 on the selection clips."""
    return ((metrics["ownership"]["auroc"] or 0.0) + (metrics["interaction"]["macro_f1"] or 0.0)) / 2


def class_weights(labels, classes):
    """Square-root inverse frequency, so rare interactions count more but not wildly."""
    counts = np.bincount(labels[labels >= 0], minlength=classes)
    return np.sqrt(counts.sum() / (classes * np.maximum(counts, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True, help="windows path without extension")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--select-from", type=Path, help="export_yolo_dataset summary.json; its val clips select")
    parser.add_argument("--select-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--interaction-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (leave cores for other jobs)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    data, info = load_windows(args.train)
    test, _ = load_windows(args.test)
    clips = info["clips"]
    if args.select_from:
        select_clips = set(json.loads(args.select_from.read_text(encoding="utf-8"))["clips"]["val"])
    else:
        select_clips = set(clips[args.select_every - 1 :: args.select_every])
    is_select = np.isin(data["clip"], [i for i, clip in enumerate(clips) if clip in select_clips])
    train, select = subset(data, ~is_select), subset(data, is_select)
    print(
        f"windows: train {len(train['ownership'])}, select {len(select['ownership'])}"
        f" ({len(select_clips & set(clips))} clips), test {len(test['ownership'])}",
        flush=True,
    )

    model = stlstm.PairwiseSTLSTM(
        embed=args.embed, hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        bidirectional=not args.unidirectional,
    ).to(device)
    model.set_normalization(train["features"])
    owned = train["ownership"][train["ownership"] >= 0]
    pos_weight = torch.tensor((owned == 0).sum() / max((owned == 1).sum(), 1), device=device)
    own_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    inter_loss = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights(train["interaction"], len(INTERACTION_CLASSES)), dtype=torch.float32, device=device),
        ignore_index=-1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tensors = {key: torch.as_tensor(value, device=device) for key, value in train.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    run_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    history, best, best_epoch, waited = [], -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train["ownership"]), device=device)
        totals = [0.0, 0.0]
        for start in range(0, len(order), args.batch):
            index = order[start : start + args.batch]
            own_logit, inter_logit = model(tensors["features"][index], tensors["masks"][index])
            own_target = tensors["ownership"][index].float()
            known = (own_target >= 0).float()
            loss_own = (own_loss(own_logit, own_target.clamp_min(0)) * known).sum() / known.sum().clamp_min(1)
            loss_inter = inter_loss(inter_logit, tensors["interaction"][index].long())
            loss = loss_own + args.interaction_weight * loss_inter
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals[0] += loss_own.item() * len(index)
            totals[1] += loss_inter.item() * len(index)
        scheduler.step()
        metrics = score(select, *predict(model, select, device))
        value = selection_score(metrics)
        row = {
            "epoch": epoch,
            "loss_ownership": round(totals[0] / len(order), 4),
            "loss_interaction": round(totals[1] / len(order), 4),
            "select_ownership_auroc": metrics["ownership"]["auroc"],
            "select_ownership_f1": round(metrics["ownership"]["f1"], 4),
            "select_interaction_macro_f1": metrics["interaction"]["macro_f1"],
            "select_score": round(value, 4),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if value > best:
            best, best_epoch, waited = value, epoch, 0
            stlstm.save(model, args.output / "model.pt", epoch=epoch, args=run_args)
        else:
            waited += 1
        print(
            f"epoch {epoch:3d}  loss own {row['loss_ownership']:.3f} inter {row['loss_interaction']:.3f}"
            f"  | select own AUROC {row['select_ownership_auroc']:.3f} F1 {row['select_ownership_f1']:.3f}"
            f" inter macro-F1 {row['select_interaction_macro_f1']:.3f}{'  *best' if best_epoch == epoch else ''}",
            flush=True,
        )
        if waited >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs", flush=True)
            break

    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model, checkpoint = stlstm.load(args.output / "model.pt", device)
    select_own, select_inter = predict(model, select, device)
    known = select["ownership"] >= 0
    threshold = stlstm.best_threshold(select["ownership"][known] == 1, select_own[known])
    stlstm.save(model, args.output / "model.pt", epoch=best_epoch, args=run_args, threshold=threshold)
    own_scores, inter_probs = predict(model, test, device)
    majority = int(np.bincount(train["interaction"][train["interaction"] >= 0]).argmax())
    inter_known = test["interaction"] >= 0
    report = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.perf_counter() - started, 1),
        "ownership_threshold": threshold,
        "windows": {"train": len(train["ownership"]), "select": len(select["ownership"]), "test": len(test["ownership"])},
        "select": score(select, select_own, select_inter, threshold),
        "test": score(test, own_scores, inter_probs, threshold),
        "test_threshold_0.5": score(test, own_scores, inter_probs)["ownership"],
        "baselines_test": {
            "ownership_gap_rule": gap_rule(train, test),
            "interaction_majority": stlstm.multiclass_metrics(
                test["interaction"][inter_known], np.full(int(inter_known.sum()), majority)
            ),
        },
        "args": run_args,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "test_predictions.npz", ownership=own_scores, interaction=inter_probs)
    own, inter = report["test"]["ownership"], report["test"]["interaction"]
    rule = report["baselines_test"]["ownership_gap_rule"]
    print(json.dumps({
        "best_epoch": best_epoch,
        "ownership_threshold": round(threshold, 4),
        "test_ownership": {k: own[k] for k in ("auroc", "f1", "precision", "recall", "count")},
        "gap_rule_ownership": {k: rule[k] for k in ("auroc", "f1", "precision", "recall")},
        "test_interaction": {
            "macro_f1": inter["macro_f1"], "accuracy": inter["accuracy"],
            "per_class_f1": {k: round(v["f1"], 3) for k, v in inter["per_class"].items()},
        },
        "majority_interaction_macro_f1": report["baselines_test"]["interaction_majority"]["macro_f1"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
