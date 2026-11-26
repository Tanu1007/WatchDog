"""Pairwise ST-LSTM: a bidirectional LSTM over one person-item window.

Input per frame: the 12 pairwise features (standardized with training-set
statistics stored in the model) concatenated with the 3 mask flags. The final
forward and backward hidden states feed two heads:

- ownership: one logit, "this person owns this item";
- interaction: logits over pairwise.INTERACTION_CLASSES at the window's end.

At inference both directions run over the already-observed window ending at
the score time, as the project context specifies. Requires PyTorch and
scikit-learn (the `stlstm` extra).
"""

from itertools import groupby

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

from .pairwise import FEATURE_NAMES, INTERACTION_CLASSES, MASK_NAMES, build_windows


class PairwiseSTLSTM(nn.Module):
    def __init__(
        self,
        features=len(FEATURE_NAMES),
        masks=len(MASK_NAMES),
        classes=len(INTERACTION_CLASSES),
        embed=64,
        hidden=64,
        layers=1,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()
        self.config = {
            "features": features, "masks": masks, "classes": classes, "embed": embed,
            "hidden": hidden, "layers": layers, "dropout": dropout, "bidirectional": bidirectional,
        }
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))
        self.embed = nn.Sequential(nn.Linear(features + masks, embed), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            embed, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        summary = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.ownership = nn.Linear(summary, 1)
        self.interaction = nn.Linear(summary, classes)

    def set_normalization(self, features):
        """Standardize with per-feature mean/std of a (N, T, F) training array."""
        flat = torch.as_tensor(features, dtype=torch.float32).reshape(-1, features.shape[-1])
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-3))

    def forward(self, features, masks):
        x = torch.cat([(features - self.mean) / self.std, masks.float()], dim=-1)
        _, (hidden, _) = self.lstm(self.embed(x))
        if self.lstm.bidirectional:
            summary = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            summary = hidden[-1]
        summary = self.dropout(summary)
        return self.ownership(summary).squeeze(-1), self.interaction(summary)


def predict(model, features, masks, batch=256):
    """Ownership probabilities (N,) and interaction probabilities (N, classes)."""
    model.eval()
    device = model.mean.device
    # New features are appended to the schema, so older models read a prefix.
    features = features[..., : model.config["features"]]
    ownership = [np.zeros(0, dtype=np.float32)]
    interaction = [np.zeros((0, model.config["classes"]), dtype=np.float32)]
    with torch.no_grad():
        for start in range(0, len(features), batch):
            own, inter = model(
                torch.as_tensor(features[start : start + batch], device=device),
                torch.as_tensor(masks[start : start + batch], device=device),
            )
            ownership.append(torch.sigmoid(own).cpu().numpy())
            interaction.append(torch.softmax(inter, -1).cpu().numpy())
    return np.concatenate(ownership), np.concatenate(interaction)


def pair_scores(model, frames, hop=15):
    """[(end_timestamp, person, item, ownership probability)] for every candidate
    pair, on windows ending every `hop` frames, in time order."""
    features, masks, _, _, meta = build_windows(frames, {}, [], hop=hop)
    ownership, _ = predict(model, features, masks)
    return sorted(
        (m["end_timestamp"], m["person"], m["item"], float(score)) for m, score in zip(meta, ownership)
    )


def bind_owners(scores, threshold):
    """Causal per-pair binding from pair_scores output.

    Each pair keeps its best score so far, so evidence from earlier windows (for
    example carrying) persists after the pair stops moving. An unbound item is
    bound, at the first scoring time when any of its pairs' best score reaches
    the threshold, to the person with the highest best score; like the rules
    engine, a binding is then kept. Returns {(item, owner): seconds bound}.
    """
    best, bound, last_seen = {}, {}, {}
    # All scores at one time are updated before any binding at that time.
    for timestamp, group in groupby(sorted(scores), key=lambda score: score[0]):
        items = set()
        for _, person, item, score in group:
            best[(person, item)] = max(best.get((person, item), 0.0), score)
            last_seen[item] = timestamp
            items.add(item)
        for item in items - bound.keys():
            eligible = [(s, p) for (p, i), s in best.items() if i == item and s >= threshold]
            if eligible:
                bound[item] = (max(eligible)[1], timestamp)
    return {(item, owner): last_seen[item] - since for item, (owner, since) in bound.items()}


def save(model, path, **extra):
    torch.save({"config": model.config, "state": model.state_dict(), **extra}, path)


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PairwiseSTLSTM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


def binary_metrics(labels, scores, threshold=0.5):
    """Ownership metrics; AUROC is None when only one class is present."""
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else None,
        "accuracy": float(accuracy_score(labels, predicted)) if len(labels) else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "count": int(len(labels)),
        "positives": int(labels.sum()),
    }


def best_threshold(labels, scores):
    """Score threshold with the highest F1 (ties: the higher threshold); 0.5 if undefined."""
    labels = np.asarray(labels, dtype=bool)
    if not 0 < labels.sum() < len(labels):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)[:-1]
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def multiclass_metrics(labels, predicted, classes=INTERACTION_CLASSES):
    """Interaction metrics; macro-F1 averages only classes present in labels."""
    indices = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=indices, zero_division=0
    )
    present = [i for i in indices if support[i] > 0]
    return {
        "accuracy": float(accuracy_score(labels, predicted)) if len(labels) else None,
        "macro_f1": float(np.mean(f1[present])) if present else None,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(classes)
        },
        "confusion": confusion_matrix(labels, predicted, labels=indices).tolist(),
    }
