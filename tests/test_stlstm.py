import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from watchdog_cctv import stlstm
from watchdog_cctv.pairwise import FEATURE_NAMES, INTERACTION_CLASSES, MASK_NAMES

GAP = FEATURE_NAMES.index("box_gap")


def toy_windows(count=64, length=20, seed=0):
    """Owners stay close (gap ~0.1), others far (gap ~2); everything else is noise."""
    generator = np.random.default_rng(seed)
    features = generator.normal(0, 1, (count, length, len(FEATURE_NAMES))).astype(np.float32)
    ownership = np.arange(count) % 2
    features[..., GAP] = np.where(ownership[:, None] == 1, 0.1, 2.0) + generator.normal(0, 0.05, (count, length))
    masks = np.ones((count, length, len(MASK_NAMES)), dtype=np.uint8)
    masks[..., 2] = 0
    return features, masks, ownership


class ModelTests(unittest.TestCase):
    def test_shapes_and_normalization(self):
        features, masks, _ = toy_windows()
        model = stlstm.PairwiseSTLSTM(embed=16, hidden=8)
        model.set_normalization(features)
        self.assertAlmostEqual(float(model.mean[GAP]), float(features[..., GAP].mean()), places=4)
        own, inter = model(torch.as_tensor(features), torch.as_tensor(masks))
        self.assertEqual(tuple(own.shape), (64,))
        self.assertEqual(tuple(inter.shape), (64, len(INTERACTION_CLASSES)))

    def test_learns_a_separable_ownership_signal(self):
        torch.manual_seed(0)
        features, masks, ownership = toy_windows()
        model = stlstm.PairwiseSTLSTM(embed=16, hidden=8, dropout=0.0)
        model.set_normalization(features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        x, m, y = torch.as_tensor(features), torch.as_tensor(masks), torch.as_tensor(ownership).float()
        for _ in range(60):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, m)[0], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = torch.sigmoid(model(x, m)[0]).detach().numpy()
        self.assertEqual(stlstm.binary_metrics(ownership, scores)["auroc"], 1.0)

    def test_save_and_load_round_trip(self):
        features, masks, _ = toy_windows(count=4)
        model = stlstm.PairwiseSTLSTM(embed=16, hidden=8, bidirectional=False)
        model.set_normalization(features)
        model.eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            stlstm.save(model, path, epoch=3)
            loaded, checkpoint = stlstm.load(path)
        self.assertEqual(checkpoint["epoch"], 3)
        x, m = torch.as_tensor(features), torch.as_tensor(masks)
        torch.testing.assert_close(loaded(x, m)[0], model(x, m)[0])


class BindingTests(unittest.TestCase):
    def test_best_score_so_far_binds_and_keeps_owner(self):
        scores = [
            (1.0, 5, 9, 0.2),  # (time, person, item, score)
            (1.0, 6, 9, 0.3),
            (2.0, 5, 9, 0.9),  # person 5 crosses the threshold first
            (2.0, 6, 9, 0.4),
            (3.0, 5, 9, 0.1),  # later low scores do not unbind
            (3.0, 6, 9, 0.95),
            (2.0, 7, 4, 0.6),  # second item, never above 0.7
        ]
        self.assertEqual(stlstm.bind_owners(sorted(scores), 0.7), {(9, 5): 1.0})

    def test_scheduled_binder_answers_from_bind_time(self):
        events, _ = stlstm.binding_events([(1.0, 5, 9, 0.2), (2.0, 5, 9, 0.9)], 0.7)
        binder = stlstm.ScheduledBinder(events)
        self.assertIsNone(binder(9, 1.9))
        self.assertEqual(binder(9, 2.0), 5)
        self.assertIsNone(binder(4, 5.0))

    def test_highest_eligible_person_wins_a_tie_in_time(self):
        scores = [(1.0, 5, 9, 0.8), (1.0, 6, 9, 0.9)]
        self.assertEqual(stlstm.bind_owners(scores, 0.7), {(9, 6): 0.0})


class MetricTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = stlstm.binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
        self.assertEqual(metrics["auroc"], 0.75)
        self.assertEqual((metrics["precision"], metrics["recall"]), (0.5, 0.5))
        self.assertIsNone(stlstm.binary_metrics([1, 1], [0.2, 0.8])["auroc"])

    def test_best_threshold_maximizes_f1(self):
        # Scores 0.9/0.7 are owners, 0.6/0.2 are not: any cut in (0.6, 0.7] is perfect.
        threshold = stlstm.best_threshold([1, 1, 0, 0], [0.9, 0.7, 0.6, 0.2])
        self.assertEqual(threshold, 0.7)
        self.assertEqual(stlstm.binary_metrics([1, 1, 0, 0], [0.9, 0.7, 0.6, 0.2], threshold)["f1"], 1.0)
        self.assertEqual(stlstm.best_threshold([1, 1], [0.3, 0.4]), 0.5)

    def test_macro_f1_skips_absent_classes(self):
        metrics = stlstm.multiclass_metrics([0, 0, 1, 1], [0, 0, 1, 0])
        # Classes 2 and 3 have no support, so macro-F1 averages classes 0 and 1 only.
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertEqual(metrics["confusion"][1], [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
