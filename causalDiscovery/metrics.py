"""Graph metrics with explicit diagonal and threshold conventions."""
import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


def graph_metrics(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores)
    if truth.shape != scores.shape or truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError('Ground truth and scores must be matching square matrices')
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError('Expected finite scores and binary ground truth')
    result = {}
    for name, keep in [('offdiag', ~np.eye(len(truth), dtype=bool)),
                       ('full', np.ones_like(truth, dtype=bool))]:
        y, s = truth[keep].astype(int), scores[keep]
        order = np.argsort(-s, kind='stable')
        pred = np.zeros_like(y)
        pred[order[:int(y.sum())]] = 1
        fixed = (s >= .5).astype(int)
        valid = len(np.unique(y)) == 2
        precision, recall, _ = precision_recall_curve(y, s) if valid else (None, None, None)
        result[name] = dict(
            auroc=float(roc_auc_score(y, s)) if valid else None,
            auprc=float(auc(recall, precision)) if valid else None,
            shd_density_oracle=int(np.count_nonzero(y != pred)),
            f1_density_oracle=float(f1_score(y, pred, zero_division=0)) if y.sum() else None,
            shd_0_5=int(np.count_nonzero(y != fixed)),
            f1_0_5=float(f1_score(y, fixed, zero_division=0)) if y.sum() else None,
            true_edges=int(y.sum()), possible_edges=len(y))
    return result


def typed_metrics(mean_truth, variance_truth, mean_scores, variance_scores):
    off = ~np.eye(len(mean_truth), dtype=bool)
    known = off & ((mean_truth != 0) | (variance_truth != 0))
    # Some future benchmarks may contain both edge types; do not force a label there.
    exclusive = known & ~((mean_truth != 0) & (variance_truth != 0))
    truth = variance_truth[exclusive] != 0
    pred = variance_scores[exclusive] > mean_scores[exclusive]
    return dict(mean=graph_metrics(mean_truth != 0, mean_scores),
                variance=graph_metrics(variance_truth != 0, variance_scores),
                type_correct_on_known_exclusive_edges=int((truth == pred).sum()),
                type_total_known_exclusive_edges=len(truth),
                type_accuracy_on_known_exclusive_edges=float((truth == pred).mean()) if len(truth) else None)
