import numpy as np

def pure_roc_auc(y_true, y_score):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def pure_pr_auc(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recalls = tp / max(1, np.sum(y_true))
    precisions = tp / (tp + fp)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

y_true = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1])
y_score = np.array([0.1, 0.2, 0.15, 0.8, 0.3, 0.7, 0.9, 0.05, 0.4, 0.65])
print("ROC-AUC:", pure_roc_auc(y_true, y_score))
print("PR-AUC:", pure_pr_auc(y_true, y_score))
