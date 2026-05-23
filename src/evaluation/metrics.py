# File: src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(all_preds, all_targets, threshold=0.5):
    
    y_pred = (all_preds >= threshold).astype(int)
    y_true = all_targets.astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    }


def print_metrics_table(metrics_dict, title="EVALUATION RESULT"):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    for k, v in metrics_dict.items():
        print(f"{k:<12}: {v:.4f}")

    print("=" * 40 + "\n")