from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob_asd: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob_asd = np.asarray(y_prob_asd, dtype=float)
    y_pred = (y_prob_asd >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob_asd)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
    }


def mean_std_summary(
    rows: List[Dict[str, float]],
    keys: List[str],
) -> Dict[str, Union[float, str]]:
    summary: Dict[str, Union[float, str]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
    return summary


def format_summary_for_report(
    summary: Dict[str, Union[float, str]],
    keys: List[str],
) -> Dict[str, str]:
    percent_keys = {
        "accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "domain_accuracy",
        "best_val_domain_accuracy",
    }
    auc_keys = {"auc", "best_val_auc"}
    report: Dict[str, str] = {}
    for key in keys:
        mean = float(summary[f"{key}_mean"])
        std = float(summary[f"{key}_std"])
        if key in auc_keys:
            report[key] = f"{mean:.4f} ± {std:.4f}"
        elif key in percent_keys:
            report[key] = f"{mean * 100.0:.2f} ± {std * 100.0:.2f}"
        else:
            report[key] = f"{mean:.2f} ± {std:.2f}"
    return report

