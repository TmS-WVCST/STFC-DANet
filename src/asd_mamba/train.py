from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def class_weight_from_labels(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def split_train_val(
    train_indices: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple:
    if val_ratio <= 0.0:
        return train_indices, np.asarray([], dtype=np.int64)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )
    local_train, local_val = next(
        splitter.split(np.zeros_like(train_indices), labels[train_indices])
    )
    return train_indices[local_train], train_indices[local_val]


def autocast_context(device: torch.device, use_amp: bool):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def make_grad_scaler(device: torch.device, use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def should_update_best(
    current: Dict[str, float],
    best_score: float,
    monitor_metric: str,
    min_delta: float,
) -> bool:
    if monitor_metric == "loss":
        return current["loss"] < best_score - min_delta
    if monitor_metric == "auc":
        current_auc = current["auc"]
        if np.isnan(current_auc):
            return False
        return current_auc > best_score + min_delta
    raise ValueError("Unsupported monitor_metric: {}".format(monitor_metric))

