import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asd_mamba.config import TrainConfig
from asd_mamba.data.dataset import AbideDannDataset, load_domains, load_fc, load_labels
from asd_mamba.models.cnn_transformer import CnnTransformerClassifier
from asd_mamba.train import (
    autocast_context,
    class_weight_from_labels,
    make_grad_scaler,
    should_update_best,
    split_train_val,
)
from asd_mamba.utils.metrics import classification_metrics, format_summary_for_report, mean_std_summary
from asd_mamba.utils.seed import seed_everything


MetricValue = Union[float, int]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def requires_fc(config: TrainConfig) -> bool:
    return config.model_type != "cnn_transformer" or config.use_fc_branch


def requires_domain(config: TrainConfig) -> bool:
    return config.model_type != "cnn_transformer" or config.use_dann


def ensure_dann_files(config: TrainConfig) -> None:
    required = ["X.npy", "y.npy"]
    if requires_fc(config):
        required.append("fc.npy")
    if requires_domain(config):
        required.append("domain.npy")
    data_dir = Path(config.data_dir)
    missing = [name for name in required if not (Path(data_dir) / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing DANN cached files: {}. Please rerun: python scripts/preprocess_author_npz.py".format(
                ", ".join(missing)
            )
        )


def compute_fc_standardizer(data_dir: Path, train_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fc = load_fc(data_dir)
    train_fc = np.asarray(fc[train_indices], dtype=np.float32)
    mean = train_fc.mean(axis=0).astype(np.float32)
    std = train_fc.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def build_model(config: TrainConfig, num_domains: int) -> nn.Module:
    if config.model_type != "cnn_transformer":
        raise ValueError("Unsupported model_type: {}".format(config.model_type))
    return CnnTransformerClassifier(
        num_rois=config.num_rois,
        num_classes=2,
        d_model=config.d_model,
        cnn_layers=config.cnn_layers,
        cnn_kernel_size=config.cnn_kernel_size,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        transformer_ff_dim=config.transformer_ff_dim,
        fc_dim=config.fc_dim,
        feature_dim=config.dann_feature_dim,
        use_fc_branch=config.use_fc_branch,
        fc_branch_type=config.fc_branch_type,
        use_spectral_branch=config.use_spectral_branch,
        spectral_bins=config.spectral_bins,
        use_dann=config.use_dann,
        num_domains=num_domains,
        dropout=config.dropout,
    )


def domain_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().detach().cpu())


def dann_lambda_for_epoch(config: TrainConfig, epoch: int) -> float:
    if not config.use_dann:
        return 0.0
    if not config.dann_warmup:
        return config.dann_lambda
    progress = (epoch - 1) / max(config.epochs - 1, 1)
    return float(config.dann_lambda * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    label_criterion: nn.Module,
    domain_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    lambda_value: float,
) -> Dict[str, float]:
    model.train()
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_domain_acc = 0.0
    total_samples = 0

    for x_time, x_fc, y, domain in loader:
        x_time = x_time.to(device, non_blocking=True)
        x_fc = x_fc.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        domain = domain.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            outputs = model(x_time, x_fc, lambda_value=lambda_value)
            if isinstance(outputs, tuple):
                label_logits, domain_logits, _ = outputs
            else:
                label_logits, domain_logits = outputs, None
            label_loss = label_criterion(label_logits, y)
            if domain_logits is not None:
                domain_loss = domain_criterion(domain_logits, domain)
                loss = label_loss + domain_loss
            else:
                domain_loss = label_loss.new_tensor(0.0)
                loss = label_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x_time.shape[0]
        total_label_loss += float(label_loss.detach().cpu()) * batch_size
        total_domain_loss += float(domain_loss.detach().cpu()) * batch_size
        if domain_logits is not None:
            total_domain_acc += domain_accuracy(domain_logits, domain) * batch_size
        total_samples += batch_size

    total_samples = max(total_samples, 1)
    return {
        "label_loss": total_label_loss / total_samples,
        "domain_loss": total_domain_loss / total_samples,
        "domain_accuracy": total_domain_acc / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_criterion: nn.Module,
    domain_criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_prob_asd = []
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_domain_acc = 0.0
    total_samples = 0

    for x_time, x_fc, y, domain in loader:
        x_time = x_time.to(device, non_blocking=True)
        x_fc = x_fc.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        domain_device = domain.to(device, non_blocking=True)
        outputs = model(x_time, x_fc, lambda_value=0.0)
        if isinstance(outputs, tuple):
            label_logits, domain_logits, _ = outputs
        else:
            label_logits, domain_logits = outputs, None
        label_loss = label_criterion(label_logits, y_device)
        if domain_logits is not None:
            domain_loss = domain_criterion(domain_logits, domain_device)
        else:
            domain_loss = label_loss.new_tensor(0.0)
        prob = torch.softmax(label_logits, dim=1)[:, 1]

        batch_size = x_time.shape[0]
        total_label_loss += float(label_loss.detach().cpu()) * batch_size
        total_domain_loss += float(domain_loss.detach().cpu()) * batch_size
        if domain_logits is not None:
            total_domain_acc += domain_accuracy(domain_logits, domain_device) * batch_size
        total_samples += batch_size
        y_true.append(y.numpy())
        y_prob_asd.append(prob.detach().cpu().numpy())

    metrics = classification_metrics(
        y_true=np.concatenate(y_true),
        y_prob_asd=np.concatenate(y_prob_asd),
    )
    total_samples = max(total_samples, 1)
    metrics["loss"] = total_label_loss / total_samples
    metrics["domain_loss"] = total_domain_loss / total_samples
    metrics["domain_accuracy"] = total_domain_acc / total_samples
    return metrics


def train_one_fold(
    config: TrainConfig,
    fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    device: torch.device,
) -> Dict[str, MetricValue]:
    fold_dir = config.output_dir / "fold_{}".format(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_indices, val_indices = split_train_val(
        train_indices=train_indices,
        labels=labels,
        val_ratio=config.val_ratio,
        seed=config.seed + fold,
    )
    require_fc = requires_fc(config)
    require_domain = requires_domain(config)
    if require_fc:
        fc_mean, fc_std = compute_fc_standardizer(config.data_dir, train_indices)
    else:
        fc_mean, fc_std = None, None

    train_dataset = AbideDannDataset(
        config.data_dir,
        train_indices,
        fc_mean,
        fc_std,
        require_fc=require_fc,
        require_domain=require_domain,
    )
    val_dataset = AbideDannDataset(
        config.data_dir,
        val_indices,
        fc_mean,
        fc_std,
        require_fc=require_fc,
        require_domain=require_domain,
    )
    test_dataset = AbideDannDataset(
        config.data_dir,
        test_indices,
        fc_mean,
        fc_std,
        require_fc=require_fc,
        require_domain=require_domain,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    num_domains = int(domains.max()) + 1
    model = build_model(config, num_domains=num_domains).to(device)
    parameter_count = count_trainable_parameters(model)
    label_criterion = nn.CrossEntropyLoss(
        weight=class_weight_from_labels(labels[train_indices], device=device)
    )
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.lr_scheduler_gamma,
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = make_grad_scaler(device, use_amp)

    start_time = time.perf_counter()
    history = []
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_score = float("inf") if config.monitor_metric == "loss" else -float("inf")
    patience_counter = 0
    best_model_path = fold_dir / "best_model.pt"

    progress = tqdm(range(1, config.epochs + 1), desc="DANN Fold {}".format(fold), leave=False)
    for epoch in progress:
        lambda_value = dann_lambda_for_epoch(config, epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            label_criterion=label_criterion,
            domain_criterion=domain_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            lambda_value=lambda_value,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_loader, label_criterion, domain_criterion, device)

        history_row = {"epoch": epoch, "dann_lambda": lambda_value}
        history_row.update({"train_{}".format(key): value for key, value in train_metrics.items()})
        history_row.update({"val_{}".format(key): value for key, value in val_metrics.items()})
        history.append(history_row)

        if should_update_best(
            current=val_metrics,
            best_score=best_score,
            monitor_metric=config.monitor_metric,
            min_delta=config.early_stopping_min_delta,
        ):
            best_score = val_metrics["loss"] if config.monitor_metric == "loss" else val_metrics["auc"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        progress.set_postfix(
            label_loss="{:.4f}".format(train_metrics["label_loss"]),
            val_auc="{:.4f}".format(val_metrics["auc"]),
            domain_acc="{:.4f}".format(train_metrics["domain_accuracy"]),
            grl_lambda="{:.4f}".format(lambda_value),
            no_improve="{}/{}".format(patience_counter, config.early_stopping_patience),
            min_epoch=config.early_stopping_min_epochs,
        )
        if (
            epoch >= config.early_stopping_min_epochs
            and patience_counter >= config.early_stopping_patience
        ):
            break

    train_time = time.perf_counter() - start_time
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    metrics = evaluate(model, test_loader, label_criterion, domain_criterion, device)
    metrics.update(
        {
            "fold": fold,
            "train_size": int(len(train_indices)),
            "val_size": int(len(val_indices)),
            "test_size": int(len(test_indices)),
            "parameters": int(parameter_count),
            "train_time_seconds": float(train_time),
            "best_epoch": int(best_epoch),
            "epochs_trained": int(len(history)),
            "best_val_auc": float(best_metrics.get("auc", float("nan"))),
            "best_val_loss": float(best_metrics.get("loss", float("nan"))),
            "best_val_domain_accuracy": float(best_metrics.get("domain_accuracy", float("nan"))),
        }
    )

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    if fc_mean is not None and fc_std is not None:
        np.save(fold_dir / "fc_mean.npy", fc_mean)
        np.save(fold_dir / "fc_std.npy", fc_std)
    return metrics


def run_dann_cross_validation(config: TrainConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dann_files(config)
    cv_start_time = time.perf_counter()

    labels = load_labels(config.data_dir)
    domains = load_domains(config.data_dir) if requires_domain(config) else np.zeros_like(labels)
    splitter = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by default, but torch.cuda.is_available() is False.")
    device = torch.device(config.device)

    with (config.output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump({key: str(value) for key, value in asdict(config).items()}, f, indent=2)

    fold_rows: List[Dict[str, MetricValue]] = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros_like(labels), labels),
        start=1,
    ):
        fold_rows.append(
            train_one_fold(
                config=config,
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
                labels=labels,
                domains=domains,
                device=device,
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(config.output_dir / "fold_metrics.csv", index=False)

    metric_keys = [
        "accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "loss",
        "domain_accuracy",
        "train_time_seconds",
        "parameters",
        "best_epoch",
        "epochs_trained",
        "best_val_auc",
        "best_val_loss",
        "best_val_domain_accuracy",
    ]
    summary_raw = mean_std_summary(fold_rows, metric_keys)
    summary = format_summary_for_report(summary_raw, metric_keys)
    total_train_time = float(sum(row["train_time_seconds"] for row in fold_rows))
    total_wall_time = float(time.perf_counter() - cv_start_time)
    parameter_count = int(fold_rows[0]["parameters"]) if fold_rows else 0
    summary_raw["device"] = str(device)
    summary_raw["num_samples"] = int(len(labels))
    summary_raw["num_asd"] = int((labels == 1).sum())
    summary_raw["num_control"] = int((labels == 0).sum())
    summary_raw["num_domains"] = int(domains.max()) + 1
    summary_raw["parameter_count"] = parameter_count
    summary_raw["total_train_time_seconds"] = total_train_time
    summary_raw["total_train_time_minutes"] = total_train_time / 60.0
    summary_raw["total_wall_time_seconds"] = total_wall_time
    summary_raw["total_wall_time_minutes"] = total_wall_time / 60.0
    summary["device"] = str(device)
    summary["num_samples"] = int(len(labels))
    summary["num_asd"] = int((labels == 1).sum())
    summary["num_control"] = int((labels == 0).sum())
    summary["num_domains"] = int(domains.max()) + 1
    summary["parameter_count"] = parameter_count
    summary["total_train_time_seconds"] = total_train_time
    summary["total_train_time_minutes"] = total_train_time / 60.0
    summary["total_wall_time_seconds"] = total_wall_time
    summary["total_wall_time_minutes"] = total_wall_time / 60.0

    pd.DataFrame([summary_raw]).to_csv(config.output_dir / "summary_metrics_raw.csv", index=False)
    with (config.output_dir / "summary_metrics_raw.json").open("w", encoding="utf-8") as f:
        json.dump(summary_raw, f, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(config.output_dir / "summary_metrics.csv", index=False)
    with (config.output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"fold_metrics": fold_rows, "summary": summary, "summary_raw": summary_raw}

