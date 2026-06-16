import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asd_mamba.config import TrainConfig
from asd_mamba.data.dataset import load_domains, load_labels
from asd_mamba.train_dann import ensure_dann_files, train_one_fold
from asd_mamba.utils.metrics import format_summary_for_report, mean_std_summary
from asd_mamba.utils.seed import seed_everything


def resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    if output_dir.parts[:1] == ("outputs",):
        return ROOT / output_dir
    return ROOT / "outputs" / output_dir


def load_site_names(data_dir: Path) -> Dict[int, str]:
    mapping_path = data_dir / "site_mapping.json"
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        site_to_id = json.load(f)
    return {int(site_id): str(site_name) for site_name, site_id in site_to_id.items()}


def print_summary(summary: dict) -> None:
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("F1", "f1"),
        ("AUC", "auc"),
    ]:
        print("{}: {}".format(label, summary[key]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN-Transformer with Leave-One-Site-Out classification on ABIDE."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/abide_cc400_author_npz")
    parser.add_argument("--output-dir", type=Path, default=Path("cnn_transformer_loso"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--monitor", choices=["auc", "loss"], default="auc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-rois", type=int, default=392)
    parser.add_argument("--use-fc-branch", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--fc-branch-type",
        choices=["mlp", "wide_mlp", "gated_mlp", "transformer"],
        default="mlp",
    )
    parser.add_argument("--fc-dim", type=int, default=76636)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--use-spectral-branch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--spectral-bins", type=int, default=32)
    parser.add_argument("--use-dann", type=int, choices=[0, 1], default=1)
    parser.add_argument("--dann-lambda", type=float, default=0.1)
    parser.add_argument("--no-dann-warmup", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--cnn-layers", type=int, default=2)
    parser.add_argument("--cnn-kernel-size", type=int, default=5)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=resolve_output_dir(args.output_dir),
        seed=args.seed,
        folds=0,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        early_stopping_min_epochs=args.min_epochs,
        early_stopping_min_delta=args.min_delta,
        monitor_metric=args.monitor,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_gamma=args.lr_scheduler_gamma,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        device=args.device,
        model_type="cnn_transformer",
        num_rois=args.num_rois,
        d_model=args.d_model,
        dropout=args.dropout,
        fc_dim=args.fc_dim,
        use_fc_branch=bool(args.use_fc_branch),
        fc_branch_type=args.fc_branch_type,
        use_spectral_branch=bool(args.use_spectral_branch),
        spectral_bins=args.spectral_bins,
        use_dann=bool(args.use_dann),
        dann_feature_dim=args.feature_dim,
        dann_lambda=args.dann_lambda,
        dann_warmup=not args.no_dann_warmup,
        cnn_layers=args.cnn_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
    )


def run_leave_one_site_out(config: TrainConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dann_files(config)

    domain_path = Path(config.data_dir) / "domain.npy"
    if not domain_path.exists():
        raise FileNotFoundError("Leave-One-Site-Out requires domain.npy in {}".format(config.data_dir))
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by default, but torch.cuda.is_available() is False.")

    labels = load_labels(config.data_dir)
    domains = load_domains(config.data_dir)
    site_names = load_site_names(Path(config.data_dir))
    device = torch.device(config.device)
    loso_start_time = time.perf_counter()

    config_payload = {key: str(value) for key, value in asdict(config).items()}
    config_payload["split_protocol"] = "leave_one_site_out"
    with (config.output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    fold_rows: List[Dict[str, object]] = []
    for fold, held_out_site in enumerate(np.sort(np.unique(domains)), start=1):
        test_indices = np.flatnonzero(domains == held_out_site)
        train_indices = np.flatnonzero(domains != held_out_site)
        site_name = site_names.get(int(held_out_site), "site_{}".format(int(held_out_site)))
        print(
            "LOSO fold {} / {}: hold out {} (n={})".format(
                fold,
                len(np.unique(domains)),
                site_name,
                len(test_indices),
            )
        )

        metrics = train_one_fold(
            config=config,
            fold=fold,
            train_indices=train_indices,
            test_indices=test_indices,
            labels=labels,
            domains=domains,
            device=device,
        )
        metrics.update(
            {
                "held_out_site_id": int(held_out_site),
                "held_out_site": site_name,
                "test_asd": int((labels[test_indices] == 1).sum()),
                "test_control": int((labels[test_indices] == 0).sum()),
            }
        )
        fold_dir = config.output_dir / "fold_{}".format(fold)
        with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        fold_rows.append(metrics)

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
    total_train_time = float(sum(float(row["train_time_seconds"]) for row in fold_rows))
    total_wall_time = float(time.perf_counter() - loso_start_time)
    parameter_count = int(fold_rows[0]["parameters"]) if fold_rows else 0

    summary_raw["split_protocol"] = "leave_one_site_out"
    summary_raw["device"] = str(device)
    summary_raw["num_samples"] = int(len(labels))
    summary_raw["num_asd"] = int((labels == 1).sum())
    summary_raw["num_control"] = int((labels == 0).sum())
    summary_raw["num_sites"] = int(len(np.unique(domains)))
    summary_raw["parameter_count"] = parameter_count
    summary_raw["total_train_time_seconds"] = total_train_time
    summary_raw["total_train_time_minutes"] = total_train_time / 60.0
    summary_raw["total_wall_time_seconds"] = total_wall_time
    summary_raw["total_wall_time_minutes"] = total_wall_time / 60.0

    summary["split_protocol"] = "leave_one_site_out"
    summary["device"] = str(device)
    summary["num_samples"] = int(len(labels))
    summary["num_asd"] = int((labels == 1).sum())
    summary["num_control"] = int((labels == 0).sum())
    summary["num_sites"] = int(len(np.unique(domains)))
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


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_leave_one_site_out(config)
    print("Leave-One-Site-Out training completed.")
    print_summary(result["summary"])


if __name__ == "__main__":
    main()
