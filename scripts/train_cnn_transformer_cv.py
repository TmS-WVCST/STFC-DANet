import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asd_mamba.config import TrainConfig
from asd_mamba.train_dann import run_dann_cross_validation


def resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    if output_dir.parts[:1] == ("outputs",):
        return ROOT / output_dir
    return ROOT / "outputs" / output_dir


def print_summary(summary: dict) -> None:
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("F1", "f1"),
        ("AUC", "auc"),
    ]:
        print(f"{label}: {summary[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-Transformer with 5-fold CV on ABIDE.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/abide_cc400_author_npz")
    parser.add_argument("--output-dir", type=Path, default=Path("cnn_transformer_author_npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--monitor", choices=["auc", "loss"], default="auc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--no-dann-warmup", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--cnn-layers", type=int, default=2)
    parser.add_argument("--cnn-kernel-size", type=int, default=5)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=resolve_output_dir(args.output_dir),
        seed=args.seed,
        folds=args.folds,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        early_stopping_min_epochs=args.min_epochs,
        early_stopping_min_delta=args.min_delta,
        monitor_metric=args.monitor,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
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
        dann_warmup=not args.no_dann_warmup,
        cnn_layers=args.cnn_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
    )
    result = run_dann_cross_validation(config)
    print("Training completed.")
    print_summary(result["summary"])


if __name__ == "__main__":
    main()

