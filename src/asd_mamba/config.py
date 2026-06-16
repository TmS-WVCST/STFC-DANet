from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreprocessConfig:
    abide_root: Path = PROJECT_ROOT / "ABIDE/ABIDE_pcp"
    output_dir: Path = PROJECT_ROOT / "data/abide_cc400_author_npz"
    pipeline: str = "cpac"
    strategy: str = "filt_noglobal"
    atlas: str = "cc400"
    target_timepoints: int = 316
    min_timepoints: int = 316
    crop_mode: str = "first"


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = PROJECT_ROOT / "data/abide_cc400_author_npz"
    output_dir: Path = PROJECT_ROOT / "outputs/stfc_danet"
    seed: int = 42
    folds: int = 5
    val_ratio: float = 0.2
    epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_min_epochs: int = 50
    early_stopping_min_delta: float = 1e-4
    monitor_metric: str = "auc"
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_gamma: float = 0.75
    num_workers: int = 0
    amp: bool = True
    device: str = "cuda"
    model_type: str = "cnn_transformer"
    num_rois: int = 392
    d_model: int = 128
    dropout: float = 0.2
    cnn_layers: int = 2
    cnn_kernel_size: int = 5
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 256
    fc_dim: int = 76636
    use_fc_branch: bool = True
    fc_branch_type: str = "mlp"
    use_spectral_branch: bool = False
    spectral_bins: int = 32
    use_dann: bool = True
    dann_feature_dim: int = 128
    dann_lambda: float = 0.1
    dann_warmup: bool = True

