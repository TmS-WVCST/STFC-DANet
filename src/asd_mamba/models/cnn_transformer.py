import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_value * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: torch.Tensor, lambda_value: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_value)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class MultiScaleTemporalStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        branch_channels = [out_channels // 3, out_channels // 3]
        branch_channels.append(out_channels - sum(branch_channels))
        kernels = [3, 5, 7]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for channels, kernel in zip(branch_channels, kernels)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fuse(x)


class ResidualTemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class FcTransformerBranch(nn.Module):
    """Encode FC as ROI-wise connectivity tokens instead of a flat vector."""

    def __init__(
        self,
        num_rois: int,
        fc_dim: int,
        d_model: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ff_dim: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        expected_fc_dim = num_rois * (num_rois - 1) // 2
        if fc_dim != expected_fc_dim:
            raise ValueError(
                "FC-Transformer expects fc_dim={} for num_rois={}, got {}.".format(
                    expected_fc_dim,
                    num_rois,
                    fc_dim,
                )
            )
        row_indices, col_indices = torch.triu_indices(num_rois, num_rois, offset=1)
        self.num_rois = num_rois
        self.register_buffer("row_indices", row_indices, persistent=False)
        self.register_buffer("col_indices", col_indices, persistent=False)
        self.token_projection = nn.Sequential(
            nn.Linear(num_rois, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.roi_embedding = nn.Parameter(torch.zeros(1, num_rois, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
            )
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        nn.init.normal_(self.roi_embedding, std=0.02)

    def forward(self, x_fc: torch.Tensor) -> torch.Tensor:
        fc_matrix = x_fc.new_zeros(x_fc.shape[0], self.num_rois, self.num_rois)
        fc_matrix[:, self.row_indices, self.col_indices] = x_fc
        fc_matrix[:, self.col_indices, self.row_indices] = x_fc
        tokens = self.token_projection(fc_matrix) + self.roi_embedding
        tokens = self.norm(self.transformer(tokens))
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return self.projector(pooled)


class GatedFcMlpBranch(nn.Module):
    """Hierarchical FC encoder with latent gating for connectomic features."""

    def __init__(
        self,
        fc_dim: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(fc_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.GELU(),
        )

    def forward(self, x_fc: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x_fc)
        hidden = hidden * self.gate(hidden)
        return self.output_projection(hidden)


class WideFcMlpBranch(nn.Module):
    """Hierarchical FC encoder with a wider first projection for high-dimensional FC."""

    def __init__(
        self,
        fc_dim: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fc_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x_fc: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_fc)


class SpectralTemporalBranch(nn.Module):
    """Encode low-frequency BOLD spectral magnitude as complementary dynamics."""

    def __init__(
        self,
        spectral_bins: int,
        d_model: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if spectral_bins <= 0:
            raise ValueError("spectral_bins must be positive.")
        self.spectral_bins = spectral_bins
        self.token_projection = nn.Sequential(
            nn.Linear(spectral_bins, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x_time, dim=1).abs()
        spectrum = spectrum[:, 1 : self.spectral_bins + 1, :]
        if spectrum.shape[1] < self.spectral_bins:
            pad_size = self.spectral_bins - spectrum.shape[1]
            spectrum = F.pad(spectrum, (0, 0, 0, pad_size))
        spectrum = torch.log1p(spectrum).transpose(1, 2)
        tokens = self.token_projection(spectrum)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
        return self.projector(pooled)


class CnnTransformerClassifier(nn.Module):
    """Temporal CNN before Transformer for ABIDE ROI time series.

    Input shape is [batch, time, roi]. The CNN extracts local temporal patterns
    from the 200 ROI channels, then Transformer encodes global temporal context.
    """

    def __init__(
        self,
        num_rois: int = 200,
        num_classes: int = 2,
        d_model: int = 128,
        cnn_layers: int = 2,
        cnn_kernel_size: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        fc_dim: int = 76636,
        feature_dim: int = 128,
        use_fc_branch: bool = False,
        fc_branch_type: str = "mlp",
        use_spectral_branch: bool = False,
        spectral_bins: int = 32,
        use_dann: bool = False,
        num_domains: int = 20,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads.")
        if fc_branch_type not in {"mlp", "wide_mlp", "gated_mlp", "transformer"}:
            raise ValueError(
                "fc_branch_type must be 'mlp', 'wide_mlp', 'gated_mlp', or 'transformer'."
            )

        self.use_fc_branch = use_fc_branch
        self.use_spectral_branch = use_spectral_branch
        self.use_dann = use_dann
        self.temporal_stem = MultiScaleTemporalStem(
            in_channels=num_rois,
            out_channels=d_model,
            dropout=dropout,
        )
        self.temporal_cnn = nn.Sequential(
            *[
                ResidualTemporalConvBlock(
                    channels=d_model,
                    kernel_size=cnn_kernel_size,
                    dropout=dropout,
                )
                for _ in range(cnn_layers)
            ]
        )
        self.cnn_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = SinusoidalPositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
            )
        self.norm = nn.LayerNorm(d_model)
        self.time_projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.fc_branch = (
            self.build_fc_branch(
                num_rois=num_rois,
                fc_dim=fc_dim,
                d_model=d_model,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_ff_dim=transformer_ff_dim,
                feature_dim=feature_dim,
                branch_type=fc_branch_type,
                dropout=dropout,
            )
            if use_fc_branch
            else None
        )
        self.spectral_branch = (
            SpectralTemporalBranch(
                spectral_bins=spectral_bins,
                d_model=d_model,
                feature_dim=feature_dim,
                dropout=dropout,
            )
            if use_spectral_branch
            else None
        )
        fusion_branches = 1 + int(use_fc_branch) + int(use_spectral_branch)
        self.fusion = (
            nn.Sequential(
                nn.Linear(feature_dim * fusion_branches, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if fusion_branches > 1
            else nn.Identity()
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )
        self.domain_classifier = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_domains),
            )
            if use_dann
            else None
        )
        self.grl = GradientReversal() if use_dann else None
        nn.init.normal_(self.cls_token, std=0.02)

    def build_fc_branch(
        self,
        num_rois: int,
        fc_dim: int,
        d_model: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ff_dim: int,
        feature_dim: int,
        branch_type: str,
        dropout: float,
    ) -> nn.Module:
        if branch_type == "mlp":
            return nn.Sequential(
                nn.Linear(fc_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, feature_dim),
                nn.GELU(),
            )
        if branch_type == "gated_mlp":
            return GatedFcMlpBranch(
                fc_dim=fc_dim,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        if branch_type == "wide_mlp":
            return WideFcMlpBranch(
                fc_dim=fc_dim,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        if branch_type == "transformer":
            return FcTransformerBranch(
                num_rois=num_rois,
                fc_dim=fc_dim,
                d_model=d_model,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_ff_dim=transformer_ff_dim,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        raise ValueError("Unsupported fc_branch_type: {}".format(branch_type))

    def encode_time(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.temporal_stem(x)
        x = self.temporal_cnn(x)
        x = x.transpose(1, 2)
        x = self.cnn_norm(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_pool = x[:, 0]
        sequence = x[:, 1:]
        mean_pool = sequence.mean(dim=1)
        max_pool = sequence.max(dim=1).values
        pooled = torch.cat([cls_pool, mean_pool, max_pool], dim=-1)
        return self.time_projector(pooled)

    def extract_features(
        self,
        x_time: torch.Tensor,
        x_fc: torch.Tensor = None,
    ) -> torch.Tensor:
        features = [self.encode_time(x_time)]
        if self.use_fc_branch and x_fc is None:
            raise ValueError("x_fc is required when use_fc_branch=True.")
        if self.use_fc_branch:
            if self.fc_branch is None:
                raise RuntimeError("FC branch is enabled but was not initialized.")
            features.append(self.fc_branch(x_fc))
        if self.use_spectral_branch:
            if self.spectral_branch is None:
                raise RuntimeError("Spectral branch is enabled but was not initialized.")
            features.append(self.spectral_branch(x_time))
        if len(features) == 1:
            return features[0]
        return self.fusion(torch.cat(features, dim=-1))

    def forward(
        self,
        x_time: torch.Tensor,
        x_fc: torch.Tensor = None,
        lambda_value: float = 0.0,
    ):
        features = self.extract_features(x_time, x_fc)
        label_logits = self.label_classifier(features)
        if not self.use_dann:
            return label_logits
        if self.grl is None or self.domain_classifier is None:
            raise RuntimeError("DANN is enabled but the domain head was not initialized.")
        reversed_features = self.grl(features, lambda_value)
        domain_logits = self.domain_classifier(reversed_features)
        return label_logits, domain_logits, features

