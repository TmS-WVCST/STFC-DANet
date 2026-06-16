# STFC-DANet

Repository: [TianmingSong/STFC-DANet](https://github.com/TianmingSong/STFC-DANet)

本仓库提供 **STFC-DANet** 的官方实现，用于 ABIDE rs-fMRI 的 ASD（自闭症谱系障碍）二分类诊断。模型以 `CNN-Transformer` 为主干，在其上融合三个模块：

```text
Backbone:    CNN-Transformer (ROI time series)
Innovation 1: Spectral Branch (低频 BOLD 频谱)
Innovation 2: FC Branch (静态功能连接)
Innovation 3: DANN (跨站点域对抗)
```

## Highlights

1. **STFC-DANet 模型。** 融合时域动态、低频谱信息、静态功能连接与域对抗学习，同时建模 ROI time series、spectral BOLD fluctuation、functional connectivity 与跨站点 domain shift。

2. **跨站点泛化验证。** 提供标准 5-fold 交叉验证与 Leave-One-Site-Out (LOSO) 协议，评估模型对不同采集站点分布偏移的泛化能力。

## Setup

以下命令均假设当前工作目录为本仓库根目录：

```bash
pip install -r requirements.txt
```

## Data

由于体积过大，原始与预处理后的数据文件 **未包含在仓库中**，需要本地自行准备：

```text
ABIDE_CC400_NotCC_NotQC_shufflerandomSeed42.npz              # 作者提供的 FC 连接 npz
ABIDE_CC400_NOTQC_TPE_withSITE_shuffle_randomSeed1234_filtGlobal.npz  # 作者提供的时序 npz
```

将以上两个 `.npz` 放在仓库根目录后，运行预处理脚本生成本地训练缓存：

```bash
python scripts/preprocess_author_npz.py \
  --connectivity-npz ABIDE_CC400_NotCC_NotQC_shufflerandomSeed42.npz \
  --timeseries-npz ABIDE_CC400_NOTQC_TPE_withSITE_shuffle_randomSeed1234_filtGlobal.npz \
  --output-dir data/abide_cc400_author_npz
```

预处理脚本会自动创建 `data/abide_cc400_author_npz/` 目录并生成：

```text
X.npy              # ROI time series
fc.npy             # 功能连接矩阵上三角 flatten 特征
y.npy              # ASD / control 标签
domain.npy         # 站点 domain label
metadata.csv
site_mapping.json
preprocess_summary.json
```

上述生成文件均为本地数据缓存，已在 `.gitignore` 中忽略，不随仓库分发。

## Network Overview

### Backbone: CNN-Transformer

主干只使用 ROI time series：

```text
ROI time series [B, T, ROI]
→ Multi-scale temporal CNN stem
→ residual temporal CNN blocks
→ Transformer Encoder
→ CLS / mean / max pooling
→ time feature
→ classifier
```

该分支学习 BOLD 时间序列中的局部动态模式和长程时间依赖。

### Innovation 1: Spectral Branch

从同一份 ROI time series 中提取低频谱信息：

```text
ROI time series
→ rFFT along time
→ remove DC component
→ keep first 32 low-frequency bins
→ log1p magnitude
→ per-ROI spectral projection
→ mean / max pooling over ROI
→ spectral feature
```

理论动机：rs-fMRI 中的 BOLD 信号主要反映低频自发神经活动，ASD 相关异常不仅可能出现在时域动态模式中，也可能体现在不同 ROI 的低频振荡强度和频谱能量分布上。Spectral Branch 将 ROI time series 转换到频域，直接建模低频 BOLD fluctuation，为时域分支提供互补信息。当前默认设置：

```text
spectral_bins = 32
```

### Innovation 2: FC Branch

使用静态功能连接特征：

```text
FC upper triangle [B, 76636]
→ Linear(76636, 512)
→ LayerNorm + GELU + Dropout
→ Linear(512, 128)
→ LayerNorm + GELU + Dropout
→ Linear(128, 128)
→ FC feature
```

理论动机：ASD 被认为与脑功能网络连接异常有关。ROI time series 反映动态活动轨迹，而功能连接矩阵刻画 ROI 之间的长期同步关系。FC Branch 将 pairwise functional connectivity 作为独立输入，补充 time branch 难以直接捕获的全局连接模式。默认使用 `FC-MLP`，另提供 `wide_mlp`、`gated_mlp`、`transformer` 作为 FC 分支结构对比。

### Innovation 3: DANN

DANN 用于降低 ABIDE 多站点数据的站点偏差：

```text
fused feature
→ label classifier
→ Gradient Reversal Layer
→ domain classifier
```

`dann_lambda` 默认值：

```text
dann_lambda = 0.1
```

理论动机：ABIDE 是典型的多站点数据集，不同站点在扫描仪、采集协议、被试组成和预处理质量上存在差异。DANN 通过 Gradient Reversal Layer 让共享特征在保持 ASD 分类能力的同时降低站点可判别性，鼓励模型学习 site-invariant representation。

### Final Model

```text
CNN-Transformer Time Branch
+ Spectral Branch
+ FC-MLP Branch
+ DANN
```

三路特征融合后进入 label classifier 和 domain adversarial head。

## Training

### Final Model（5-fold 交叉验证）

```bash
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --spectral-bins 32 --output-dir stfc_danet
```

`--output-dir` 为相对名称时会保存到 `outputs/<name>`；绝对路径或以 `outputs/` 开头的路径按原路径使用。

### Ablation Studies

```bash
# Baseline: 仅 CNN-Transformer
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --use-fc-branch 0 --use-dann 0 --output-dir abl_baseline

# + Spectral Branch
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --use-fc-branch 0 --use-dann 0 --output-dir abl_spectral

# + FC Branch
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --use-fc-branch 1 --use-dann 0 --output-dir abl_fc

# + DANN
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --use-fc-branch 0 --use-dann 1 --output-dir abl_dann

# + Spectral + FC
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --use-fc-branch 1 --use-dann 0 --output-dir abl_spectral_fc

# + Spectral + DANN
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --use-fc-branch 0 --use-dann 1 --output-dir abl_spectral_dann

# + FC + DANN
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --use-fc-branch 1 --use-dann 1 --output-dir abl_fc_dann

# Full model: Spectral + FC + DANN
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --use-fc-branch 1 --use-dann 1 --output-dir abl_all
```

可选 FC branch 结构对比：

```bash
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --fc-branch-type wide_mlp --output-dir abl_fc_wide_mlp
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --fc-branch-type gated_mlp --output-dir abl_fc_gated_mlp
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 0 --fc-branch-type transformer --output-dir abl_fc_transformer
```

可选 spectral bins 对比：

```bash
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --spectral-bins 16 --output-dir abl_all_spectral_bins16
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --spectral-bins 24 --output-dir abl_all_spectral_bins24
python scripts/train_cnn_transformer_cv.py --use-spectral-branch 1 --spectral-bins 48 --output-dir abl_all_spectral_bins48
```

### Leave-One-Site-Out (LOSO)

LOSO 用于评估模型跨采集站点泛化能力。CC400 缓存包含 20 个站点，该协议依次留出一个站点作为测试集，其余 19 个站点用于训练和验证。

```bash
python scripts/train_cnn_transformer_loso.py --output-dir loso_stfc_danet --device cuda
```

如果没有 CUDA：

```bash
python scripts/train_cnn_transformer_loso.py --output-dir loso_stfc_danet --device cpu --no-amp
```

## Outputs

每个实验目录会保存：

```text
fold_metrics.csv
summary_metrics.csv
summary_metrics.json
summary_metrics_raw.csv
summary_metrics_raw.json
fold_*/history.csv
fold_*/metrics.json
fold_*/model.pt
fold_*/best_model.pt
```

控制台最后输出：

```text
Accuracy
Sensitivity
Specificity
F1
AUC
```
