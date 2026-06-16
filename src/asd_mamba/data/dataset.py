from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_labels(data_dir: Path) -> np.ndarray:
    return np.load(Path(data_dir) / "y.npy")


class AbideDannDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        indices: Iterable[int],
        fc_mean: Optional[np.ndarray] = None,
        fc_std: Optional[np.ndarray] = None,
        require_fc: bool = True,
        require_domain: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.X = np.load(self.data_dir / "X.npy", mmap_mode="r")
        self.fc = np.load(self.data_dir / "fc.npy", mmap_mode="r") if require_fc else None
        self.y = np.load(self.data_dir / "y.npy", mmap_mode="r")
        self.domain = np.load(self.data_dir / "domain.npy", mmap_mode="r") if require_domain else None
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.fc_mean = fc_mean.astype(np.float32) if fc_mean is not None else None
        self.fc_std = fc_std.astype(np.float32) if fc_std is not None else None

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        x = torch.from_numpy(np.array(self.X[index], dtype=np.float32, copy=True))
        if self.fc is None:
            fc_tensor = torch.empty(0, dtype=torch.float32)
        else:
            fc = np.asarray(self.fc[index], dtype=np.float32)
            if self.fc_mean is not None and self.fc_std is not None:
                fc = (fc - self.fc_mean) / self.fc_std
            fc_tensor = torch.from_numpy(fc.astype(np.float32))
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        domain_value = 0 if self.domain is None else int(self.domain[index])
        domain = torch.tensor(domain_value, dtype=torch.long)
        return x, fc_tensor, y, domain


def load_domains(data_dir: Path) -> np.ndarray:
    return np.load(Path(data_dir) / "domain.npy")


def load_fc(data_dir: Path) -> np.ndarray:
    return np.load(Path(data_dir) / "fc.npy", mmap_mode="r")

