from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGETS = ["steering", "throttle", "brake"]


class DrivingDataset(Dataset):
    def __init__(self, manifest_csv: Path, raw_dir: Path, augment: bool = False):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.raw_dir = raw_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.raw_dir / row["image_path"]).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1, :].copy()
                steering = -float(row["steering"])
            else:
                steering = float(row["steering"])
            if np.random.rand() < 0.35:
                arr = np.clip(arr * np.random.uniform(0.75, 1.25), 0, 1)
            if np.random.rand() < 0.25:
                arr = np.clip(arr + np.random.normal(0, 0.018, arr.shape), 0, 1)
        else:
            steering = float(row["steering"])

        arr = arr.astype(np.float32, copy=False)
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        y = torch.tensor([steering, float(row["throttle"]), float(row["brake"])], dtype=torch.float32)
        return x, y
