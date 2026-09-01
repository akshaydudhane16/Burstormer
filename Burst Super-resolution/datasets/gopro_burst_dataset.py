"""PyTorch Dataset for the burst dataset produced by build_bursts.py.

Drop this file into `Burst Super-resolution/datasets/` alongside the
repo's other dataset classes.

Expected layout (written by build_bursts.py):
    root/
      scene_000000/
        hr.png
        lr/im_00.png ... im_{k-1}.png
        meta.json
      scene_000001/
        ...

Returns a tuple with the same shape as Burstormer's training_step /
validation_step expect (`x, y, flow_vectors, meta_info = batch`):
    burst:        float tensor [burst_size, 3, h, w] in [0, 1]
    frame_gt:     float tensor [3, H, W] in [0, 1]
    flow_vectors: placeholder (unused by the stock training/validation_step;
                   real per-frame homographies are in meta.json if you want
                   to wire up an auxiliary alignment loss later)
    meta_info:    dict of a few plain scalars (safe for default_collate at
                   batch_size=1, which is what the training scripts use)
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GoProBurstDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.scenes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if not self.scenes:
            raise RuntimeError(f"No scene_* folders found under {root}")

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def _load_rgb01(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, index):
        scene_name = self.scenes[index]
        scene_dir = os.path.join(self.root, scene_name)

        hr = self._load_rgb01(os.path.join(scene_dir, "hr.png"))
        frame_gt = torch.from_numpy(hr).permute(2, 0, 1).contiguous()

        lr_dir = os.path.join(scene_dir, "lr")
        lr_files = sorted(os.listdir(lr_dir))
        burst_np = np.stack([self._load_rgb01(os.path.join(lr_dir, f)) for f in lr_files], axis=0)
        burst = torch.from_numpy(burst_np).permute(0, 3, 1, 2).contiguous()  # [burst_size, 3, h, w]

        meta_path = os.path.join(scene_dir, "meta.json")
        meta_info = {"scene": scene_name}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                raw = json.load(f)
            meta_info.update({
                "sr_factor": raw.get("sr_factor", 4),
                "reference_index": raw.get("reference_index", 0),
                "clip": raw.get("clip", ""),
            })

        flow_vectors = torch.zeros(1)  # unused by the stock loss; see docstring

        return burst, frame_gt, flow_vectors, meta_info