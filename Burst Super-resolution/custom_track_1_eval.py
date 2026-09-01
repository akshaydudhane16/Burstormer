## Burstormer Option B -- inference / evaluation on the GoPro burst dataset
## produced by build_bursts.py. Adapted from the repo's own Track_1_evaluation.py:
##
##   - loads a trained CHECKPOINT instead of training from scratch
##   - swaps the Zurich-RAW-to-RGB SyntheticBurst val set for GoProBurstDataset
##   - swaps `burstormer` (4-channel packed RAW) for Network_option_b's
##     `Burstormer` (3-channel RGB), matching what was actually trained
##   - output PNGs are standard 8-bit RGB (x255) instead of the original's
##     14-bit RAW-range (x2**14), since Option B's output/target is RGB, not RAW
##   - adds a per-scene bicubic-upsampled-reference-frame BASELINE and its
##     PSNR, so you can see the delta over "just upsample the single
##     reference frame" -- the sanity check burstormer-gopro-setup.md
##     Phase 2 recommends before trusting the network's numbers
##   - writes one row per scene to results.csv, plus optional pred/gt/bicubic
##     PNGs per scene, instead of only printing progress
##
## Usage:
##   # aggregate PSNR only, no image files written
##   python custom_Track_1_evaluation.py \
##       --dataset_root ./gopro_dataset \
##       --weights ./logs/Track_1_GoPro/saved_model/last.ckpt
##
##   # also dump pred/gt/bicubic PNGs per scene, and only look at 20 scenes
##   python custom_Track_1_evaluation.py \
##       --dataset_root ./gopro_dataset \
##       --weights ./logs/Track_1_GoPro/saved_model/epoch=142-val_psnr=43.71.ckpt \
##       --result_dir ./eval_results --save_images --limit 20

import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader

seed_everything(50)

torch.set_float32_matmul_precision('high')  # free perf on Ada/Ampere Tensor Cores, no accuracy cost here

######################################## Model and Dataset ########################################################

from Network_option_b import Burstormer
from datasets.gopro_burst_dataset import GoProBurstDataset
from utils.metrics import PSNR

######################################################################################################

parser = argparse.ArgumentParser(description='GoPro burst super-resolution evaluation (Option B, RGB-native Burstormer)')
parser.add_argument('--dataset_root', default='./gopro_dataset', type=str,
                     help='root dir written by build_bursts.py -- expects a val/ subfolder')
parser.add_argument('--weights', required=True, type=str,
                     help='path to a .ckpt file, e.g. logs/Track_1_GoPro/saved_model/last.ckpt '
                          'or one of the epoch=NN-val_psnr=NN.NN.ckpt top-k checkpoints')
parser.add_argument('--result_dir', default='./eval_results', type=str,
                     help='where to write results.csv (and images, if --save_images)')
parser.add_argument('--save_images', action='store_true',
                     help='also save pred.png / gt.png / bicubic.png per scene under '
                          '<result_dir>/<scene_name>/ -- off by default since a large val '
                          'set can mean a lot of disk')
parser.add_argument('--limit', type=int, default=None,
                     help='only evaluate the first N scenes (useful for a quick spot-check '
                          'instead of a full val-set pass)')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

######################################### Load Burstormer (Option B) ###############################################

model = Burstormer.load_from_checkpoint(args.weights)
model.cuda()
model.eval()

######################################### Aggregate PSNR via Trainer.validate ######################################
# Reuses the same validation_step / on_validation_epoch_end the training run logged
# val_psnr with, so this number is directly comparable to the curve you were
# watching in TensorBoard -- just computed over the FULL val set in one pass
# rather than the periodic val_check_interval sampling used during training.

val_dataset = GoProBurstDataset(os.path.join(args.dataset_root, "val"))
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    benchmark=True,
    deterministic=False,  # DeformConv2d's backward isn't deterministic on CUDA -- irrelevant here
                           # (no backward pass during validate) but kept consistent with training
    logger=False,          # this is an eval-only run; don't create a new lightning_logs/version_N
)
print("Running full-val-set aggregate PSNR (same metric as training's val_psnr)...")
trainer.validate(model, val_loader)

# PL's Trainer teardown moves the LightningModule back to CPU after
# fit/validate finishes (to free GPU memory) -- it does NOT leave it on GPU
# for further manual use. Move it back before the per-scene loop below, or
# every model(...) call there will hit a CPU-weights/CUDA-input mismatch.
model.cuda()
model.eval()

######################################### Per-scene outputs + baseline comparison ##################################

result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

psnr_fn = PSNR(boundary_ignore=40)  # same boundary_ignore Network_option_b.valid_psnr uses
rows = []

n_scenes = len(val_dataset) if args.limit is None else min(args.limit, len(val_dataset))

for idx in range(n_scenes):
    burst, frame_gt, flow_vectors, meta_info = val_dataset[idx]
    scene_name = meta_info["scene"]

    burst_in = burst.unsqueeze(0).cuda()   # [1, burst_size, 3, h, w] -- matches training_step's `x`
    gt = frame_gt.unsqueeze(0).cuda()      # [1, 3, H, W]

    with torch.no_grad():
        pred = model(burst_in).clamp(0.0, 1.0)

    net_psnr = psnr_fn(pred, gt).item()

    # Naive baseline: bicubic-upsample the burst's own reference frame (index 0,
    # per GoProBurstDataset/build_bursts.py convention). This is the "did the
    # network learn anything beyond single-frame upsampling" check from
    # burstormer-gopro-setup.md's Phase 2 recommendation.
    ref_lr = burst_in[:, 0]  # [1, 3, h, w]
    bicubic = F.interpolate(ref_lr, size=gt.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    bicubic_psnr = psnr_fn(bicubic, gt).item()

    print(f"[{idx + 1}/{n_scenes}] {scene_name}: "
          f"net_psnr={net_psnr:.2f} dB, bicubic_psnr={bicubic_psnr:.2f} dB, "
          f"delta={net_psnr - bicubic_psnr:+.2f} dB")

    rows.append({
        "scene": scene_name,
        "net_psnr": round(net_psnr, 3),
        "bicubic_psnr": round(bicubic_psnr, 3),
        "delta_db": round(net_psnr - bicubic_psnr, 3),
    })

    if args.save_images:
        scene_dir = os.path.join(result_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        def to_bgr_u8(t):
            arr = (t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).cpu().numpy().astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(scene_dir, "pred.png"), to_bgr_u8(pred))
        cv2.imwrite(os.path.join(scene_dir, "gt.png"), to_bgr_u8(gt))
        cv2.imwrite(os.path.join(scene_dir, "bicubic.png"), to_bgr_u8(bicubic))

######################################### Write summary CSV ########################################################

csv_path = os.path.join(result_dir, "results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scene", "net_psnr", "bicubic_psnr", "delta_db"])
    writer.writeheader()
    writer.writerows(rows)

mean_net = float(np.mean([r["net_psnr"] for r in rows]))
mean_bicubic = float(np.mean([r["bicubic_psnr"] for r in rows]))

print(f"\nDone. {len(rows)} scene(s) evaluated.")
print(f"Mean net PSNR:     {mean_net:.2f} dB")
print(f"Mean bicubic PSNR: {mean_bicubic:.2f} dB")
print(f"Mean delta:        {mean_net - mean_bicubic:+.2f} dB")
print(f"Per-scene results written to {csv_path}")
if args.save_images:
    print(f"Images written under {result_dir}/<scene_name>/{{pred,gt,bicubic}}.png")