## Burstormer, Option B (RGB-native) -- trained on coregistered GoPro bursts
## instead of the Zurich RAW-to-RGB synthetic pipeline.
##
## Drop this file next to Burstormer_Track_1_training.py in "Burst Super-resolution/",
## along with Network_option_b.py and datasets/gopro_burst_dataset.py.
## Point --outdir of build_bursts.py at the same path as `dataset_root` below.

import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
seed_everything(13)

from Network_option_b import Burstormer
from datasets.gopro_burst_dataset import GoProBurstDataset
from torch.utils.data.dataloader import DataLoader

log_dir = './logs/Track_1_GoPro/'


class Args:
    def __init__(self):
        self.dataset_root = "./gopro_dataset"   # output of build_bursts.py
        self.model_dir = log_dir + "saved_model"
        self.NUM_WORKERS = 6


args = Args()


def load_data(dataset_root):
    train_dataset = GoProBurstDataset(os.path.join(dataset_root, "train"))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                               drop_last=True, num_workers=args.NUM_WORKERS, pin_memory=True)

    val_dataset = GoProBurstDataset(os.path.join(dataset_root, "val"))
    val_loader = DataLoader(val_dataset, batch_size=1,
                             num_workers=args.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


model = Burstormer()
model.cuda()

# model.summarize() was removed from LightningModule in modern pytorch_lightning;
# use the standalone utility instead.
try:
    from pytorch_lightning.utilities.model_summary import summarize
    print(summarize(model, max_depth=1))
except ImportError:
    pass

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir, exist_ok=True)

train_loader, val_loader = load_data(args.dataset_root)

checkpoint_callback = ModelCheckpoint(
    monitor='val_psnr',
    dirpath=args.model_dir,
    filename='{epoch:02d}-{val_psnr:.2f}',
    save_top_k=3,
    save_last=True,
    mode='max',
)

# NOTE -- modern pytorch_lightning (2.x) Trainer kwargs. The original repo's
# script was written against pytorch_lightning==1.5.10 and used gpus=-1,
# accelerator='ddp', auto_select_gpus=True, precision=16,
# progress_bar_refresh_rate=100 -- all of which were removed/renamed since.
# If you deliberately installed the pinned install.yml env instead, use the
# original kwargs from Burstormer_Track_1_training.py rather than these.
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=300,
    precision="16-mixed",
    gradient_clip_val=0.01,
    callbacks=[checkpoint_callback],
    benchmark=True,
    deterministic=False,        # or "warn" — DeformConv2d's backward isn't deterministic on CUDA
    val_check_interval=0.25,
    enable_progress_bar=True,
    # resume from a checkpoint: ckpt_path=args.model_dir + '/last.ckpt'
)

trainer.fit(model, train_loader, val_loader)