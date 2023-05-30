## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os
import cv2
import torch
import argparse
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

seed_everything(50)

######################################## Model and Dataset ########################################################
from Network import burstormer
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst

##################################################################################################################

parser = argparse.ArgumentParser(description='Synthetic burst super-resolution using Burstormer')

parser.add_argument('--input_dir', default='./syn_burst_val', type=str, help='Directory of NTIRE21 BurstSR validation images')
parser.add_argument('--result_dir', default='./Results/Synthetic/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/Synthetic/epoch=292-val_psnr=42.83.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load Burstormer ####################################################

model = burstormer()
model = burstormer.load_from_checkpoint(args.weights)
model.cuda()
model.summarize()



######################################### Synthetic Burst Validation set #####################################

test_zurich_raw2rgb = ZurichRAW2RGB(root="./Zurich-RAW-to-DSLR-Dataset",  split='test')
test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=14, crop_sz=384)    
test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False)

##############################################################################################################

trainer = Trainer(gpus=-1,
                    auto_select_gpus=True,
                    accelerator='ddp',
                    max_epochs=300,
                    precision=16,
                    gradient_clip_val=0.01,
                    benchmark=True,#If true enables cudnn.benchmark. This flag is likely to increase the speed of your system if your input sizes don’t change. However, if it does, then it will likely make your system slower.
                    deterministic=False,
                    val_check_interval=0.125,                    #limit_train_batches=0.2,
                    progress_bar_refresh_rate=100,
                    resume_from_checkpoint = "./Trained_models/Synthetic/epoch=292-val_psnr=42.83.ckpt")

trainer.validate(model, test_loader, ckpt_path= "./Trained_models/Synthetic/epoch=292-val_psnr=42.83.ckpt")



######################################### NTIRE21 BurstSR Validation ####################################################

dataset = SyntheticBurstVal(args.input_dir)

result_dir = args.result_dir + 'Developement Phase'
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True) 


for idx in range(len(dataset)):

    burst, burst_name = dataset[idx]            

    print("Processing Burst:::: ", burst_name)

    burst = burst.cuda()
    burst = burst.unsqueeze(0)
    with torch.no_grad():
        net_pred = model(burst)

    # Normalize to 0  2^14 range and convert to numpy array
    net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

    # Save predictions as png
    cv2.imwrite('{}/{}.png'.format(result_dir, burst_name), net_pred_np)
