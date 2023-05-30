import os
import torch
import argparse
import numpy as np
import cv2

from pytorch_lightning import seed_everything

################# Dataset ##############################################

from dataset.color_denoise_test_set import ColorDenoiseTestSet
from torch.utils.data.dataloader import DataLoader
from data.postprocessing_functions import DenoisingPostProcess

################# Model ##############################################

from Network import Burstormer

################# Utils ##############################################
import lpips
lpips_fn = lpips.LPIPS(net='alex')
from pytorch_msssim import ssim
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=8)

######################################################################

parser = argparse.ArgumentParser(description='Color Burst De-noising using Burstormer')

parser.add_argument('--input_dir', default='./input/color_testset', type=str, help='Path to Color Noisy testset')
parser.add_argument('--result_dir', default='./Results/color_denoised_output/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/color_denoising/Burstormer.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load Burstormer ####################################################

model = Burstormer(mode='color').load_from_checkpoint(args.weights, mode='color')

print("Model Loaded")
model.cuda()
model.summarize()

#####################################################################

seed_everything(13)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)   

##### Load Color TestSet ############################################

color_testset = ColorDenoiseTestSet(root=args.input_dir, noise_level=8)
test_loader = DataLoader(color_testset, batch_size=1, pin_memory=True)        
process_fn = DenoisingPostProcess(return_np=False)

#####################################################################

score = 0
PSNR = []
SSIM = []
LPIPS = []

i=0

for i, data in enumerate(test_loader):
    
    burst, labels, info = data            
    noise_estimate = info['sigma_estimate']

    if noise_estimate.dim() == 4:
        noise_estimate_re = noise_estimate.unsqueeze(1).repeat(1, burst.shape[1], 1, 1, 1)
    else:
        noise_estimate_re = noise_estimate

    burst = burst.cuda()
    labels = labels.cuda()
    noise_estimate = noise_estimate.cuda()

    burst = torch.cat((burst, noise_estimate), dim=2)


    with torch.no_grad():
        output = model(burst)
        output = output.clamp(0.0, 1.0)

    output1 = process_fn.process(output, info)
    labels1 = process_fn.process(labels, info)
    
    ### PSNR computation 
    PSNR.append(psnr_fn(output1, labels1).cpu().numpy())
    
    ### LPIPS computation
    var1 = 2*output1-1
    var2 = 2*labels1-1
    LPIPS.append(torch.squeeze(lpips_fn(var1.cpu(), var2.cpu())).detach().numpy())
    
    ### SSIM computation  
    SSIM.append(torch.squeeze(ssim(output1*255, labels1*255, data_range=255, size_average=True)).cpu().detach().numpy())
    
    noisy_input = burst[0,0, :4].permute(1,2,0).contiguous().cpu().numpy()
    noisy_input = cv2.cvtColor(noisy_input, cv2.COLOR_BGR2RGB)
    
    output = output[0].permute(1,2,0).contiguous().cpu().numpy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    labels = labels[0].permute(1,2,0).contiguous().cpu().numpy()
    labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
    
    output = np.concatenate((noisy_input, output, labels), axis=1)*255
    
    cv2.imwrite('{}/{}.png'.format(args.result_dir, str(i)), output)

Average_PSNR = sum(PSNR)/len(PSNR)
Average_SSIM = sum(SSIM)/len(SSIM)
Average_LPIPS = sum(LPIPS)/len(LPIPS)
average_eval_par = '\nAverage Evaluation Measures ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f}\n'.format(Average_PSNR, Average_SSIM, Average_LPIPS)

print(average_eval_par)
