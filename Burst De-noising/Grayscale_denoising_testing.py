import os
import torch
import argparse
import numpy as np
import cv2

from pytorch_lightning import seed_everything

################# Dataset ##############################################

from dataset.grayscale_denoise_test_set import GrayscaleDenoiseTestSet
from torch.utils.data.dataloader import DataLoader
from data.postprocessing_functions import DenoisingPostProcess

################# Model ##############################################

from Network import Burstormer

################# Utils ##############################################
import lpips
lpips_fn = lpips.LPIPS(net='alex')
from pytorch_msssim import ssim
from utils.metrics import PSNR
psnr_fn = PSNR()

######################################################################

parser = argparse.ArgumentParser(description='Color Burst De-noising using Burstormer')

parser.add_argument('--input_dir', default='./input/gray_testset.npz', type=str, help='Path to Grayscale Noisy testset')
parser.add_argument('--result_dir', default='./Results/grayscale_denoised_output/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/grayscale_denoising/Burstormer.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load Burstormer ####################################################

model = Burstormer(mode='grayscale').load_from_checkpoint(args.weights, mode='grayscale')
model = model
model.cuda()
model.summarize()

#####################################################################

seed_everything(13)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)   

##### Load Color TestSet ############################################

gray_testset = GrayscaleDenoiseTestSet(root=args.input_dir, noise_level=8)
test_loader = DataLoader(gray_testset, batch_size=1, pin_memory=True)        
process_fn = DenoisingPostProcess(return_np=True)

#####################################################################

for i, data in enumerate(test_loader):
    
    burst, labels, info = data            
    noise_estimate = info['sigma_estimate']        

    burst = burst[:, :, :, :200, :316]
    labels = labels[:, :, :200, :316]
    noise_estimate = noise_estimate[:, :, :, :200, :316]

    burst = burst.cuda()
    labels = labels.cuda()
    noise_estimate = noise_estimate.cuda()
                
    with torch.no_grad():
        output = model(burst, noise_estimate)
    output = output.clamp(0.0, 1.0)
    
    ### PSNR computation
    PSNR = psnr_fn(output, labels).cpu().numpy()
    
    ### LPIPS computation
    var1 = 2*output-1
    var2 = 2*labels-1
    LPIPS = lpips_fn(var1.cpu(), var2.cpu())
    LPIPS = torch.squeeze(LPIPS).detach().numpy()
    
    ### SSIM computation            
    SSIM = ssim(output*255, labels*255, data_range=255, size_average=True)
    SSIM = torch.squeeze(SSIM).cpu().detach().numpy()
    
    eval_Par = 'Evaluation Measures for Burst {:d} ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f} \n'.format(i, PSNR, SSIM, LPIPS)
    print(eval_Par)       
    
    noisy_input = torch.mean(burst, 1)
    noisy_input = process_fn.process(noisy_input[0].cpu(), info)            
    output = process_fn.process(output[0].cpu(), info)            
    labels = process_fn.process(labels[0].cpu(), info)
    
    output = np.concatenate((noisy_input, output, labels), axis=1)
    
    cv2.imwrite('{}/{}.png'.format(args.result_dir, str(i)), output)