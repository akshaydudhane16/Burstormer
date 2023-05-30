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
from pytorch_msssim import ssim
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=2)
loss_fn_alex = lpips.LPIPS(net='alex')


######################################################################

parser = argparse.ArgumentParser(description='Color Burst De-noising using Burstormer')

parser.add_argument('--input_dir', default='./input/gray_testset.npz', type=str, help='Path to Grayscale Noisy testset')
parser.add_argument('--result_dir', default='./Results/grayscale_denoised_output/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/grayscale_denoising/Burstormer.ckpt', type=str, help='Path to weights')

args = parser.parse_args()

######################################### Load Burstormer ####################################################

model = Burstormer(mode='grayscale').load_from_checkpoint(args.weights, mode='grayscale')
model.cuda()
model.summarize()

#####################################################################

seed_everything(13)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)   

##### Load Color TestSet ############################################

gray_testset = GrayscaleDenoiseTestSet(root=args.input_dir, noise_level=8)
test_loader = DataLoader(gray_testset, batch_size=1, pin_memory=True)        
process_fn = DenoisingPostProcess(return_np=False)

#####################################################################


score = 0
PSNR = []
SSIM = []
LPIPS = []


for i, data in enumerate(test_loader):
    
    burst, labels, info = data
    
    noise_estimate = info['sigma_estimate']
    
    if noise_estimate.dim() == 4:
        noise_estimate_re = noise_estimate.unsqueeze(1).repeat(1, burst.shape[1], 1, 1, 1)
    else:
        noise_estimate_re = noise_estimate

    burst = torch.cat((burst, noise_estimate_re), dim=2)
    
    burst = burst[:, :, :, :200, :320]
    labels = labels[:, :, :200, :320]

    burst = burst.cuda()
    labels = labels.cuda()
             
    with torch.no_grad():
        output = model(burst)
        output = output.clamp(0.0, 1.0)
    
    ### PSNR computation
    output = process_fn.process(output, info)
    labels = process_fn.process(labels, info)
          
    PSNR.append(psnr_fn(output, labels).cpu().numpy() )
    
    ### LPIPS computation
    var1 = 2*output-1
    var2 = 2*labels-1
    LPIPS.append(torch.squeeze(loss_fn_alex(var1.cpu(), var2.cpu())).detach().numpy())
    
    ### SSIM computation
    SSIM.append(torch.squeeze(ssim(output*255, labels*255, data_range=255, size_average=True)).cpu().detach().numpy())          
    
    noisy_input = burst[0,0, :1].permute(1,2,0).contiguous().cpu().numpy()            
    output = output[0].permute(1,2,0).contiguous().cpu().numpy()            
    labels = labels[0].permute(1,2,0).contiguous().cpu().numpy()
    
    output = np.concatenate((noisy_input, output, labels), axis=1)*255
    
    cv2.imwrite('{}/{}.png'.format(args.result_dir, str(i)), output)
    
    i+=1
    
Average_PSNR = sum(PSNR)/len(PSNR)
Average_SSIM = sum(SSIM)/len(SSIM)
Average_LPIPS = sum(LPIPS)/len(LPIPS)
average_eval_par = '\nAverage Evaluation Measures ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f}\n'.format(Average_PSNR, Average_SSIM, Average_LPIPS)
        
print(average_eval_par)
