#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.ops import deform_conv2d, DeformConv2d
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
torch.cuda.empty_cache()
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision import transforms as transforms
import sys
sys.path.append('../')
import os
import math
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from utils.data_format_utils import torch_to_numpy, numpy_to_torch
from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import SimplePostProcess
#from pwcnet.pwcnet import PWCNet
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)
from utils.warp import warp
import torch.nn.functional as F
import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import * 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#import lpips
from pytorch_msssim import ssim
import time

import glob
import cv2
import numpy as np
from torchvision.utils import save_image
import random
from torch.nn.modules.utils import _pair, _single

from collections import OrderedDict

cudnn.benchmark = True


# In[2]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[ ]:


import os
import sys
import tempfile
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

################# Dataset ##############################################

from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.synthetic_burst_val_set import SyntheticBurstVal

################# Utils ##############################################

from utils.data_format_utils import torch_to_numpy, numpy_to_torch
from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import SimplePostProcess
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)
from utils.warp import warp
from pytorch_msssim import ssim

import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import *

###################################################################################################

import time
import glob
import cv2
import numpy as np



######################################## Model ########################################################

from torchvision.ops import deform_conv2d, DeformConv2d


class combined(nn.Module):
    
    def __init__(self, in_channels):
        
        super(combined, self).__init__()
        
        self.mod1 = InceptionBlock(in_channels)
        
        self.mod2 = CHW_nonlocal(64)    
        
    def forward(self, x):
        
        return self.mod2(self.mod1(x))  


class RG(nn.Module):
    
    def __init__(self, in_channels, num_rcab):
        
        super(RG, self).__init__()
        
        self.module = [combined(in_channels)  for _ in range(num_rcab)]
        
        self.module.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
        
        self.module = nn.Sequential(*self.module)
        
    def forward(self, x):
        
        return x + self.module(x)  

def conv1x1(in_channels, out_channels, stride=1):
    
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
        
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)
class UPN(nn.Module):
        
    def __init__(self, n_feats):
            
        super(UPN, self).__init__()
        self.conv21 = conv3x3(n_feats, n_feats*4)
        self.ps21 = nn.PixelShuffle(2)

        self.conv31 = conv3x3(n_feats, n_feats*4)
        self.ps31 = nn.PixelShuffle(2)
        self.conv32 = conv3x3(n_feats, n_feats*4)
        self.ps32 = nn.PixelShuffle(2)

        self.conv41 = conv3x3(n_feats, n_feats*4)
        self.ps41 = nn.PixelShuffle(2)
        self.conv42 = conv3x3(n_feats, n_feats*4)
        self.ps42 = nn.PixelShuffle(2)
        self.conv43 = conv3x3(n_feats, n_feats*4)
        self.ps43 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True) 

    def forward(self,x):
        
        x1 = x 
        x2 = self.conv21(x) 
        x2 = self.lrelu (self.ps21 (x2))

        x3 = self.conv31(x)
        x3 =  self.lrelu (self.ps31(x3))
        x3 =  self.conv32(x3)
        x3 =  self.lrelu (self.ps32(x3))

        x4 =  self.conv41(x)
        x4 =  self.lrelu(self.ps41(x4))
        x4 =  self.conv42(x4)
        x4 =  self.lrelu(self. ps42(x4))
        x4 =  self.conv43(x4)
        x4 =  self.lrelu(self.ps43(x4))

        return x1, x2, x3, x4

           
    
class feat_int(nn.Module):
    
    def __init__(self, n_feats):
        super(feat_int, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv14 = conv1x1(n_feats, n_feats)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True) 

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv24 = conv1x1(n_feats, n_feats)

        self.conv34 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv41_1 = conv3x3(n_feats, n_feats, 2)
        self.conv41_2 = conv3x3(n_feats, n_feats, 2)
        self.conv41_3  = conv3x3(n_feats, n_feats, 2)
        self.conv42_1 = conv3x3(n_feats, n_feats, 2)
        self.conv42_2 = conv3x3(n_feats, n_feats, 2)
        self.conv43 = conv3x3(n_feats, n_feats, 2)

        self.merge1 = conv3x3(n_feats*4, n_feats)
        self.merge2 = conv3x3(n_feats*4, n_feats)
        self.merge3 = conv3x3(n_feats*4, n_feats)
        self.merge4 = conv3x3(n_feats*4, n_feats)
        
    def forward(self, x1, x2, x3,x4):
        
        x12 = F.interpolate(x1, scale_factor = 2, mode = 'bilinear')
        x12 = self.lrelu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor = 4, mode = 'bilinear')
        x13 = self.lrelu(self.conv13(x13))
        x14 = F.interpolate(x1, scale_factor = 8, mode = 'bilinear')
        x14 = self.lrelu(self.conv14(x14))

        x21 = self.lrelu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor = 2, mode = 'bilinear')
        x23 = self.lrelu(self.conv23(x23))
        x24 = F.interpolate(x2, scale_factor = 4, mode = 'bilinear')
        x24 = self.lrelu(self.conv24(x24))

        x34 = F.interpolate(x3, scale_factor = 2, mode = 'bilinear')
        x34 = self.lrelu(self.conv34(x34))

        x31 = self.lrelu(self.conv31_1(x3))
        x31 = self.lrelu(self.conv31_2(x31))
        x32 = self.lrelu(self.conv32(x3))

        x41 = self.lrelu(self.conv41_1(x4))
        x41 = self.lrelu(self.conv41_2(x41))
        x41 = self.lrelu(self.conv41_3(x41))
        x42 = self.lrelu(self.conv42_1(x4))
        x42 = self.lrelu(self.conv42_2(x42))
        x43 = self.lrelu(self.conv43(x4))

        x1 = self.lrelu(self.merge1( torch.cat((x1, x21, x31, x41), dim=1)))
        x2 = self.lrelu(self.merge2( torch.cat((x2, x12, x32, x42), dim=1)))
        x3 = self.lrelu(self.merge3( torch.cat((x3, x13, x23, x43), dim=1)))
        x4 = self.lrelu(self.merge4( torch.cat((x4, x14, x24, x34), dim=1)))

        return x1, x2, x3, x4
    
class final_feat(nn.Module):

    def __init__(self, n_feats):
        
        super(final_feat, self).__init__()
        
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv14 = conv1x1(n_feats, n_feats)
        self.conv21 = conv1x1(n_feats, n_feats)
        self.conv22 = conv1x1(n_feats, n_feats)
        self.conv31 = conv1x1(n_feats, n_feats)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True) 
        self.merge1 = conv3x3(n_feats*4, n_feats)

    def forward(self, x1, x2, x3, x4):
        
        x12 = F.interpolate(x1, scale_factor = 2, mode = 'bicubic')
        x12 = self.lrelu(self.conv12(x12))
        x13 = F.interpolate(x12, scale_factor = 2, mode = 'bicubic')
        x13 = self.lrelu(self.conv13(x13))
        x14 = F.interpolate(x13, scale_factor = 2, mode = 'bicubic')
        x14 = self.lrelu(self.conv14(x14))
      
        x21 = F.interpolate(x2, scale_factor = 2, mode = 'bicubic')
        x21 = self.lrelu(self.conv21(x21)) 
        x22 = F.interpolate(x21, scale_factor = 2, mode = 'bicubic')
        x22 = self.lrelu(self.conv22(x22))

        x31 = F.interpolate(x3, scale_factor = 2, mode = 'bicubic')
        x31 = self.lrelu(self.conv31(x31))

        x4 = x4 
  
        x5 = self.lrelu(x14+ x22+ x31+ x4)
        
        return x5
        
class Upsam(nn.Module):

    def __init__(self, n_feats=64):
        super(Upsam, self).__init__()

        self.upi = UPN(n_feats=64)

        self.upm = feat_int(n_feats=64)

        self.upf = final_feat(n_feats=64)
         
        self.lo = CHW_nonlocal()

    def forward(self,x):
        
        feat_1, feat_2, feat_3, feat_4 = self.upi(x) 
        
        feat_21, feat_22, feat_23, feat_24 = self.upm(feat_1, feat_2, feat_3, feat_4)
        
        feat_3 = self.upf(feat_21, feat_22, feat_23, feat_24)
        
        feat_4 = self.lo(feat_3)
        
        return feat_4
                
class RCAB(nn.Module):
    
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, bn=False, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), res_scale=1, groups =1):

        super(RCAB, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction

        modules_body = [nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups), act, nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups)]
        self.body   = nn.Sequential(*modules_body)

        self.gcnet = nn.Sequential(ContextBlock2d(n_feat, n_feat))
        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.gcnet(res)
        res = self.conv1x1(res)
        res += x
        
        return res
################################################## Non-local block

class NonLocalBlock2D(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        
        nn.init.constant_(self.W.bias, 0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

    def forward(self, x):
        
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1).contiguous()

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        z = W_y + x

        return z
    
#######################################################################################################
    
class InceptionBlock(torch.nn.Module):
    
    def __init__(self,  in_channels):
        
        super(InceptionBlock, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels,in_channels*4,1,padding =0, bias =False),nn.LeakyReLU(negative_slope=0.2,inplace=True),nn.Conv2d(in_channels*4, in_channels*2, 1, padding=0, bias=False),nn.LeakyReLU(negative_slope=0.2,inplace=True),nn.Conv2d(in_channels*2, in_channels, 1, padding=0, bias=False))
        self.branch1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1, padding=0)
        self.branch3x3 = nn.Conv2d(in_channels,in_channels,kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels,in_channels,kernel_size=5, padding=2)
        self.branch7x7 = nn.Conv2d(in_channels,in_channels,kernel_size=7, padding=3)
        self.branch1 = nn.Conv2d(in_channels,in_channels,kernel_size=1, padding=0)
        self.branch11a = nn.Conv2d(in_channels*2,in_channels,kernel_size=1, padding=0)
        self.branch12a = nn.Conv2d(in_channels*2,in_channels,kernel_size=1, padding=0)
        self.branch13a = nn.Conv2d(in_channels*2,in_channels,kernel_size=1, padding=0)
        self.branch14a = nn.Conv2d(in_channels,in_channels,kernel_size=1, padding=0)
        self.branch15a = nn.Conv2d(in_channels,in_channels,kernel_size=1, padding=0)
        self.branch16a = nn.Conv2d(in_channels,in_channels,kernel_size=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels*3,in_channels,kernel_size=1, padding=0)
        
    def forward(self,x):
        
        conv = self.seq(x)
        
        branch1 = self.branch1x1(conv)
        
        branch31 = self.branch3x3(conv)
        
        branch51 = self.branch5x5(conv)
        
        branch71 = self.branch7x7(conv)
        
        a1 = [branch31,branch1]
        
        o1 = torch.cat(a1,1)
        
        o1 = self.branch11a(o1)

        a2 = [branch51,o1]
        
        o2 = torch.cat(a2,1)
        
        o2 = self.branch12a(o2)

        a3 = [branch71,o2]
        
        o3 = torch.cat(a3,1)
        
        o3 = self.branch13a(o3)
        
        branch1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)(o1)
        
        branch2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)(o2)
        
        branch3 = nn.LeakyReLU(negative_slope=0.2,inplace=True)(o3)
        
        out = [branch1,branch2,branch3]
        
        out1 = torch.cat(out,1)
        
        out2 = self.conv1x1(out1)
        
        out1 = x + out2
        
        return out1

class NonLocalBlock2D_channel(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_channel, self).__init__()
        
        self.in_channels = in_channels
        
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        
        nn.init.constant_(self.W.weight, 0)        
        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        height = x.size(1)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)              ####bXCXC
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        z = W_y + x
        
        return z

class NonLocalBlock2D_heigth(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_heigth, self).__init__()
        
        self.in_channels = in_channels
        
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        
        nn.init.constant_(self.W.weight, 0)        
        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
    def forward(self, x):
        
        batch_size = x.size(0)
        
        height = x.size(2)
        
        g_x = self.g(x).view(batch_size, height, -1)
        
        theta_x = self.theta(x).view(batch_size, height, -1)
        
        phi_x = self.phi(x).view(batch_size, height, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)                         #########bXHXH
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, *x.size()[1:3], height)
        
        W_y = self.W(y)
        
        z = W_y + x
        
        return z

class NonLocalBlock2D_width(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_width, self).__init__()
        
        self.in_channels = in_channels
        
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        
        nn.init.constant_(self.W.weight, 0)        
        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        width = x.size(3)
        
        g_x = self.g(x).view(batch_size, width, -1)
        
        theta_x = self.theta(x).view(batch_size, width, -1)
        
        phi_x = self.phi(x).view(batch_size, width, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)                                   #########bXWXW
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()   
        
        y = y.view(batch_size, *x.size()[1:3], width)
        
        W_y = self.W(y)
        
        z = W_y + x
        
        return z

class CA(nn.Module):
    
    def __init__(self):
        
        super(CA,self).__init__()
        
        self.seq = nn.Sequential(nn.Conv2d(64,32,1,padding =0, bias =False), nn.Conv2d(32, 64, 1, padding=0, bias=False))
        
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)     #c  
        
        self.max_pool  = nn.AdaptiveMaxPool2d(1)            #1x1X32 
        
        self.sigmoid   = nn. Sigmoid()
    
    def forward(self,x):
        
        feat_add = self.avg_pool(x) + self.max_pool(x)
        
        y = self.seq(feat_add)  
        
        return self.sigmoid(y)
    
class CHW_nonlocal(nn.Module): 
    
    def __init__(self, nf=64):
        
        super( CHW_nonlocal, self).__init__()
        
        self.c1 = NonLocalBlock2D_channel(in_channels = nf, inter_channels = nf)
        
        self.c2 = NonLocalBlock2D_heigth(in_channels = nf, inter_channels = nf)
        
        self.c3 = NonLocalBlock2D_width(in_channels = nf, inter_channels = nf)
        
        self.c4 = nn.Conv3d(14, 14, 3, stride = 1, padding = 1)  
        
        self.ca = CA()
        
        
        
    def forward(self, x):
        
        a5 = self.c1(x) + self.c2(x) + self.c3(x)
        
        a5_att = F.softmax(a5, dim=1)
        
        a5 =  self.c4(x.unsqueeze(0))
        
        a6 = a5.squeeze(0) * a5_att
        
        out = x  + a6
        
        return out
    
class NonLocalBlock2D_channelfus(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_channelfus, self).__init__()
        
        self.in_channels = in_channels
        
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        
        nn.init.constant_(self.W.weight, 0)        
        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        height = x.size(1)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)              ####bXCXC
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        
        z = W_y + x
        
        return z

class NonLocalBlock2D_heigthfus(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_heigthfus, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        nn.init.constant_(self.W.weight, 0)        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)
    
    def forward(self, x, ref):
        
        batch_size = x.size(0)
        
        height = x.size(2)
        
        g_x = self.g(x).view(batch_size, height, -1)
        
        theta_x = self.theta(ref).view(batch_size, height, -1)
        
        phi_x = self.phi(x).view(batch_size, height, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)                         #########bXHXH
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        
        y = y.view(batch_size, *x.size()[1:3], height)
        
        W_y = self.W(y)
        
        z = W_y + x
        
        return z

class NonLocalBlock2D_widthfus(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        
        super(NonLocalBlock2D_widthfus, self).__init__()
        
        self.in_channels = in_channels
        
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)        
        
        nn.init.constant_(self.W.weight, 0)        
        
        nn.init.constant_(self.W.bias, 0)        
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, ref):
        
        batch_size = x.size(0)
        
        width = x.size(3)
        
        g_x = self.g(x).view(batch_size, width, -1)
        
        theta_x = self.theta(ref).view(batch_size, width, -1)
        
        phi_x = self.phi(x).view(batch_size, width, -1)
        
        phi_x = phi_x.permute(0, 2, 1).contiguous()
        
        f = torch.matmul(theta_x, phi_x)                                   #########bXWXW
        
        f_div_C = F.softmax(f, dim=1)
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()   
        
        y = y.view(batch_size, *x.size()[1:3], width)
        
        W_y = self.W(y)
        
        z = W_y + x
        
        return z

############################################################FUSION_MODULE####################################################
class CrossNonLocal_Fusion(nn.Module): 
    
    def __init__(self, nf=64):
        
        super(CrossNonLocal_Fusion, self).__init__()

        self.non_local_w = NonLocalBlock2D_widthfus(nf, inter_channels=nf)
        
        self.non_local_h = NonLocalBlock2D_heigthfus(nf, inter_channels=nf)
        
        self.non_local_c = NonLocalBlock2D_channelfus(nf, inter_channels=nf)

        # fusion conv: using 1x1 to save parameters and computation
        
        act = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.fea_fusion = nn.Sequential(nn.Conv2d(args.burst_size * nf * 3, nf, 1, 1, bias=True), act)

    def forward(self, aligned_fea):

        
        ref = aligned_fea[0].unsqueeze(0)
        
        c_l = []
        
        w_l = []
        
        h_l = []
        
        for i in range(args.burst_size):
            
            nbr = aligned_fea[i].unsqueeze(0)
            
            w_l.append(self.non_local_w(nbr,ref))
            
            h_l.append(self.non_local_h(nbr,ref))
            
            c_l.append(self.non_local_c(nbr))

        aligned_fea_w = torch.cat(w_l, dim=1)
        
        aligned_fea_c = torch.cat(c_l, dim=1)
        
        aligned_fea_h = torch.cat(h_l, dim=1)
        
        aligned_fea = torch.cat([aligned_fea_c, aligned_fea_h,aligned_fea_w], dim=1)
        
        fea = self.fea_fusion(aligned_fea)
        
        return fea   

class Base_Model(nn.Module):
    
    def __init__(self):
        
        super(Base_Model, self).__init__()

        self.train_loss = nn.L1Loss()
        
        self.valid_psnr = PSNR(boundary_ignore=40)
        
        self.upsam = Upsam()
        
        num_features = args.num_features
        
        self.burst_size = args.burst_size
        
        bias = False
        
        act = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv1 = nn.Sequential(nn.Conv2d(4, num_features, kernel_size=3, padding=1, bias=bias))

        self.encoder = nn.Sequential(*[RG(num_features, 3) for _ in range(3)])
        
        self.bottleneck = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1, bias=bias)
        
        #### Offset Setting   
        self.down1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, bias=bias)
        
        self.down2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, bias=bias)
        
        self.down3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, bias=bias)
        
        filter_size = 9
        
        deform_groups = 8
        
        out_channels = deform_groups * 3 * filter_size
        
        self.cas_offset_conv = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        self.offset_conv = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        self.offset_conv1 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        self.offset_conv2 = nn.Conv2d(num_features, out_channels, 3, stride=1, padding=1, bias=bias)
        
        #### Deform Conv Setting
        
        self.cas_deform = DeformConv2d(num_features, num_features, 3, padding = 1, groups = deform_groups)
        
        self.deform = DeformConv2d(num_features, num_features, 3, padding = 1, groups = deform_groups)
        
        self.deform1 = DeformConv2d(num_features, num_features, 3, padding = 1, groups = deform_groups)
        
        self.deform2 = DeformConv2d(num_features, num_features, 3, padding = 1, groups = deform_groups)
        
        #### Bottleneck conv
        
        self.cas_bottleneck = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act) 
        
        self.bottleneck = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck_o = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck_a = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck1 = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck1_o = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck1_a = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.bottleneck2 = nn.Sequential(nn.Conv2d(num_features*2, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        ### Non-Local block
        
        self.offset_fusion = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.offset_fusion0 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.offset_fusion1 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.offset_fusion2 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias), act)
        
        self.fusion = CrossNonLocal_Fusion(num_features)
        
        ### Attention
        
        self.level1_feat_ext = nn.Sequential(*[RG(num_features, 2) for _ in range(1)])
        
        self.level2_feat_ext = nn.Sequential(*[RG(num_features, 2) for _ in range(1)])
        
        #########################
        
        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        
        self.up1 = nn.ConvTranspose2d(num_features, num_features, 3, stride = 2, padding = 1, output_padding = 1, bias = bias)
        
        self.up2 = nn.ConvTranspose2d(num_features, num_features, 3, stride = 2, padding = 1, output_padding = 1, bias = bias)
        
        self.up3 = nn.ConvTranspose2d(num_features, num_features, 3, stride = 2, padding = 1, output_padding = 1, bias = bias)
        
        self.conv3 = nn.Sequential(nn.Conv2d(num_features, 3, kernel_size = 3, padding = 1, bias = bias))      
        
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim = 1)
        
        offset = torch.cat((o1, o2), dim = 1)
        
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def def_alignment(self, burst_feat):        ##########  [14,64,48,48]
        
        down_feat1 = self.down1(burst_feat)
        
        down_feat1 = self.level2_feat_ext(down_feat1)
        
        down_feat2 = self.down2(down_feat1)
        
        down_feat2 = self.level1_feat_ext(down_feat2)
        
        B, f, H, W = burst_feat.size()
        
        ref = burst_feat[0].unsqueeze(0)
        
        level1_ref = down_feat1[0].unsqueeze(0)
        
        level2_ref = down_feat2[0].unsqueeze(0)
        
        aligned_burst_feat = []
        
        for i in range(B):
            
            ################ Level 2 #################
            level2_cur = down_feat2[i].unsqueeze(0)
            
            level2_feat = self.bottleneck2(torch.cat([level2_ref, level2_cur], dim=1))
            
            level2_feat = self.offset_fusion2(level2_feat)
            
            up_level2_feat = F.interpolate(level2_feat, scale_factor = 2, mode = 'bilinear')*2
            
            level2_offset, level2_mask = self.offset_gen(self.offset_conv2(level2_feat))
            
            level2_aligned = self.deform2(level2_cur, level2_offset, level2_mask)
            
            up_level2_aligned = F.interpolate(level2_aligned, scale_factor = 2, mode = 'bilinear')       
                       
            ################ Level 1 #################
            
            level1_cur = down_feat1[i].unsqueeze(0)
            
            level1_feat = self.bottleneck1(torch.cat([level1_ref, level1_cur], dim=1))            
            
            level1_feat = self.bottleneck1_o(torch.cat([up_level2_feat, level1_feat], dim=1))  
            
            level1_feat = self.offset_fusion1(level1_feat)
            
            up_level1_feat =  F.interpolate(level1_feat, scale_factor = 2, mode = 'bilinear')*2
            
            level1_offset, level1_mask = self.offset_gen(self.offset_conv1(level1_feat))
            
            level1_aligned = self.deform1(level1_cur, level1_offset, level1_mask)
            
            level1_aligned = self.bottleneck1_a(torch.cat([level1_aligned, up_level2_aligned], dim=1))
            
            up_level1_aligned = F.interpolate(level1_aligned, scale_factor = 2, mode = 'bilinear')
            
            ############## Level 0 #################
            
            cur = burst_feat[i].unsqueeze(0)
            
            feat = self.bottleneck(torch.cat([ref, cur], dim=1))
            
            feat = self.bottleneck_o(torch.cat([up_level1_feat, feat], dim=1))
            
            feat = self.offset_fusion0(feat)
            
            offset, mask = self.offset_gen(self.offset_conv(feat))
            
            aligned_feat = self.deform(cur, offset, mask)
            
            aligned_feat = self.bottleneck_a(torch.cat([aligned_feat, up_level1_aligned], dim=1))
            
            ################ Cascading ###############  
            
            cas_feat = self.cas_bottleneck(torch.cat([ref, aligned_feat], dim=1))
            
            cas_feat = self.offset_fusion(cas_feat)
            
            cas_offset, cas_mask = self.offset_gen(self.cas_offset_conv(cas_feat))
            
            aligned_burst_feat.append(self.cas_deform(cas_feat, cas_offset, cas_mask))           
            
        aligned_burst_feat = torch.stack(aligned_burst_feat, dim=1)
        
        aligned_burst_feat = aligned_burst_feat[0]
        
        return aligned_burst_feat
    
    def forward(self, burst):
        
        burst = burst[0]
        
        burst_feat = self.conv1(burst)
        
        burst_feat = self.encoder(burst_feat)
        
        burst_feat = self.def_alignment(burst_feat)
        
        burst_feat = self.fusion(burst_feat)
        
        burst_feat = self.up1(burst_feat)
        
        burst_feat = self.up1(burst_feat)
        
        burst_feat = self.up1(burst_feat)
        
        burst_feat = self.conv3(burst_feat)
        
        return burst_feat

    def training_step(self, train_batch, batch_idx):
        
        x, y, flow_vectors, meta_info = train_batch
        
        pred = self.forward(x)
        
        pred = pred.clamp(0.0, 1.0)
        
        loss = self.train_loss(pred, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        
        x, y, flow_vectors, meta_info = val_batch
        
        pred = self.forward(x)
        
        pred = pred.clamp(0.0, 1.0)
        
        PSNR = self.valid_psnr(pred, y)
        
        return PSNR

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)        
        
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


def load_data(image_dir, burst_size):

    train_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='train')
    train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=burst_size, crop_sz=384)    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=args.NUM_WORKERS, pin_memory=True)

    test_zurich_raw2rgb = ZurichRAW2RGB(root=image_dir,  split='test')
    test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=burst_size, crop_sz=384)    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader


PATH = './Nancy_logs/trial_20/'

class Args:
	def __init__(self):
		self.arch = "base_model"
		self.image_dir = "./Zurich-RAW-to-DSLR-Dataset"
		self.test_image_dir = "./test_images"
		self.model_dir = PATH + "saved_model/"
		self.output_dir = PATH + "output"
		self.num_features = 64
		self.num_rg = 5
		self.num_rcab = 5
		self.reduction = 16
		self.batch_size = 1
		self.num_epochs = 500
		self.lr = 1e-4
		self.burst_size = 14
		self.NUM_WORKERS = 10
		
args = Args()


class Aligned_frame(nn.Module):
    def __init__(self, alignment_net):
        super().__init__()
        self.alignment_net = alignment_net

    def forward(self, pred, gt):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))
        return flow


class Network():

    def __init__(self, args):
        super().__init__()
        
        self.L1loss = nn.L1Loss()
        #self.loss_fn_alex = lpips.LPIPS(net='alex')
        self.p = 0.5
        
    def train(self):
               
        
        ################# Resume Training if pre-trained model exists 
        
        if os.path.exists(args.model_dir + 'model_best.pth'):
            model = Base_Model().cuda()                       
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)        
            checkpoint = torch.load(args.model_dir + 'model_best.pth')

            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except:
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print('Previously trained model weights state_dict loaded...')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
                print('Previously trained optimizer state_dict loaded...')
            prev_epochs = checkpoint['epoch']
            print('Trained model loss function loaded...')
            print(f"Previously trained for {prev_epochs} number of epochs...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
                print('Learning Rate :::', param_group['lr'])
            if prev_epochs<args.num_epochs:
                epochs = args.num_epochs - prev_epochs
                print(f"{epochs} more epochs needs to be complete...")
            else:
                extra_epochs = 10
                epochs = args.num_epochs + extra_epochs
                print(f"Resume trainining for {extra_epochs} more epochs...")      
        else:
            model = Base_Model().cuda()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)  
            epochs = args.num_epochs
            prev_epochs = 0
            if not os.path.exists(args.model_dir):            
                os.makedirs(args.model_dir)        
        
        
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])  
        print(params)
        
        ################## Count and assigns multiple of GPUs
            
        num_devices = torch.cuda.device_count()
        
        if num_devices > 1:
            print("Let's use", num_devices, "GPUs!")
            model = nn.DataParallel(model)
            batch_size = 4
        else:
            batch_size = 1

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs*1.5, eta_min=1e-6)
        
        
        ################## DATA Loaders ########################################
            
        train_zurich_raw2rgb = ZurichRAW2RGB(root=args.image_dir,  split='train')
        train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=384)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4*num_devices)
        
        test_zurich_raw2rgb = ZurichRAW2RGB(root=args.image_dir,  split='test')
        test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=384)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4*num_devices)        
        
        best_PSNR = 0
        best_epoch = 0
        best_iter = 0       
        
        
        ################# Model Training ####################################
        
        for epoch in range(epochs):
            
            epoch_start_time = time.time()
            epoch_losses = AverageMeter()
            epoch_loss = 0
            
            with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as _tqdm:
                
                _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))
                
                for i, data in enumerate(train_data_loader):
                    
                    
                    optimizer.zero_grad()
                    #for param in model.parameters():
                        #param.grad = None

                    burst, labels, flow_vectors, meta_info = data

                    burst = burst.cuda()
                    labels = labels.cuda()

                    preds = model(burst)
                    preds = preds.clamp(0.0, 1.0)

                    loss = self.L1loss(preds, labels)         
                    epoch_loss +=loss.item()
                    
                    epoch_losses.update(loss.item(), len(burst))
                    _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    _tqdm.update(len(burst))
                    
                    #print("Epoch: {}\tItr: {}\tLoss: {:.4f}".format(epoch, i, epoch_loss/(i+1)))

                    if i%2499==0 and i>0:
                        PSNR = self.validation(model, test_data_loader)
                        if PSNR > best_PSNR:
                            best_PSNR = PSNR
                            best_epoch = epoch
                            best_iter = i
                            torch.save({'epoch': epoch + prev_epochs + 1,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()},
                                       os.path.join(args.model_dir, "model_best.pth"))

                        print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (epoch, i, PSNR, best_epoch, best_iter, best_PSNR))

                    loss.backward()
                    optimizer.step()
            
            scheduler.step()  
                          
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss/len(train_dataset), scheduler.get_last_lr()[0]))
            print("------------------------------------------------------------------")                        
            
            torch.save({'epoch': epoch + prev_epochs + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},os.path.join(args.model_dir, f"model_epoch_{epoch}.pth"))
             
            torch.save({'epoch': epoch + prev_epochs + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},os.path.join(args.model_dir, "model_latest.pth"))
    
    
    #################### Validation #################################
    
    
    
    def validation(self, model, test_data_loader):       
        PSNR = []
        for i, data in enumerate(test_data_loader):
            burst, labels, flow_vectors, meta_info = data
            burst = burst.cuda()
            labels = labels.cuda()
                
            with torch.no_grad():
                output = model(burst)
                output = output.clamp(0.0, 1.0)                  
            PSNR.append(psnr_fn(output, labels).cpu().numpy())            
        mean_psnr = sum(PSNR) / len(PSNR)
        return mean_psnr


#######################################################################################################
################################### Model Testing on Overall testing set and compitition validation set
#######################################################################################################


class Testing():

    def __init__(self, args):
        super().__init__()
        
        self.loss_fn_alex = lpips.LPIPS(net='alex')
        
    def test(self):
        
        test_zurich_raw2rgb = ZurichRAW2RGB(root=args.image_dir,  split='test')
        test_dataset = SyntheticBurst(test_zurich_raw2rgb, burst_size=args.burst_size, crop_sz=384)
        test_data_loader = DataLoader(test_dataset, batch_size=1)
        
        postprocess_fn = SimplePostProcess(return_np=True)
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)        
        
        
        if os.path.exists(args.model_dir + 'model_best.pth'):
            model = Base_Model().cuda()            
            checkpoint = torch.load(args.model_dir + 'model_best.pth')
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except:
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                             
            model.eval()
        else:
            print("Error !!! Model not found.")
            
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])        
        
        score = 0
        PSNR = []
        SSIM = []
        LPIPS = []
        
        i=0
        a_file = open(PATH + "test.txt", "w")
        for d in test_data_loader:
            burst, labels, flow_vectors, meta_info = d
            meta_info = convert_dict(meta_info, burst.shape[0])
            
            burst = burst.cuda()
            labels = labels.cuda()
            
            burst_rgb = rgb2raw.demosaic(burst[0])            
            burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
            burst_rgb = F.interpolate(burst_rgb, scale_factor=4, mode='bilinear', align_corners=True)           
            
            with torch.no_grad():
                output = model(burst)
            output = output.clamp(0.0, 1.0)
            
            ### PSNR computation
            PSNR_temp = psnr_fn(output, labels).cpu().numpy()            
            PSNR.append(PSNR_temp)
            
            ### LPIPS computation
            var1 = 2*output-1
            var2 = 2*labels-1
            LPIPS_temp = self.loss_fn_alex(var1.cpu(), var2.cpu())
            LPIPS_temp = torch.squeeze(LPIPS_temp).detach().numpy()
            LPIPS.append(LPIPS_temp)
            
            ### SSIM computation            
            SSIM_temp = ssim(output*255, labels*255, data_range=255, size_average=True)
            SSIM_temp = torch.squeeze(SSIM_temp).cpu().detach().numpy()
            SSIM.append(SSIM_temp)
            
            eval_Par = 'Evaluation Measures for Burst {:d} ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f} \n'.format(i, PSNR_temp, SSIM_temp, LPIPS_temp)
            print(eval_Par)
            a_file.write(eval_Par)           
            
            
            input_burst = postprocess_fn.process(burst_rgb[0], meta_info[0])
            labels = postprocess_fn.process(labels[0].cpu(), meta_info[0])
            output = postprocess_fn.process(output[0].cpu(), meta_info[0])
            
            input_burst = cv2.cvtColor(input_burst, cv2.COLOR_RGB2BGR)   
            labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)            
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            output = np.concatenate((input_burst, output, labels), axis=1)
            
            cv2.imwrite('{}/{}.png'.format(args.output_dir, str(i)), output)
            i+=1
            
        Average_PSNR = sum(PSNR)/len(PSNR)
        Average_SSIM = sum(SSIM)/len(SSIM)
        Average_LPIPS = sum(LPIPS)/len(LPIPS)
        average_eval_par = '\nAverage Evaluation Measures ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f}\n'.format(Average_PSNR, Average_SSIM, Average_LPIPS)
        
        a_file.write(average_eval_par)
        a_file.write("Total number of parameters are ::: " + str(params))
        a_file.close()
                
        print(average_eval_par)
    
    def NTIRE_submission(self):
        
        device = 'cuda'
        
        dataset = SyntheticBurstVal('./syn_burst_val')
        out_dir = PATH + 'NTIRE_Submission'

        if os.path.exists(args.model_dir + 'model_latest.pth'):
            model = Base_Model(args).cuda()            
            checkpoint = torch.load(args.model_dir + 'model_latest.pth')
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except:
                state_dict = checkpoint["model_state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                             
            model.eval()
        else:
            print("Error !!! Model not found.")

        os.makedirs(out_dir, exist_ok=True)

        for idx in range(len(dataset)):
            burst, burst_name = dataset[idx]            

            burst = burst.cuda()
            burst = burst.unsqueeze(0)
            with torch.no_grad():
                net_pred = model(burst)

            # Normalize to 0  2^14 range and convert to numpy array
            net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

            # Save predictions as png
            cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)



model = Network(args)

model.train()

Testing(args).test()

Testing(args).NTIRE_submission()


# In[ ]:





# ### 

# In[ ]:




