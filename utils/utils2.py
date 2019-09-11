from __future__ import division

from .models import *
from .utils import *
from .datasets import *


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys


def load_model(params):
# Set up model
    model = Darknet(params['model_def'], img_size=params['img_size']).to(params['device'])
    model.load_darknet_weights(params['weights_path'])
    model.eval()  # Set in evaluation mode
    return model

def cv_img_to_tensor(img, dim = (416, 416)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1))
    x = x.unsqueeze(0).float()     
    _, _, h, w = x.size()
    ih, iw = dim[0],dim[1]
    dim_diff = np.abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
    x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
    img_padded_size = x.shape[2]
    x = F.interpolate(x, size=(ih, iw), mode='bilinear',align_corners=False) # x = (1, 3, 416, 416)
    return x, pad,img_padded_size

def orig_coords_to_yolo(pad,img_padded_size,  coords):
    ratio = 416/img_padded_size
    (x1,y1,x2,y2)  = coords
    x1 += pad[0]
    y1 += pad[2]
    x2 += pad[1]
    y2 += pad[3]    
       
    x1 *= ratio
    y1 *= ratio
    x2 *= ratio
    y2 *= ratio        
    
    return (int(x1),int(y1),int(x2),int(y2))       

