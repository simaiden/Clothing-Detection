from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from colores import color_imagen

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys

import numpy as np

import json 
import glob

from utils2 import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {   "model_def" : "df2cfg/yolov3-df2.cfg",
"weights_path" : "weights/yolov3-df2_8000.weights",
"class_path":"df2cfg/df2.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device
}


classes = load_classes(params['class_path']) 
#print(classes)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


#model = load_model(params)

img = cv2.imread('C:\\DeepFashion2\\train\image\\000021.jpg')
(x1, y1, x2, y2) = (113, 314, 287, 504)

x , pad ,img_padded_size= cv_img_to_tensor(img)
asd = x.cpu().numpy()[0].transpose(1, 2, 0)
asd = cv2.cvtColor(asd, cv2.COLOR_RGB2BGR)
print(asd.shape)
print(img.shape)
(x11, y11, x22, y22) = orig_coords_to_yolo(pad,img_padded_size,(x1, y1, x2, y2))

cv2.rectangle(img,(x1,y1) , (x2,y2) , (255,0,0),3)
cv2.rectangle(asd,(x11,y11) , (x22,y22) , (0,0,255),3)

cv2.imshow('asd',img)
cv2.imshow('asd2',asd)

cv2.waitKey(0)