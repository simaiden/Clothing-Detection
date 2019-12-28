
from yolo.utils.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *
#from yolo.utils.utils2 import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys
import argparse

class YOLOv3Predictor(object):
    def __init__(
        self,
        params,
    ):
        self.params = params
        self.model = self.load_model()
        self.orig_detections = None
        print('Model loaded successfully from {}.'.format(params["weights_path"]))

    def get_detections(self,img):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        x , _ ,_= self.cv_img_to_tensor(img)
    
        x.to(self.params['device']) 
        input_img= Variable(x.type(Tensor))  
        detections = self.model(input_img)
        detections = non_max_suppression(detections, self.params['conf_thres'], self.params['nms_thres'])
        
        if detections[0] is not None:
            self.orig_detections = detections[0].clone()
            #detections_org = detections[0].clone()
            detections = rescale_boxes(detections[0], self.params['img_size'], img.shape[:2])
            #unique_labels = detections[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)
            detections = detections.tolist()
            for det in detections:
                del det[5]  #delete 'conf' item, we won't use it
            
            return detections 
        return [] 

    
    def load_classes(self):
        return load_classes(self.params)


    
    def load_model(self):
    # Set up model
        model = Darknet(self.params['model_def'], img_size=self.params['img_size']).to(self.params['device'])
        model.load_darknet_weights(self.params['weights_path'])
        model.eval()  # Set in evaluation mode
        return model

    def cv_img_to_tensor(self,img, dim = (416, 416)):
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

    def orig_coords_to_yolo(self,pad,img_padded_size,  coords):
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

