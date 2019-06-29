from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2

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
    x = F.upsample(x, size=(ih, iw), mode='bilinear') # x = (1, 3, 416, 416)
    return x

def getYoloDetections(img,model,params):
    x = cv_img_to_tensor(img)
    x.to(params['device'])   

    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])

    if detections[0] is not None:
        # Rescale boxes to original image        
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
    return detections

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
print(classes)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
cmap = plt.get_cmap("Set1")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)

model = load_model(params)

img = cv2.imread('weon.jpg')
img2= img.copy()


detections = getYoloDetections(img,model,params)
if detections is not None:

    detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    #bbox_colors = random.sample(colors, n_cls_preds , seed)
    bbox_colors = colors[:n_cls_preds]
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        color = tuple(c*255 for c in color)
        
        cv2.rectangle(img2,(x1,y1) , (x2,y2) , color,2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2,classes[int(cls_pred)] + ' '+str(cls_conf.item()),(x1,y1-15), font, 0.5,color,2,cv2.LINE_AA)
    cv2.imshow('asdasd',img2)
    cv2.waitKey(0)