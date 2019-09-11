from __future__ import division

from utils.models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2

from utils import load_model, cv_img_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {   "model_def" : "df2cfg/yolov3-df2.cfg",
"weights_path" : "weights/yolov3-df2_15000.weights",
"class_path":"df2cfg/df2.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device
}


classes = load_classes(params['class_path']) 
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
cmap = plt.get_cmap("tab20b")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)

model = load_model(params)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
while(True):
    #img = cv2.imread('weon.jpg')
    _, frame = cap.read()
    img = frame.copy()
    x , _,_ = cv_img_to_tensor(img)
    x.to(device)   
    
            # Get detections
    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])
   
    if detections[0] is not None:

            # Rescale boxes to original image
                
            detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds , seed)
            bbox_colors = colors[:n_cls_preds]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color = tuple(c*255 for c in color)
                color = (color[2],color[1],color[0])
                cv2.rectangle(frame,(x1,y1) , (x2,y2) , color,3)
                #print(int(cls_pred))
                font = cv2.FONT_HERSHEY_SIMPLEX
                text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf.item())
                cv2.rectangle(frame,(x1-2,y1-25) , (x1 + 8.5*len(text),y1) , color,-1)
                cv2.putText(frame,text,(x1,y1-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
             
    cv2.imshow('Detections',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
                
        