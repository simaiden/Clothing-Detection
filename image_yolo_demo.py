

from yolo.utils.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *
from yolo.utils.utils2 import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys
import argparse

#parser = argparse.ArgumentParser()

#parser.add_argument('--model_cfg', type=string, default="yolo/df2cfg/yolov3-df2.cfg", help='CFG file with YOLOv3 config')
#parser.add_argument('--weights_path', type=string, default="yolo/df2cfg/yolov3-df2.cfg", help='CFG file with YOLOv3 config')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device
}

# params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
# "weights_path" : "yolo/weights/yolov3-modanet_17000.weights",
# "class_path":"yolo/modanetcfg/modanet.names",
# "conf_thres" : 0.1,
# "nms_thres" :0.4,
# "img_size" : 416,
# "device" : device
# }



classes = load_classes(params['class_path']) 
#print(classes)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
cmap = plt.get_cmap("tab20")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)

model = load_model(params)

print('Model loaded successfully from {}.'.format(params["weights_path"]))


#cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
while(True):
    img_path = input('Insert path to image: ')
    if img_path=='exit':
        break

    img = cv2.imread(img_path)
    if img is None:
        print('Image not found...')
        continue
    
    img2= img.copy()     
    x , _ ,_= cv_img_to_tensor(img)
    
    x.to(device)   

            # Get detections
    #with torch.no_grad():
    input_img= Variable(x.type(Tensor))  
    detections = model(input_img)
    detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])

    if detections[0] is not None:

        detections_org = detections[0].clone()
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        print(unique_labels)
        n_cls_preds = len(unique_labels)
        print(n_cls_preds)
        #bbox_colors = random.sample(colors, n_cls_preds , seed)
        bbox_colors = colors[:n_cls_preds]
        print(bbox_colors)
        #print(detections)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            print(color)
            color = tuple(c*255 for c in color)
            color = (color[2],color[1],color[0])
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf.item())

            cv2.rectangle(img2,(x1,y1) , (x2,y2) , color,3)
            cv2.rectangle(img2,(x1-2,y1-25) , (x1 + 8.5*len(text),y1) , color,-1)
            cv2.putText(img2,text,(x1,y1-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)

            #x_f = model.get_yolo_feature_vec( (x1, y1, x2, y2))
            #print(x_f)
        
    else:
        print('No detections...') 

    cv2.imshow('Detections',img2)
        
    cv2.waitKey(0)
