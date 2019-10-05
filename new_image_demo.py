import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


#YOLO PARAMS
yolo_df2_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_15000.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'modanet'


if dataset == 'df2': #deepfashion2
    yolo_params = yolo_df2_params

if dataset == 'modanet':
    yolo_params = yolo_modanet_params


#Classes
classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("tab20")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 20)])
np.random.shuffle(colors)



#Faster RCNN / RetinaNet
#model = 'faster'
detectron = Predictor(model='retinanet',dataset= dataset, CATEGORIES = classes)

#YOLO
#yolo = YOLOv3Predictor(params=yolo_params)




img = cv2.imread('tests/tipo.jpg')
detections = detectron.get_detections(img)
#detections = yolo.get_detections(img)
#print(detections)



unique_labels = np.array(list(set([det[-1] for det in detections])))

n_cls_preds = len(unique_labels)
bbox_colors = colors[:n_cls_preds]


if len(detections) != 0 :
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           

            
            color = bbox_colors[np.where(unique_labels == cls_pred)[0]][0]
            
            
            color = tuple(c*255 for c in color)
            color = (color[2],color[1],color[0])            
                   
            font = cv2.FONT_HERSHEY_SIMPLEX   
        
        
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)

            cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
            cv2.rectangle(img,(x1-2,y1-25) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img,text,(x1,y1-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)

            
cv2.imshow('Detections',img)
cv2.waitKey(0)