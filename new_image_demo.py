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
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
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
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)



#


model = 'yolo'

if model == 'yolo':
    detectron = YOLOv3Predictor(params=yolo_params)
else:
    detectron = Predictor(model=model,dataset= dataset, CATEGORIES = classes)

#Faster RCNN / RetinaNet / Mask RCNN



while(True):
    path = input('img path: ')
    if not os.path.exists(path):
        print('Img does not exists..')
        continue
    img = cv2.imread(path)
    detections = detectron.get_detections(img)
    #detections = yolo.get_detections(img)
    #print(detections)

    

    #unique_labels = np.array(list(set([det[-1] for det in detections])))

    #n_cls_preds = len(unique_labels)
    #bbox_colors = colors[:n_cls_preds]

    
    if len(detections) != 0 :
        detections.sort(reverse=False ,key = lambda x:x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                
                #feat_vec =detectron.compute_features_from_bbox(img,[(x1, y1, x2, y2)])
                #feat_vec = detectron.extract_encoding_features(img)
                #print(feat_vec)
                #print(a.get_field('features')[0].shape)
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           

                
                #color = bbox_colors[np.where(unique_labels == cls_pred)[0]][0]
                color = colors[int(cls_pred)]
                
                color = tuple(c*255 for c in color)
                color = (.7*color[2],.7*color[1],.7*color[0])       
                    
                font = cv2.FONT_HERSHEY_SIMPLEX   
            
            
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                
                cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
                y1 = 0 if y1<0 else y1
                y1_rect = y1-25
                y1_text = y1-5

                if y1_rect<0:
                    y1_rect = y1+27
                    y1_text = y1+20
                cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                
                

                
    cv2.imshow('Detections',img)
    img_id = path.split('/')[-1].split('.')[0]
    cv2.imwrite('output/ouput-test_{}_{}_{}.jpg'.format(img_id,model,dataset),img)
    cv2.waitKey(0)