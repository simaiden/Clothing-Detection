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
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.25,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'df2'

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




#Faster RCNN / RetinaNet / Mask RCNN
model = 'maskrcnn'
detectron = Predictor(model=model,dataset= dataset, CATEGORIES = classes)

#YOLO


yolo_descriptors = np.load('mask_df2_shop_descriptors.npy', allow_pickle = True)
shop_descriptors = np.array([vec for vec in yolo_descriptors[:,1]])
shop_imgs_ids = np.array([id for id in yolo_descriptors[:,0]])
#print(yolo_descriptors)

#etectron = YOLOv3Predictor(params=yolo_params)


while(True):
    path = input('img path: ')
    #path = 'tests/000081.jpg'
    if not os.path.exists(path):
        print('Img does not exists..')
        continue
    img = cv2.imread(path)
    detections = detectron.get_detections(img)
    
    closest_img_paths = []
    if len(detections) >0:
        # for x1, y1, x2, y2, _, _, cls_pred in detectron.orig_detections:
                
                
        #         yolo_fv = detectron.model.get_yolo_feature_vec( (x1, y1, x2, y2))
        #         #print(yolo_fv)
        #         closest_img = closest_distances(yolo_fv,shop_descriptors)
        #         closest_img = shop_imgs_ids[closest_img]
        #         #print(closest_img)
        #         closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))

        
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                color = colors[int(cls_pred)]
                
                color = tuple(c*255 for c in color)
                color = (.7*color[2],.7*color[1],.7*color[0])         
                    
                font = cv2.FONT_HERSHEY_SIMPLEX   
            
                #print(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text =  "%s " % (classes[int(cls_pred)])

                cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)

                y1_rect = y1-25
                y1_text = y1-5

                if y1_rect<0:
                    y1_rect = y1+27
                    y1_text = y1+20
                cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                    

                
                feat_vec =detectron.compute_features_from_bbox(img,[(x1, y1, x2, y2)])
                closest_img = closest_distances(feat_vec,shop_descriptors,norm='cosine')
                closest_img = shop_imgs_ids[closest_img]
                #print(closest_img)
                closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))
        if(len(closest_img_paths)>=1):
                print(closest_img_paths)
                for im_path in closest_img_paths:
                    
                    path = 'Deepfashion2Val/image/{}.jpg'.format(im_path[0]) 
                    img_retrieval = cv2.imread(path)
                    print(path  )
                    cv2.namedWindow(im_path[1],cv2.WINDOW_NORMAL)
                    cv2.imshow(im_path[1],img_retrieval)
    else:
        print('No detections')
    cv2.namedWindow('Detections',cv2.WINDOW_NORMAL)
    cv2.imshow('Detections',img)
    
    cv2.waitKey(0)
    for im_path in closest_img_paths:
            cv2.destroyWindow(im_path[1])
    