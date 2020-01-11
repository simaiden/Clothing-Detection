

import torch
import os
import cv2
import sys
import json
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
from shutil import copy

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


yolo_params = yolo_df2_params

classes = load_classes(yolo_params["class_path"])
detector = YOLOv3Predictor(params=yolo_params)



feat_vecs = []

annos_list = glob.glob('Deepfashion2Val/annos/*.json')
j=0
for anno in tqdm(annos_list):
    with open(anno) as json_file:  
        data = json.load(json_file)
    #for item in data.values():
        #print (item)
    image_id = anno.split('/')[-1].split('.')[0]
    path = 'Deepfashion2Val/image/{}.jpg'.format(image_id)
    if data['source'] == 'shop':

        del data['pair_id']
        del data['source']

        
            
            
        img  = cv2.imread(path)
        #detections = detector.get_detections(img,original=True)

        _ , pad , img_padded_size= detector.cv_img_to_tensor(img)    


        detections = detector.get_detections(img)

        
        #print(yolo_feat_vec)
        #print(yolo_feat_vec.shape)
        

        if len(detections) >0:
            for x1, y1, x2, y2, _, _, cls_pred in detector.orig_detections:
                
                bbox = (x1, y1, x2, y2)
                yolo_fv = detector.model.get_yolo_feature_vec(bbox)
                #print(yolo_fv.shape)
                #closest_img = closest_distances(yolo_fv,shop_descriptors)
                #closest_img = shop_imgs_ids[closest_img]
                
                #closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))
                feat_vecs.append((image_id,yolo_fv,bbox))
    else:
        a=1
        #copy(path,'Deepfashion2Val/users_imgs/{}.jpg'.format(image_id))

np.save( 'yolo_df2_shop_descriptors_retrieval_test_4', feat_vecs) 
#np.save( 'yolo_df2_shop_descriptors_new4', feat_vecs_test) 