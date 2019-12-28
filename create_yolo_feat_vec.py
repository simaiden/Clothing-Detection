
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

        for item in data:
            anno = data[item]
            
            
            bbox = anno['bounding_box']
            bbox = tuple(bb for bb in bbox)
            
            
            img  = cv2.imread(path)
            #detections = detector.get_detections(img,original=True)

            _ , pad , img_padded_size= detector.cv_img_to_tensor(img)    


            _ = detector.get_detections(img)

            bbox = detector.orig_coords_to_yolo(pad,img_padded_size,bbox)
            yolo_feat_vec = detector.model.get_yolo_feature_vec(bbox)
            #print(yolo_feat_vec)
            #print(yolo_feat_vec.shape)
            feat_vecs.append((image_id,yolo_feat_vec))
    else:
        copy(path,'Deepfashion2Val/users_imgs/{}.jpg'.format(image_id))

np.save( 'yolo_df2_shop_descriptors_new', feat_vecs) 