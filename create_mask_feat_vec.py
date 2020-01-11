
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
classes = load_classes("yolo/df2cfg/df2.names")

detectron = Predictor(model='maskrcnn',dataset= 'df2', CATEGORIES = classes)


feat_vecs = []
feat_vecs_retrieval_test = []
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
            feat_vec =detectron.compute_features_from_bbox(img,[bbox])
            
            feat_vecs.append((image_id,feat_vec))


        detections = detectron.get_detections(img)
        if len(detections) != 0 :
            #detections.sort(reverse=False ,key = lambda x:x[4])
            for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                    
                    
                   
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    feat_vec_test =detectron.compute_features_from_bbox(img,[(x1, y1, x2, y2)])
                    #print(feat_vec_test.shape)
                    feat_vecs_retrieval_test.append((image_id,feat_vec_test,(x1, y1, x2, y2)))
        
        
np.save( 'mask_df2_shop_descriptors', feat_vecs) 
np.save( 'mask_df2_shop_descriptors_retrieval', feat_vecs_retrieval_test)