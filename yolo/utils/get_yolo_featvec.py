from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from colores import color_imagen

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import cv2
import sys

import numpy as np

import json 
import glob

from utils2 import *


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
#print(classes)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


model = load_model(params)



classes_id_names= {1:'short sleeve top', 
2:'long sleeve top', 
3:'short sleeve outwear', 
4:'long sleeve outwear', 
5:'vest', 
6:'sling', 
7:'shorts', 
8:'trousers', 
9:'skirt', 
10:'short sleeve dress', 
11:'long sleeve dress', 
12:'vest dress', 
13:'sling dress'}

classes_yolo_vec_dic= {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10:[], 11: [], 12: [], 13: []}

imgs_paths= {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10:[], 11: [], 12: [], 13: []}

annos_list = glob.glob('C:\DeepFashion2\\train\\annos\*.json')
j=0
for anno in annos_list:
    if j==40000:
        break
    with open(anno) as json_file:  
        data = json.load(json_file)
    
    if data['source'] == 'shop':
        
        file_name = anno.split('\\')[4].split('.')[0]        
        img_path= 'C:\DeepFashion2/train/image/' + file_name + '.jpg'
        print(img_path + ' '+ str(j))
        img = cv2.imread(img_path)
        x , pad , img_padded_size= cv_img_to_tensor(img)    
        x.to(device) 
        #print(pad , img_padded_size)
        for i in range(10):
            key = 'item{}'.format(i+1)
            if key in data:
                item_dic = data[key]
                cat_id = item_dic['category_id']
                bbox = item_dic['bounding_box']
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                
                with torch.no_grad():
                    input_img= Variable(x.type(Tensor))  
                    detections = model(input_img)
                bbox = orig_coords_to_yolo(pad,img_padded_size,bbox)
                yolo_feat_vec = model.get_yolo_feature_vec(bbox)
                
                classes_yolo_vec_dic[cat_id].append(yolo_feat_vec)
                imgs_paths[cat_id].append(img_path)
                #print(yolo_feat_vec.shape)
                #print(to_write)
                
            else:
                #print(classes_hog_vec_dic)
                break
        
        j+=1
    



for cls_id in classes_yolo_vec_dic:
    
    
    yolo_feat_vec = np.array(classes_yolo_vec_dic[cls_id])
    print(yolo_feat_vec.shape)
    
    
    vec_yolo_tuple = []
    for i in range(len(yolo_feat_vec)):
        vec_yolo_tuple.append((imgs_paths[cls_id][i] ,yolo_feat_vec[i]))

  
    np.save('yolo_descriptors/'+ classes_id_names[cls_id],vec_yolo_tuple) 
   