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
from closest_paths import n_paths_cercanos, get_hog
from joblib import dump, load

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



yolo_descriptors = {}
i=0
for i,clss in enumerate(classes):
    #pca_objs[i] =  load('pca_objs/{}{}'.format(clss,'.joblib'))
    yolo_descriptors[i] = np.load('yolo_descriptors/{}{}'.format(clss,'.npy'), allow_pickle = True)

print('Descriptors loaded')


cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Retrieval', cv2.WINDOW_NORMAL)

while(True):
    #img = cv2.imread('weon.jpg')
    _, frame = cap.read()
    img = frame.copy()
    x , _ ,_ = cv_img_to_tensor(img)
    x.to(device)   
    
            # Get detections
    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])
    #closest_img = None
    if detections[0] is not None:

        # Rescale boxes to original image
        detections_org = detections[0].clone()
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        #bbox_colors = random.sample(colors, n_cls_preds , seed)
        bbox_colors = colors[:n_cls_preds]
        closest_img_paths = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = tuple(c*255 for c in color)
            color = (color[2],color[1],color[0])
            cv2.rectangle(frame,(x1,y1) , (x2,y2) , color,3)
            #print(int(cls_pred))
            font = cv2.FONT_HERSHEY_SIMPLEX
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf.item())
            cv2.rectangle(frame,(x1-2,y1-25) , (x1 + 8.5*len(text),y1) , color,-1)
            cv2.putText(frame,text,(x1,y1-5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            #try:
            #    pca = pca_objs[int(cls_pred)]
            #    hog_detection = get_hog(frame,(int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())))
            #    hog_pca = pca.transform(hog_detection.reshape(1, -1))
            #    closest_img = n_paths_cercanos(hog_pca,hog_descriptors,int(cls_pred),n=1)
            #    closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))
            #except:
            #    continue
        for x1, y1, x2, y2, _, _, cls_pred in detections_org:
            
            yolo_fv = model.get_yolo_feature_vec( (x1, y1, x2, y2))
            #print(yolo_fv.shape)
            closest_img = n_paths_cercanos(yolo_fv,yolo_descriptors,int(cls_pred),n=1)
            closest_img_paths.append((closest_img[0], classes[int(cls_pred)]))

             
    
        if(len(closest_img_paths)>=1):
            for im_path in closest_img_paths:
                    img_retrieval = cv2.imread(im_path[0])
                    cv2.imshow(im_path[1],img_retrieval)
    cv2.imshow('Detections',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
                
        