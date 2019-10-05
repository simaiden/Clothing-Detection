from __future__ import division

import sys
#sys.path.append('utils')

from yolo.utils.models import *
from yolo.utils.utils import *
from yolo.utils.utils2 import *


import matplotlib.pyplot as plt
import cv2
import json
import glob
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
# "conf_thres" : 0,
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

print('Model loaded successfully.')

results = []

subset = 'validation'
path = '/media/simon/5AF29F83F29F61D5/DeepFashion2/' + '{}/image/'.format(subset)
num_images = len(glob.glob(path + '/*.jpg'))

for num in tqdm(range(1,num_images+1)):
    image_name = path  + str(num).zfill(6)+'.jpg'

    img = cv2.imread(image_name)
    x , _ ,_= cv_img_to_tensor(img)

    with torch.no_grad():
        input_img= Variable(x.type(Tensor))  
        detections = model(input_img)
        detections = non_max_suppression(detections, params['conf_thres'], params['nms_thres'])

    if detections[0] is not None:
        detections = rescale_boxes(detections[0], params['img_size'], img.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            dic = {
                "image_id": num,
                "category_id": int(cls_pred.item()) + 1, 
                "bbox": [round(x1.item(),2), round(y1.item(),2), round(x2.item()-x1.item(),2), round(y2.item()-y1.item(),2)], 
                "score": cls_conf.item()
                 }
            results.append(dic)


json_name = 'deepfashion2_{}_eval.json'.format(subset)
with open(json_name, 'w') as f:
  json.dump(results, f)