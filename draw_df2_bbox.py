from yolo.utils.utils import load_classes
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import json 
#classes = load_classes('yolo/df2cfg/df2.names')

annotations = 'Deepfashion2Val/annos'

#img_id = 21167
#img_id = str(img_id).zfill(6)




cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
font = cv2.FONT_HERSHEY_SIMPLEX  


#img_path = '/home/simon/Memoria/Clothes-Detection/Deepfashion2Val/image/{}.jpg'.format(img_id)
img_path = input('img path: ')
img_id = img_path.split('/')[-1].split('.')[0]
annotations_path = annotations + '/{}.json'.format(img_id)
print(img_path)
img = cv2.imread(img_path)



with open(annotations_path) as f:
    info = json.load(f)
#Colors

del info['source']
del info['pair_id']

for item in info:
    bbox = info[item]['bounding_box']
    cls_name = info[item]['category_name']
    cls_id = info[item]['category_id'] - 1
   
    x1, y1, x2, y2 = bbox[0] , bbox[1] , bbox[2] , bbox[3]  

    color = colors[cls_id]
                
    color = tuple(c*255 for c in color) 
    color = (.7*color[2],.7*color[1],.7*color[0])       
    cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
    text =  "%s" % cls_name
    
    y1_rect = y1-25
    y1_text = y1-5

    if y1_rect<0:
        y1_rect = y1+27
        y1_text = y1+20
    cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
    cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)

cv2.imshow('asd',img)
cv2.imwrite('gt_imgs/df2_{}.png'.format(img_id) , img)
cv2.waitKey(0)




