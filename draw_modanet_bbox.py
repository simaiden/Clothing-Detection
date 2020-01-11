from yolo.utils.utils import load_classes
import cv2
import numpy as np 
import matplotlib.pyplot as plt
classes = load_classes('yolo/modanetcfg/modanet.names')



img_path = input('img_path: ')
img_id = img_path.split('/')[-1].split('.')[0]
modanet_annotations = '/home/simon/Memoria/modanet/annotations'

#img_id = 432256
#943235
#681883
annotations_path = modanet_annotations + '/{}.txt'.format(int(img_id))


with open(annotations_path) as f:
    lines = f.readlines()

#Colors
cmap = plt.get_cmap("rainbow") 
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
font = cv2.FONT_HERSHEY_SIMPLEX  
#img_path = '/home/simon/Memoria/darknet_modanet/modanet/img/val/{}.jpg'.format(str(img_id).zfill(7))
img = cv2.imread(img_path)
print(img_path)
#print('home/simon/Memoria/darknet_modanet/modanet/img/val/{}.jpg'.format(str(img_id).zfill(7)))
for line in lines:
    info = line.rstrip().split()
    info = [int(a) for a in info]
    cls,x, y, w, h = info[0] , info[1] , info[2] , info[3]  , info[4]
    #print(x,y,h,w)
    color = colors[int(cls)]
                
    color = tuple(c*255 for c in color)
    color = (.7*color[2],.7*color[1],.7*color[0])       
    cv2.rectangle(img,(x,y) , (x +w,y+h) , color,3)
    text =  "%s" % classes[int(cls)]
    y1_rect = y-25
    y1_text = y-5

    if y1_rect<0:
        y1_rect = y+27
        y1_text = y+20
    cv2.rectangle(img,(x-2,y1_rect) , (x + int(8.5*len(text)),y) , color,-1)
    cv2.putText(img,text,(x,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)

cv2.imshow('asd',img)
cv2.imwrite('gt_imgs/modanet_{}.png'.format(img_id) , img)
cv2.waitKey(0)




