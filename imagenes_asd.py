import cv2
import numpy as np

models = ['yolo','retinanet','faster','maskrcnn','trident']

dataset = 'modanet'
img_path = input('img path: ')
im_id = img_path.split('/')[-1].split('.')[0]
#im_id = 433734
if dataset =='df2':
    im_id = str(im_id).zfill(6)


img_gt = cv2.imread( 'gt_imgs/{}_{}.png'.format(dataset ,im_id))
bordersize=10
img_gt = cv2.copyMakeBorder(
                            img_gt,
                            top=bordersize,
                            bottom=bordersize,
                            left=bordersize,
                            right=bordersize,
                            borderType=cv2.BORDER_CONSTANT,
                            value=[255,255,255]
                            )
assert img_gt is not None
img_cat = img_gt
for i,model in enumerate(models):
    if dataset=='modanet':
        im_id = str(im_id).zfill(7)
    img_path = 'output/ouput-test_{}_{}_{}.jpg'.format(im_id,model,dataset)
    
    #print(img_path)
    img = cv2.imread(img_path)
    assert img is not None
    img = cv2.copyMakeBorder(
                            img,
                            top=bordersize,
                            bottom=bordersize,
                            left=bordersize,
                            right=bordersize,
                            borderType=cv2.BORDER_CONSTANT,
                            value=[255,255,255]
                            )
    
    img_cat = np.hstack((img_cat,img))
cv2.namedWindow('asd',cv2.WINDOW_NORMAL)
cv2.imshow('asd',img_cat)
cv2.imwrite('output/output_cat/{}_{}_cat.jpg'.format(dataset,im_id),img_cat)
cv2.waitKey(0)