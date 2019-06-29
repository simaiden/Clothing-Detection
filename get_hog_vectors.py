
from __future__ import division

import numpy as np
import cv2
import json 
import glob

from joblib import dump, load


from sklearn.decomposition import PCA

def get_hog(image,coord=(0,0,0,0),size=(80,150)):
    hog = cv2.HOGDescriptor()
    x1,y1,x2,y2 = coord[0], coord[1], coord[2], coord[3]

    #Si w,h venian en cero no hay que recortar la imagen
    # if (w==0):
    #     w = image.shape[1]
    # if(h==0):
    #     h = image.shape[0]

    #recorte
    image = image[y1:y2, x1:x2]
    #resize
    image = cv2.resize(image,size)
    hog_im = hog.compute(image)

    return hog_im






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

classes_hog_vec_dic= {1: [], 
2: [], 
3: [], 
4: [], 
5: [], 
6: [], 
7: [], 
8: [], 
9: [], 
10:[], 
11: [], 
12: [], 
13: []}

imgs_paths= {1: [], 
2: [], 
3: [], 
4: [], 
5: [], 
6: [], 
7: [], 
8: [], 
9: [], 
10:[], 
11: [], 
12: [], 
13: []}

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
        #print(img.shape)
        for i in range(10):
            key = 'item{}'.format(i+1)
            if key in data:
                item_dic = data[key]
                cat_id = item_dic['category_id']
                bbox = item_dic['bounding_box']
                bbox = tuple(bb for bb in bbox)
                hog_vec = get_hog(img,bbox)[:,0]
                
                classes_hog_vec_dic[cat_id].append(hog_vec)
                imgs_paths[cat_id].append(img_path)
                #print(hog_vec, img_path)
                #print(to_write)
                
            else:
                #print(classes_hog_vec_dic)
                break
        
        j+=1
    


for cls_id in classes_hog_vec_dic:
    
    
    hog_np = np.array(classes_hog_vec_dic[cls_id])
    #print(hog_np.shape)
    
    pca = PCA(n_components=32)
    pca.fit(hog_np)  
    hog_pca = pca.transform(hog_np)   
    #print(hog_pca.shape)
    print('Reduce dimension from {} to {}'.format(hog_np.shape,hog_pca.shape))
    #classes_hog_vec_dic[cls_id][1] = hog_np
    vec_hog_tuple = []
    for i in range(len(hog_np)):
        vec_hog_tuple.append((imgs_paths[cls_id][i] ,hog_pca[i]))

  
    np.save('hog_descriptors/'+ classes_id_names[cls_id],vec_hog_tuple) 
    #np.save('paths_desc_imgs/'+ classes_id_names[cls_id],imgs_paths[cat_id])
    dump(pca, 'pca_objs/'+ classes_id_names[cls_id] + '.joblib')    