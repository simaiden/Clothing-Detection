import numpy as np  
from joblib import dump, load

pca_objs = {}
hog_descriptors = {}
i=0
classes = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
for clss in classes:
    pca_objs[i] =  load('pca_objs/{}{}'.format(clss,'.joblib'))
    hog_descriptors[i] = np.load('hog_descriptors/{}{}'.format(clss,'.npy'), allow_pickle = True)
    i+=1

print('Descriptors loaded')
print(hog_descriptors[0][:,1][0])
np.linalg.norm(hog_descriptors[1][:,1] - np.ones(32), axis=0)