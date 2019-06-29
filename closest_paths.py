import numpy as np  
import cv2
from joblib import dump, load

def todas_distancias(vector,vectores):
    # Es una matriz con el vector 'vector' repetido en todas las filas
    d = np.ones((vectores.shape[0],32))
    d = d*vector

    #distancias es un vector con la misma cantidad de filas que 'vectores'
    #donde cada componente es la distancia entre vector y vectores[i]
    distancias = np.linalg.norm(d-vectores,axis=1)
    return distancias




#clase es un número, descriptors es hog_descriptors
#n es el numero de imagenes mas cercanas que queremos
def get_closest_paths(descriptors,clase,n=3):
    # Descriptores
    desc = descriptors[clase][:, 1]

    # Paths
    paths_descriptors = descriptors[clase][:, 0]

    # Los arreglamos
    wea_buena = np.zeros((desc.shape[0], desc[0].shape[0]))
    for i in range(desc.shape[0]):
        wea_buena[i] = desc[i]


    #final_paths = np.zeros((desc.shape[0],n))
    final_paths = []
    for i in range(desc.shape[0]):
        paths_local = []
        # Sacamos todas las distancias
        distancias = todas_distancias(desc[i], wea_buena)

        # Arreglamos
        sorted = np.sort(distancias)
        for j in range(n):
            #Ignoramos el indice 0 porque será el del mismo vector que evaluamos
            index = np.where(distancias == sorted[j+1])[0]
            close_path = paths_descriptors[index][0]
            paths_local.append(close_path)
            print(i,j,close_path)
            #final_paths[i][j] = close_path
        print(paths_local)
        final_paths.append(paths_local)
    return final_paths

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

def n_paths_cercanos(vector,descriptores,clase,n=3):
    # Descriptores
    desc = descriptores[clase][:, 1]
    #print(desc.shape)
    # Paths
    paths_descriptors = descriptores[clase][:, 0]
    #prnt(len(descriptores[clase]))
    #print(descriptores[clase].shape)
    # Los arreglamos
    wea_buena = np.zeros((desc.shape[0],32))
    #print(wea_buena.shape)
    for i in range(desc.shape[0]):
        wea_buena[i] = desc[i]

    # Sacamos todas las distancias
    distancias = todas_distancias(vector, wea_buena)

    # Arreglamos
    sorted = np.sort(distancias)

    closest_paths = []
    for i in range(n):
        index = np.where(distancias == sorted[i+1])[0] #Ignoramos indice 0
        close_path = paths_descriptors[index]
        closest_paths.append(close_path[0])
        
        
    return closest_paths




if __name__ == "__main__":

    pca_objs = {}
    hog_descriptors = {}
    i=0
    classes = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
    for clss in classes:
        pca_objs[i] =  load('pca_objs/{}{}'.format(clss,'.joblib'))
        hog_descriptors[i] = np.load('hog_descriptors/{}{}'.format(clss,'.npy'), allow_pickle = True)
        i+=1


    pca = pca_objs[int(0)]
    hog_detection = np.ones(34020) #get_hog(frame,(x1, y1, x2, y2))
    hog_pca = pca.transform(hog_detection.reshape(1, -1))
    paths = n_paths_cercanos(hog_pca, hog_descriptors,0,n=1)
    print(paths)