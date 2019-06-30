import numpy as np
import cv2
from sklearn.cluster import KMeans



def colores_dominantes(img, n_clusters):

    #Las imagenes con opencv predeterminadamente
    #se abren en bgr. Las pasamos a lab a lo vio
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #La pasamos a lista a lo vio
    img = img.reshape((img.shape[0] * img.shape[1], 3))


    # Usamos kmeans a lo vio con el n_clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(img)

    #Los clusters
    clusters = kmeans.cluster_centers_

    return clusters




#Recibe una tupla (l,a,b), transforma a hsv y determina el nombre del color
def check_color_hsv(lab):
    l,a,b = lab[0],lab[1],lab[2]

    #Creamos una imagen de 1 pixel de 3 canales
    array = np.array([l,a,b],dtype=np.uint8)
    array = np.reshape(array,(1,1,3))

    #Transformamos a espacio hsv
    bgr = cv2.cvtColor(array,cv2.COLOR_LAB2BGR) #Q xuxa no hay lab2hsv
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)

    #h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]

    h, s, v = hsv[0, 0, 0], hsv[0, 0, 1], hsv[0, 0, 2]
    print(h,s,v)
    if s<25:
        if v<85:
            return 'negra'
        elif v<170:
            return 'gris'
        return 'blanca'

    if h<=5 or h>175:
        return 'roja'

    if h>5 and h<=15:
        return 'naranja'

    if h>15 and h<=40:
        return 'amarilla'

    if h>40 and h<=70:
        return 'verde'

    if h>70 and h<=105:
        return 'celeste'

    if h>105 and h<=130:
        return 'azul'

    if h>130 and h<=175:
        return 'morada'


def color_imagen(imagen,n_clusters=3):
    color_lab_dom = colores_dominantes(imagen,n_clusters)
    l,a,b = color_lab_dom[0][0], color_lab_dom[0][1], color_lab_dom[0][2]

    nombre_color = check_color_hsv((l,a,b))

    return nombre_color

naruto = cv2.imread("devil.jpg")
naruto_name = color_imagen(naruto)
print(naruto_name)