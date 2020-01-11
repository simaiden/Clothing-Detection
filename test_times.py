import numpy as np
import matplotlib.pyplot as plt


models = ['yolo','retinanet','faster','maskrcnn','trident']
mean = []
stds=[]
for model in models:
    tiempos = np.load('tiempos/times_{}.npy'.format(model))
    print(model)
    mean.append(np.mean(tiempos))
    stds.append(np.std(tiempos))
fig, ax = plt.subplots()  
plt.bar(list(range(1,6)),mean)

#plt.errorbar(list(range(1,6)),mean,yerr=stds, linestyle='None',capsize=27,elinewidth=3,ecolor='m')
plt.xticks(list(range(1,6)),['YOLOv3','RetinaNet','Faster R-CNN','Mask R-CNN','TridentNet'])
for i, v in enumerate(mean):
    plt.text(list(range(1,6))[i] - 0.25, v + 0.005, str(round(v,3)))
plt.ylabel('Segundos')
plt.xlabel('Modelo')
plt.title('Tiempos promedio de inferencia modelos entrenados')
plt.show()
#plt.savefig('tiempos.eps')