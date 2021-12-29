import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from pypcd import pypcd

"""
Dataset erzeugen mit der Form [x,y,z,CheckIfGate]

Was wir noch machen können(müssen).
-Mehre Datei erzuegen, zB Distanz, Winkel(Azimut, Neigung zu einem Punkt), oder Rotation des Schiffs 
    Durch Rotation des Schiffs kann man besser dem Model beibringen, was er lernen muss. Wenn die Richtung immer gleich 
    ist, kann er lernen, dass die Punkte Vorne die Schleuse sind. Aber was er lernen soll ist, ob die Fläche von anderen 
    Fläche umgeben ist oder von einigen geraden Lienien (Ecken) umgeben ist.
-Die Labelierung mit Least Square Methode
"""
#pc = pypcd.PointCloud.from_path('/Users/chitoshitamaoki/Desktop/3.Semester/AML/ML/RosbagCut/1632495999.953719854.pcd')
for k in range(25):
    path="/Users/chitoshitamaoki/Desktop/3.Semester/AML/ML/RosbagCut/data{}.pcd"####Hier zu ändern
    pcd = o3d.io.read_point_cloud(path.format(k+1))
    out_arr = np.asarray(pcd.points)
    #print("Rohdatei",out_arr)

    distance = np.zeros(out_arr.shape[0])
    for i in range(out_arr.shape[0]):
        distance[i]=np.linalg.norm(out_arr[ i ,:])

    #print(distance)
    CheckIfGate=np.zeros((out_arr.shape[0],1))
    for i in range(out_arr.shape[0]):
        if distance[i]>11.2:#should be changed #####Hier zu verbessern!!!!!
            CheckIfGate[i]=1
        else:
            CheckIfGate[i] = 0
    out_arr=np.hstack([out_arr,CheckIfGate])
    out_arr=out_arr[:,:,np.newaxis]
    #print(out_arr)

#####Datei zuordnen
    if k ==0:
        Dataset=out_arr
        #print(Dataset.shape)
    else:
        Dataset=np.block([Dataset,out_arr])

print(Dataset.shape)

"""
Datei speichern
"""
np.save(
    "Dataset",
    Dataset
)


####Datei anschaulich machen
#x = list(range(distance.shape[0]))
#plt.plot(x, distance)
#plt.show()


