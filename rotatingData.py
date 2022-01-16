'''
This is to increase the variety and the number of the dataset for the ML,
so that the ML model will be trained independent from the data to be used for training.

The data as input: shape [25, 13000, 4] =[number of measuring, number of points, label]
'''
import numpy as np
import matplotlib.pyplot as plt


#############defin the rotation matrix
##Reference Flugmechanik Euler Drehmatrizen



def T1(x, phi):
    '''
    Rotation on x axis of the ship (roll)(the axis from back to front of the ship)
    :param x: the matrix to be rotated. shape [3,1]
    :param phi: the rotation angle in degree(°)
    :return: rotated matrix
    '''
    phi_rad = np.radians(phi)
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    r=np.asarray([[1,0,0],
                  [0,c,s],
                  [0,-s,c]])
    return r @ x

def T2(x, phi):
    '''
    Rotation on y axis of the ship (pitch)(the axis from right to left of the ship)
    :param x: the matrix to be rotated. shape [3,1]
    :param phi: the rotation angle in degree(°)
    :return: rotated matrix
    '''
    phi_rad = np.radians(phi)
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    r=np.asarray([[c,0,-s],
                  [0,1,0],
                  [s,0,c]])
    return r @ x

def T3(x, phi):
    '''
    Rotation on y axis of the ship (yaw)(the axis from right to left of the ship)
    :param x: the matrix to be rotated. shape [3,1]
    :param phi: the rotation angle in degree(°)
    :return: rotated matrix
    '''
    phi_rad = np.radians(phi)
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    r=np.asarray([[c,s,0],
                  [-s,c,0],
                  [0,0,1]])
    return r @ x

#print(np.random.randint(1,4,(30,25)))


################# get data
data = np.load("LabeledOriginalMatrix.npy")

#for test
head=-1
stack=25
data = data[:head,:,:stack]
#labels=np.random.randint(1,5,(30))
#print(labels)
#data[:,-1,:]=np.random.randint(1,5,(head,stack)) ###making labels for test

#print(data)
#print(data.shape)

#####get rid of the data points on the ship

#25のやつそれぞれにこれをしないといけない　
#さもないと消す行が不明瞭→matrixの形がそれぞれ異なる　重ねられない　4もふくめて学習
'''LabelToBeDeleted=4
print(np.where(data[:,3,:]==LabelToBeDeleted))
print(np.where(data[:,:,:]==LabelToBeDeleted))
dataNew1 = np.delete(data, np.where(data[:,3,:]==LabelToBeDeleted ),0)
print(dataNew1)
dataNew2 = np.delete(data, np.where(data[:,:,:]==LabelToBeDeleted ),0)
print(dataNew2)
#dataNew = np.delete(data, np.where(data==LabelToBeDeleted),0)###not ideal
dataNew=[]
for i in range(data.shape[2]):
    dataNew += [np.delete(data, np.where(data[:,3,:]==LabelToBeDeleted ),0)]


#condition = (data[:,3,:]!=LabelToBeDeleted)
#dataNew = np.extract(condition, data)
print(dataNew1.shape)
print(dataNew2.shape)'''
dataNew=data

###########define the range of each rotation
#randomized values will be chosen from homogenous distribution (value) is the length of the matrix
roll = np.random.randint(-10,10,(10))
pitch= np.random.randint(-10,10,(10))
yaw  = np.random.randint(-60,60,(20))

############creating the dataset

x=dataNew.shape[0]
y=dataNew.shape[1]
z=dataNew.shape[2]

#Roll
DataRoll=np.zeros((x,y,z*roll.shape[0]))
for k in range(roll.shape[0]):
    for j in range(y):
        for i in range(x):
            DataRoll[i,:-1,k*z+j]=T1(dataNew[i,:-1,j],roll[k])
        DataRoll[:,-1,k*z+j]=dataNew[:,-1,j]
#Pitch
DataPitch=np.zeros((x,y,z*pitch.shape[0]))
for k in range(pitch.shape[0]):
    for j in range(y):
        for i in range(x):
            DataPitch[i,:-1,k*z+j]=T2(dataNew[i,:-1,j],pitch[k])
        DataPitch[:,-1,k*z+j]=dataNew[:,-1,j]

#Yaw
DataYaw=np.zeros((x,y,z*yaw.shape[0]))
for k in range(pitch.shape[0]):
    for j in range(y):
        for i in range(x):
            DataYaw[i,:-1,k*z+j]=T2(dataNew[i,:-1,j],yaw[k])
        DataYaw[:,-1,k*z+j]=dataNew[:,-1,j]

print(dataNew.shape)
print(DataRoll.shape)
Data=np.concatenate((dataNew, DataRoll, DataPitch, DataYaw),axis=2)

print(Data.shape)


###################save data
np.save(
    "DataForTraining",
    Data
)
















