from tensorflow.python.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


model = load_model('Convolution2_X10Y10Z20.h5')

model.summary()

data = np.load("DataForTraining.npy")

#print(data)
input = data[:, 0:-1, :].transpose(2, 0, 1)
label = data[:, -1, :].transpose(1, 0)
# print(input)
print("input", input.shape)
# print(label)
print("label", label.shape)

input_tensor = input
label_tensor=to_categorical(label)
print("input_tensor", input_tensor.shape)
print("label_tensor", label_tensor.shape)

X_train, X_test, y_train, y_test = train_test_split(input_tensor, label_tensor, test_size=0.1,
                                                    random_state=100, shuffle=True)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

result = model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(4, 1, 1)
ax.set_title("not Wall")
ax.plot(range(result.shape[1]), y_test[0,:,0])
ax.plot(range(result.shape[1]), result[0,:,0])
ax = fig.add_subplot(4, 1, 2)
ax.set_title("right Wall")
ax.plot(range(result.shape[1]), y_test[0,:,1])
ax.plot(range(result.shape[1]), result[0,:,1])
ax = fig.add_subplot(4, 1, 3)
ax.set_title("left links")
ax.plot(range(result.shape[1]), y_test[0,:,2])
ax.plot(range(result.shape[1]), result[0,:,2])
ax = fig.add_subplot(4, 1, 4)
ax.set_title("Gate")
ax.plot(range(result.shape[1]), y_test[0,:,3])
ax.plot(range(result.shape[1]), result[0,:,3])
fig.tight_layout()
#fig.savefig("Conv2ShortResult.png")
#plt.show()

Frame=0
Data=X_test[Frame,:,:]
ResultCa=result[Frame,:,:]
ResultNu=np.argmax(ResultCa, axis=1)
print(ResultNu)
y_testCa=y_test[Frame,:,:]
y_testNu=np.argmax(y_testCa, axis=1)

DataForFig=np.zeros((Data.shape[0],Data.shape[1]+1))
for i in range(Data.shape[0]):
    DataForFig[i,:]=np.append(Data[i,:],ResultNu[i])


print(DataForFig.shape)

fig2 = plt.figure(figsize=(8,8))
ax = fig2.add_subplot(111, projection='3d')

skip=5  #to reduce the number of points
for i in range(int(np.floor(DataForFig.shape[0]/skip))):
    i=i*skip
    if DataForFig[i,3]==0 and y_testNu[i]==0:#keine Wand
        ax.scatter(DataForFig[i,0],DataForFig[i,1],DataForFig[i,2], c="green")
    elif DataForFig[i,3]==1 and y_testNu[i]==1:#seitlich
        ax.scatter(DataForFig[i,0],DataForFig[i,1],DataForFig[i,2], c="yellow")
    elif DataForFig[i,3]==2 and y_testNu[i]==2:#seitlich
        ax.scatter(DataForFig[i,0],DataForFig[i,1],DataForFig[i,2], c="blue")
    elif DataForFig[i,3]==3 and y_testNu[i]==3:#gate
        ax.scatter(DataForFig[i,0],DataForFig[i,1],DataForFig[i,2], c="black")
    else:
        ax.scatter(DataForFig[i, 0], DataForFig[i, 1], DataForFig[i, 2], c="red")


plt.show()
