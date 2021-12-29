import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

data=np.load("Dataset.npy")#shape (13212, 4, 25)
#print(data.shape)
input_tensor=data[:,0:-1,:].transpose(2,0,1)
label_tensor=data[:,-1,:].transpose(1,0)
print(input_tensor)
print(input_tensor.shape)
print(label_tensor)
print(label_tensor.shape)

model_1 = Sequential()
model_1.add(Dense(5, activation='relu', input_shape=(3,)))#input_shape=(系列長T, x_tの次元) ここでは20個ずつ予想してるっぽい
model_1.add(Dropout(0.5))
model_1.add(Dense(1, activation='linear'))
model_1.summary()
model_1.compile(optimizer='adam',loss='mse',metrics=['mae'])

input_tensor=input_tensor.reshape(input_tensor.shape[0]*input_tensor.shape[1],input_tensor.shape[2])
print(input_tensor.shape)
label_tensor=label_tensor.flatten()
print(label_tensor.shape)

X_train, X_test, y_train, y_test = train_test_split(input_tensor, label_tensor, test_size=0.2,
                                                	random_state=100, shuffle = True)
print(X_train.shape)
print(y_train.shape)
#X_train=X_train.reshape(-1, X_test.shape[1]*X_test.shape[2])#X_test.shape[1]*X_test.shape[2]
#X_test=X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])

earlystopping = EarlyStopping(monitor='loss', patience=5)

model_1.fit(X_train, y_train, batch_size=10, epochs=50, callbacks=[earlystopping])

score = model_1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])