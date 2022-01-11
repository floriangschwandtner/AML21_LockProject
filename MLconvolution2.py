import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

'''
This code uses the normal method of convolution model. The code requires the whole information of all points to predict 
the labels of all the points.
It may be able to be called implicit method, because the prediction of all the points are done simultaneously.'''

# =======================Vorbereitung der Datei===============================#
'''Not needed for this method
def getInputLabel(input, label, period=20):
    """"Funktion, die input_tensor und label_tensor erzeugt. Siehe Erklärung.
    Die Spalte des input_tensors ist die Punkte, die wir beurteilen möchten, ob er auf der Schleuse ist oder nicht.
    Die Zeile enthält die Positionen der benachbernten 20 Punkten. Deswegen 3(x,y,z)*20 =60.
    label_tensor enthält das Klass des Punkt. Auf der Schleuse->1, sonst->0. Diese Klassifizierung entsteht in "CreatingData.py"."""
    period = period  ###Die Anzahl der Punkten, die bei der Schätzung eines Punktes verwendet werden.
    input_tensor = []
    label_tensor = []
    for i in range(0, len(data) - period, 1):
        input_tensor.append(input[i:i + period, :].flatten())
        label_tensor.append(label[i + period // 2])

    input_tensor = np.array(input_tensor)
    # print(np.shape(input_tensor))
    label_tensor = np.array(label_tensor)
    # print(np.shape(label_tensor))
    return input_tensor, label_tensor'''


data = np.load("DataModified.npy")  # shape (13212, 4, 25) ######Loading Data
# print(data.shape)
input = data[:, 0:-1, :].transpose(2, 0, 1)
label = data[:, -1, :].transpose(1, 0)
# print(input)
print("input", input.shape)
# print(label)
print("label", label.shape)
#input = input.reshape(input.shape[0] * input.shape[1],
#                     input.shape[2])  #####Datei in eine Reihe umformen. Siehe Erklärung
print("input", input.shape)

print("label", label.shape)

##change period
'''for 0/1 problem
P = 20
input_tensor, label_tensor = getInputLabel(input, label, period=P)  #####Datei in Tensoren umformen.
'''
#plt.plot(range(label.shape[1]),label[0,:])
#plt.show()
input_tensor = input
# ↓↓↓↓if the problem is categorical use the following code (one hot function)↓↓↓↓
label_tensor=to_categorical(label)
print("input_tensor", input_tensor.shape)
print("label_tensor", label_tensor.shape)


#plt.plot(range(label_tensor.shape[1]),label_tensor[0,:,0])
#plt.show()

# =======================Datei zum Training und der Validierung verteilen============#
'''
X ist die Positionen oder andere Informationen ->Input
y ist die Klassifizierung, in diesem Fall 0 oder 1 ->richtige Lösung zum Vergleich mit der Ausgabe vom Model
train ist zum Training. Hier 80 % der gesamten Datei. Beliebig ausgewählt.
test ist zur Validierung. Hier 20 %. (der Rest von Tensoren ausser "train")
'''

X_train, X_test, y_train, y_test = train_test_split(input_tensor, label_tensor, test_size=0.1,
                                                    random_state=100, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#label_tensor=label_tensor.reshape([label_tensor.shape[0],-1,2])
#plt.plot(range(y_train.shape[1]),y_train[50,:,0])
#plt.plot(range(y_test.shape[0]),y_test[:,0,0])
# plt.plot(range(y_test.shape[2]),y_test[0,0,:])
#plt.show()

# ===========================Model erzeugen==========================#
'''
Die Anzahl der Layer, Knoten, Aktivierungsfunktion, Dropout.... diese Hyperparameter sind anzupassen.
'''

model_1 = Sequential()
model_1.add(Dense(200, activation='relu', input_shape=(input_tensor.shape[1],input_tensor.shape[2])))  # input_shape=(系列長T, x_tの次元)
model_1.add(Dropout(0.5))
model_1.add(Dense(2, activation='softmax'))
model_1.summary()
model_1.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])




# ======================Einstellung des Earlystopping================#
'''
Wenn das Model genug trainiert ist, stoppt das Training, auch wenn die maximale Anzahl der Epochs erreicht ist.
'''
earlystopping = EarlyStopping(monitor='loss', patience=5)

# ======================Training==================#
'''
Ihr sollt selber Batch_size und epochs lernen. Etwas schwirig zu erklären. 
Batch_size ist auch anzupassen. 
'''
model_1.fit(X_train, y_train, batch_size=10, epochs=5, callbacks=[earlystopping])

# =======================Bewertung des Models================#
'''
Das Model wird hier in die test-Datei eingesetzt. test-Datei sind nicht beim Training verwendet. Deshalb kennt das Model
 nicht damit vorher gelernt.

.evaluate ist die normale Bewertungsfunktion. Das ist aber nicht so wichtig, da die Ausgabe des Model kontinuierlich, aber 
Labels binear (0,1) sind. Also, diese Funktion bewertet, ob die Ausgabe KOMPLETT mit dem y_test ist. 

Accuracy zeigt, wie gut die Beurteilung des Models ist. Die kontinuierliche Werte der Ausgabe werden 0 oder 1 geteilt, je nachdem 
wie groß der Wert ist. 
'''
score = model_1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(y_test.shape)

result = model_1.predict(X_test)
print(result.shape)
print(result)

'''
y_test=y_test.flatten()
Accuracy = 0
for i in range(len(result)):
    if result[i] > 0.5 and y_test[i] > 0.9:
        Accuracy += 1
    elif result[i] < 0.5 and y_test[i] < 0.9:
        Accuracy += 1

print("Accuracy", Accuracy / len(result))
'''


#np.savetxt('result.csv', result, delimiter=',')

# =================Einfache Visualisierung des Ergebnis=========================#
'''
Blaue Lienie (Stange) sind die y_test, also die richtige Lösung.
Orange sind die Schätzungen aus dem Model. 
Je identischer sind die beide Verteilungen, desto besser ist das Ergebnis.
'''
plt.plot(range(result.shape[1]),y_test[100,:,1])
plt.plot(range(result.shape[1]),result[100,:,1])
plt.show()
