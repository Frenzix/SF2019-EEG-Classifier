 #imports

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt 
import cv2
import numpy as np

#opening migrated data stored in the pickle files

classes = ['AbnormalEEG' , 'NormalEEG']

pickle_in = open("x_train.pickle", "rb")
x_train = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y_train.pickle" , "rb")
y_train = pickle.load(pickle_in)
pickle_in.close()

print(len(x_train))
print(x_train.shape)
print(x_train[0].shape)

plt.imshow(x_train[0].reshape(30,30))
plt.show()

#for i in range(10):
#	cv2.imshow("img", x_train[i])
#	cv2.waitKey(0)

X = x_train[0]
nodes = 50

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (30, 30, 1)))
model.add(tf.keras.layers.Dense(nodes, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(nodes, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))


#compiling / importing in the training data

model.compile(optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 20)

import os

direc = "C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data\\Testing\\"
c_ = 0

for c in classes:
    direc_ = ''.join((direc, c))
    for file in os.listdir(direc_):
        filex = os.path.join(direc_, file)
        r = cv2.imread(filex)
        r.resize(30,30)
        pred = model.predict(r.reshape(-1,30,30,1))
        pred = np.argmax([pred])        
        print(file, pred, c)
        if classes[pred] == c:
            c_ += 1
print("accuracy : " + str(float(c_/100)))
