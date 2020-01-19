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

x_train = np.array(x_train)
y_train = np.array(y_train)

#print(x_train[0])


#for i in range(10):
#	cv2.imshow("img", x_train[i])
#	cv2.waitKey(0)


nodes = 130
#x_train = np.array(x_train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (178,)))
model.add(tf.keras.layers.Dense(nodes, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(nodes, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))


#compiling / importing in the training data

model.compile(optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 50)

def accuracy():
	pi = open("x_test.pickle", "rb")
	x_test = pickle.load(pi)
	del pi

	pis = open("y_test.pickle", "rb")
	y_test = pickle.load(pis)

	preds = []
	authn = y_test

	for i in x_test:
		prediction = np.argmax([model.predict(i.reshape(-1, 178))])
		preds.append(prediction)

	output = [x for x in list(zip(preds, y_test))]
	correct = 0
	for a,b in output:
		correct += 1 if a == b else 0

	return correct/len(y_test)

if __name__ == "__main__":
	print(accuracy())
