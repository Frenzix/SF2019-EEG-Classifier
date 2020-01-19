import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

classes = ['AbnormalEEG' , 'NormalEEG']

image_dir = "C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data"
test_dir = "C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data\\Testing"
training_data, testing_data = [], []

def create_tdata():
	for c in classes:
		path = os.path.join(image_dir, c)
		c_num = classes.index(c)
		for img in os.listdir(path):
			try:
				img_ = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
				if img_ is None:
					print("failure", os.path.join(path,img))
				img2 = cv2.resize(img_, (30,30))
				training_data.append([img2, c_num])
			except Exception as e:
				print(e)

random.shuffle(training_data)

X, y = [], []

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1,30, 30, 1)
print(X.shape)
print(X)

pickle_out = open("x_train.pickle" , 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y_train.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

'''def create_test_data():
	global testing_data, training_data
	for c in classes:
		path = os.path.join(test_dir, c)
		c_num = classes.index(c)
		for img in os.listdir(path):
			try:
				i = os.path.join(path,img)
				print(i)
				img_ = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
				if img_ is None:
					print("failure", os.path.join(path,img))
				img_ = cv2.resize(img_ , (30,30))
				
				testing_data.append([img_, c_num])
			except Exception as e:
				print(e)


create_test_data()'''

X, y = [], []

for features, label in testing_data:
	X.append(features)
	y.append(label)
