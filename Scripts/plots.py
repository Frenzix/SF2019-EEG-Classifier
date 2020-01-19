import matplotlib.pyplot as plt
import os
import random
import cv2
import pickle

ip = open("x_train.pickle" , "rb")
input_ = pickle.load(ip)
ip.close()

for image in input_:
	print(image)
	plt.imshow(image.reshape(30,30))
	plt.show()