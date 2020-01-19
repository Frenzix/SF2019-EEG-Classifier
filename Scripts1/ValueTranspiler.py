import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import numpy as np

style.use("ggplot")

col = {1: "Abnormal EEG", 5: "Normal EEG"}

df = pd.read_csv("data.csv", index_col = 0)
df = df.loc[df['y'].isin([1,5])]

pickle_outtr = open("x_train.pickle" , "wb")
pickle_outla = open("y_train.pickle", "wb")

pickle_outte = open("x_test.pickle", "wb")
pickle_outlA = open("y_test.pickle" , "wb")

xte = []
yte = []

x_train, y_train, x_test, y_test = [], [], [], []

keys_ = [x for x,y in col.items()]
for i, row in df.iterrows():
	a = row.tolist()
	xte.append(np.array(a[:-1]).reshape(178))
	yte.append(0 if a[-1] == 1 else 1)

pickle.dump(xte[:4500], pickle_outtr)
pickle.dump(yte[:4500], pickle_outla)

pickle.dump(xte[4500:], pickle_outte)
pickle.dump(yte[4500:], pickle_outlA)

pickle_outtr.close()
pickle_outla.close()
pickle_outte.close()
pickle_outlA.close()



	
#for (c, abbr) in col.items():

	#df.drop(['X1'], axis=1, inplace=True)


	#img_c = 0
	#for i, row in df.iterrows():
			#row.plot()
			#plt.savefig(f"C:\\Users\\HP\\Desktop\\Scfi\\{abbr}\\img{img_c}.png")
			#plt.close()
			#img_c+=1
	#		print(i)

'''
	direc = "C:\\Users\\HP\\Desktop\\Science Fair Project\\Image Data\\Testing\\"
c_ = 0

for c in classes:
    direc_ = ''.join((direc, c))
    for file in os.listdir(direc_):
        filex = os.path.join(direc_, file)
        r = cv2.imread(filex)
        r.resize(30,30)
        pred = model.predict(r.reshape(-1,30,40,1))
        pred = np.argmax([pred])        
        print(file, pred, c)
        if classes[pred] == c:
            c_ += 1
print("accuracy : " + str(float(c_/100)))'''








'''



def allocate_data():
	global x_train, y_train
	global x_test, y_test

	x_train.append(xte[:4500])
	y_train.append(yte[:4500])

	x_test.append(xte[4500:])
	y_test.append(yte[4500:])

if __name__ == "__main__":

	allocate_data()

	assert(len(x_train[0]) == 4500)
	assert(len(y_train[0]) == 4500)

	x_train = np.array(x_train[0])
	x_test = np.array(x_test[0])
	print(x_train.shape)
	print(x_test.shape)
	#x_train.reshape(-1, 780)
	#x_test.reshape(-1, 780)

	pickle.dump(x_train, pickle_outtr)
	pickle.dump(y_train, pickle_outla)
	
	pickle.dump(x_test, pickle_outte)
	pickle.dump(y_test,pickle_outlA)
	
	pickle_outtr.close()
	pickle_outla.close()

	pickle_outte.close()
	pickle_outlA.close()'''