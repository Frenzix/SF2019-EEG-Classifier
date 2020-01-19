import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import numpy as np

style.use("ggplot")

col = {1: "Abnormal EEG", 5: "Normal EEG"}

df = pd.read_csv("data.csv", index_col = 0)
df = df.loc[df['y'].isin([1,5])]

pickle_outtr = open("x_trainx.pickle" , "wb")
pickle_outla = open("y_trainx.pickle", "wb")

x_train = []
y_train = []

keys_ = [x for x,y in col.items()]
for i, row in df.iterrows():
	a = row.tolist()

	x_train.append(a[:-1])
	y_train.append(0 if a[-1] == 1 else 1)

x_train = np.array(x_train).reshape(-1, 178)
pickle.dump(x_train, pickle_outtr)
pickle.dump(y_train, pickle_outla)

pickle_outtr.close()
pickle_outla.close()

#for (c, abbr) in col.items():

	#df.drop(['X1'], axis=1, inplace=True)


	#img_c = 0
	#for i, row in df.iterrows():
			#row.plot()
			#plt.savefig(f"C:\\Users\\HP\\Desktop\\Scfi\\{abbr}\\img{img_c}.png")
			#plt.close()
			#img_c+=1
	#		print(i)