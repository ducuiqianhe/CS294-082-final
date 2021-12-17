import numpy as np 
import pandas as pd
import glob
import gc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# with tf.device('/cpu:0'):
data_path = "./data3"
check_path = "./newModel/checkfull/weights4-01-0.92069.hdf5"

#out_name = model_path.replace("storage/model1", "submissions").replace("hdf5", "csv")
out_name = './submission/newnew_model4.csv'

print ("Loading model...")
model = load_model(check_path)

i = 1
predsall = []
for i in range(0, 153):
	a = 10*i
	if i == 152:
		b = 1532
		n = 11
	else:
		b = a+11
		n = 10

	name = [i for i in range(a+1, b)]
	print (name)
	for i in range(n):
		name[i] = "./image/test/0/" + str(name[i]) + '.jpg'
	# x = np.zeros((num_rows, 866, 1154, 3))
	x = np.zeros((n, 480, 640, 3))
	for i in range(n):
		path = name[i]
		img = load_img(path, target_size=(480, 640))
		img = img_to_array(img)
		img = np.array(img, dtype=np.uint8)
		img = img/255
		x[i, :, :, :] = img

	x = x.astype('float32')

	print ("Predicting...")
	preds = model.predict(x)
	preds = preds[:, 0]
	print ('preds: ', preds)
	predsall = np.concatenate((predsall, preds))

	gc.collect()

print (predsall)
print ("predsall: ", len(predsall))

for i, n in enumerate(predsall):
	if n < 0.5:
		predsall[i] = 0
	else:
		predsall[i] = 1



print ("Writing predictions...")
name = [i for i in range(1, 1532)]
pd.DataFrame({"name": name, "invasive": predsall}).to_csv(out_name, index=False, header=True)

print ("Done!")
gc.collect()