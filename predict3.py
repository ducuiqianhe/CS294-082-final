import numpy as np 
import pandas as pd
import glob
import gc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

with tf.device('/cpu:0'):
	data_path = "./data3"
	check_path = "./newnewModel/check0/weights-32-0.88136.hdf5"

	#out_name = model_path.replace("storage/model1", "submissions").replace("hdf5", "csv")
	out_name = './submission/newnew_model0.1.csv'

	print ("Loading model...")
	model = load_model(check_path)

	test_data = np.load(open('./data3/bottleneck_features_test.npy','rb'))


	# name = [i for i in range(1, 1532)]
	# for i in range(1531):
	# 	name[i] = "./image/test/" + str(name[i]) + '.jpg'
	# # x = np.zeros((num_rows, 866, 1154, 3))
	# x = np.zeros((1531, 480, 640, 3))
	# for i in range(1531):
	# 	if i%200==0:
	# 		print (i)
	# 	path = name[i]
	# 	img = load_img(path, target_size=(480, 640))
	# 	img = img_to_array(img)
	# 	img = np.array(img, dtype=np.uint8)
	# 	img = img/255
	# 	x[i, :, :, :] = img

	# x = x.astype('float32')

	print ("Predicting...")
	preds = model.predict(test_data, batch_size=10)
	preds = preds[:, 0]
	print ('preds: ', preds)

	gc.collect()

	print ("Writing predictions...")
	name = [i for i in range(1, 1532)]
	pd.DataFrame({"name": name, "invasive": preds}).to_csv(out_name, index=False, header=True)

	print ("Done!")
	gc.collect()