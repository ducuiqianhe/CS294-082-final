import numpy as np 
import pandas as pd
import glob
import gc
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity


# preprocess y data
def predata_y(path):
	file = pd.read_csv(path)
	num_rows = file.shape[0]
	print ("number of rows: ", num_rows)
	y = np.array(file["invasive"])
	return y

# preprocess x data
def predata_x(path, num_rows, start_num):
	name = [i for i in range(start_num, num_rows+start_num)]
	for i in range(num_rows):
		name[i] = str(path) + str(name[i]) + '.jpg'
	x = np.zeros((num_rows, 866, 1154, 3))
	for i in range(num_rows):
		if i % 200 ==0:
			print ("preprocessed" + str(i))
		path = name[i]
		img = load_img(path)
		img = img_to_array(img)
		img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
		img = np.array(img, dtype=np.uint8)
		x[i, :, :, :] = img
	return x

if __name__=="__main__":

	# x_train1 = predata_x('./train/', 1000, 1)
	# np.save('./data/x_train1', x_train1)
	# print ('x train1 saved')
	# print (x_train1.shape)
	# x_train2 = predata_x('./train/', 1000, 1000)
	# np.save('./data/x_train2', x_train2)
	# print ('x train2 saved')
	# print (x_train2.shape)
	# x_val = predata_x('./train/', 295, 2000)
	# np.save('./data/x_val', x_val)
	# print ('x val saved')
	# print (x_val.shape)

	# x_test = predata_x('./test/', 1531, 1)
	# np.save('./data/x_test', x_test)
	# print ('x test saved')
	# print (x_test.shape)
	# y_train = predata_y('./train_labels.csv')
	# np.save('./data/y_train', y_train)
	# print ('y train saved')
	# print (y_train.shape)
	y_val = predata_y('./validation.csv')
	np.save('./data/y_val', y_val)
	print ('y val saved')
	print (y_val.shape)