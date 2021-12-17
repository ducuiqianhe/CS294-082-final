# from os import listdir
# from os.path import isfile, join

# for f in listdir("F:\\python\\invasive\\newnewModel\\check0"):
# 	print(f)


import numpy as np
import math


# calculate expected MEC
train_data = np.load(open('./data3/bottleneck_features_train1.npy','rb'))
print (train_data.shape)
train_labels = np.array([0] * 738 + [1] * 1262)

data_sum = {}
print(train_data[0].shape)
print(train_data[0][0].shape)
print(train_data[0][0][0].shape)
for pic in range(2000):
	print(pic)
	pic_sum = 0
	for i in range(15):
		for j in range(20):
			for k in range(512):
				pic_sum += train_data[pic][i][j][k]
	data_sum[pic_sum] = train_labels[pic]

threshold = 0
label = 0
for i in sorted(data_sum.keys()):
	if data_sum[i] != label:
		label = data_sum[i]
		threshold += 1

max_bit = threshold*(15*20*512+1)+1
expected = math.log2(threshold+1)*15*20*512
print(max_bit)
print(expected)

