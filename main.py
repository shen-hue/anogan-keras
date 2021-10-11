from __future__ import print_function

import matplotlib

matplotlib.use('PDF')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import anogan
import keras
import pandas as pd
from load_data import sine_data_generation, anomaly_sine_data_generation
from sklearn import datasets
from scipy.io import loadmat

iris = datasets.load_iris()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=str, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

### 0. prepare data


### 0.2 load credit fraud data

tmp = loadmat('wine.mat')
X = [[row.flat[0] for row in line] for line in tmp['X']]
y = [[row.flat[0] for row in line] for line in tmp['y']]

X = np.array(X)
y = np.array(y)
X_train = X[21:][:]
y_train = y[21:][:]
X_test = X[:21][:]
y_test = y[:21][:]
X_test_l = X_train.shape[1]


# X_train = X_train[y_train == 1]       # in mnist data number"1" as normal data
# X_train = X_train[y_train == 0]         # in credit fraud data 0 as normal data

### 0.3 normalize the data(not use)
# plt.hist(X_train, 40)
# plt.savefig('X_train')
X_train = (X_train-X_train.min(axis=0))/(X_train.max(axis=0)-X_train.min(axis=0))
X_test = (X_test-X_test.min(axis=0))/(X_test.max(axis=0)-X_test.min(axis=0))



print('train shape:', X_train.shape)

### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(64, X_train)

### 2. test generator
generated_img = anogan.generate(25)
# img = anogan.combine_images(generated_img)
# # img = (img*127.5)+127.5
# img = img.astype(np.uint8)
# img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)




### 3. other class anomaly detection

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    model.load_weights('weights/encode.h5')
    # ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    ### only for simple model credit fraud
    similar_img = model.predict(test_img.reshape(1,13))


    ### only for simple model credit fraud
    np_residual = test_img.reshape(13,1) - similar_img.reshape(13,1)
    ano_score = np.mean(np_residual)

    # np_residual = (np_residual + 2)/4

    return test_img.reshape(13,1), similar_img.reshape(13,1), np_residual, ano_score



### compute anomaly score - sample from strange image
########### change!!!


n_test = X_test.shape[0]
m = range(n_test)  # X_test.shape[0]
score = np.zeros((n_test, 1))
# qurey = np.zeros((n_test, X_test_l, X_test_w, 1))
qurey = np.zeros((n_test, X_test_l, 1))
pred = np.zeros((n_test, X_test_l, 1))
diff = np.zeros((n_test, X_test_l, 1))

# train the encode on test data
model = anogan.anomaly_detector(g=None, d=None)
ano_score = model.fit(X_test,X_test,epochs=500,batch_size=1)
model.save_weights('weights/encode.h5', True)


for i in m:
    # img_idx = args.img_idx
    # label_idx = args.label_idx
    test_img = X_test[i]
    # test_img = np.random.uniform(-1,1, (28,28,1))

    start = cv2.getTickCount()
    qurey[i], pred[i], diff[i], score[i] = anomaly_detection(test_img)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    # print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
    # print("number: ", i, "score:", score[i])

np.save('result_f_wine_NN/wine_test_qurey', qurey)
np.save('result_f_wine_NN/wine_test_pred', pred)
np.save('result_f_wine_NN/wine_test_diff', diff)
np.save('result_f_wine_NN/wine_test_score', score)
np.save('result_f_wine_NN/X_test', X_test)
np.save('result_f_wine_NN/y_test', y_test)
np.save('result_f_wine_NN/X_train', X_train)
np.save('result_f_wine_NN/y_train', y_train)
