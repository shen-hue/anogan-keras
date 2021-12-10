from __future__ import print_function

import matplotlib
from keras.utils.vis_utils import plot_model

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
from sklearn.datasets import make_blobs,make_moons
from scipy.io import loadmat


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=str, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

### 0. prepare data


### 0.1 load artificial data
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)   # anomaly data
n_inliers = n_samples - n_outliers                # normal data

np.random.seed(10)
X = np.random.uniform(0,6,(n_inliers,4))
X = np.insert(X,4,values=X[:,0]+X[:,1],axis=1)
X = np.insert(X,5,values=X[:,2]+X[:,3],axis=1)
X_test = np.concatenate([X, np.random.uniform(0,6,(n_outliers,6))], axis=0)
#
#
y_train = np.asarray([0]*n_inliers)
X_test_l = X_test.shape[1]
y_test = np.concatenate([[0]*n_inliers,[1]*n_outliers],axis=0)

### 0.2 normalize the data
X_train = (X-np.min(X))/(np.max(X)-np.min(X))
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))


print('train shape:', X_train.shape)

### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(64, X_train)

### 2. test generator
generated_img = anogan.generate(25)




### 3. class anomaly detection

def anomaly_detection(test_img,j, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score = model.fit(test_img.reshape(1,-1),test_img.reshape(1,-1),epochs=500,batch_size=1)
    ano_score = ano_score.history['loss'][-1]
    model.save_weights('result_artificial/weights/test_1_'+str(j)+'.h5',True)
    model.load_weights('result_artificial/weights/test_1_'+str(j)+'.h5')
    similar_img = model.predict(test_img.reshape(1,-1))
    similar_img = similar_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)


    test_img = test_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)
    np_residual = test_img.reshape(1,6) - similar_img.reshape(1,6)


    return test_img.reshape(1,6), similar_img.reshape(1,6), np_residual, ano_score



### 4 compute anomaly score


n_test = X_test_standard.shape[0]
m = range(n_test)  # X_test.shape[0]
score = np.zeros((n_test, 1))
qurey = np.zeros((n_test, 6))
pred = np.zeros((n_test, 6))
diff = np.zeros((n_test, 6))
for i in [10]:
    test_img = X_test_standard[i]
    start = cv2.getTickCount()
    qurey[i], pred[i], diff[i], score[i] = anomaly_detection(test_img,i)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    # print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
    print("number: ", i, "score:", score[i])

# save results
if os.path.exists('result_artificial')==False:
    os.mkdir('result_artificial')
np.save('result_artificial/test_qurey', qurey)
np.save('result_artificial/test_pred', pred)
np.save('result_artificial/test_diff', diff)
np.save('result_artificial/test_score', score)
np.save('result_artificial/X_test', X_test)
np.save('result_artificial/y_test', y_test)
np.save('result_artificial/X_train', X_train)
np.save('result_artificial/y_train', y_train)