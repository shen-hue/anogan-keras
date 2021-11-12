from __future__ import print_function

import matplotlib

matplotlib.use('PDF')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
import anogan
import pandas as pd
from load_data import sine_data_generation, anomaly_sine_data_generation
from sklearn.datasets import make_blobs,make_moons
from scipy.io import loadmat


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=str, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

### 0. prepare data

### 0.2 load cluster data
n_samples = 600
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)   # anomaly data
n_inliers = n_samples - n_outliers                # normal data

# blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
# datasets = [
#     make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
#                **blobs_params)[0],
#     make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
#                **blobs_params)[0],
#     make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
#                **blobs_params)[0],
#     4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
#           np.array([0.5, 0.25])),
#     14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]
#
# X = datasets[0]
# rng = np.random.RandomState(42)
# X_test = np.concatenate([X, rng.uniform(low=-6, high=6,
#                                         size=(n_outliers, 2))], axis=0)
##  load artificial data
np.random.seed(10)
X = np.random.uniform(0,6,(n_inliers,4))
X = np.insert(X,4,values=X[:,0]+X[:,1],axis=1)
X = np.insert(X,5,values=X[:,2]+X[:,3],axis=1)
X_test = np.concatenate([X, np.random.uniform(0,6,(n_outliers,6))], axis=0)


y_train = np.asarray([0]*n_inliers)
X_test_l = X_test.shape[1]
y_test = np.concatenate([[0]*n_inliers,[1]*n_outliers],axis=0)

### 0.3 normalize the data(not use)
X_train = (X-np.min(X))/(np.max(X)-np.min(X))
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))


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
    model.load_weights('weights/artificial_encode.h5')
    # ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    ### only for simple model credit fraud
    similar_img = model.predict(test_img.reshape(1,-1))
    similar_img = similar_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)

    ### only for simple model credit fraud
    test_img = test_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)
    np_residual = test_img.reshape(1,-1) - similar_img.reshape(1,-1)
    ano_score = np.sum(abs(np_residual))

    # np_residual = (np_residual + 2)/4

    return test_img.reshape(1,-11), similar_img.reshape(1,-1), np_residual, ano_score



### compute anomaly score - sample from strange image
########### change!!!


n_test = X_test_standard.shape[0]
m = range(n_test)  # X_test.shape[0]
score = np.zeros((n_test, 1))
# qurey = np.zeros((n_test, X_test_l, X_test_w, 1))
qurey = np.zeros((n_test, X_test_l))
pred = np.zeros((n_test, X_test_l))
diff = np.zeros((n_test, X_test_l))

# train the encode on test data
model = anogan.anomaly_detector(g=None, d=None)
ano_score = model.fit(X_test_standard,X_test_standard,epochs=100,batch_size=1)
model.save_weights('weights/artificial_encode.h5', True)


for i in m:
    # img_idx = args.img_idx
    # label_idx = args.label_idx
    test_img = X_test_standard[i]
    # test_img = np.random.uniform(-1,1, (28,28,1))

    start = cv2.getTickCount()
    qurey[i], pred[i], diff[i], score[i] = anomaly_detection(test_img)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    # print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
    print("number: ", i, "score:", score[i])

np.save('result_artificial/test_qurey', qurey)
np.save('result_artificial/test_pred', pred)
np.save('result_artificial/test_diff', diff)
np.save('result_artificial/test_score', score)
np.save('result_artificial/X_test', X_test)
np.save('result_artificial/y_test', y_test)
np.save('result_artificial/X_train', X)
np.save('result_artificial/y_train', y_train)
