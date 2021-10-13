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


### 0.2 load credit fraud data
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)   # anomaly data
n_inliers = n_samples - n_outliers                # normal data

blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

X_train = datasets[2]
y_train = np.asarray([0]*255)
rng = np.random.RandomState(42)
X_test = np.concatenate([X_train, rng.uniform(low=-6, high=6,
                                   size=(n_outliers, 2))], axis=0)
X_test_l = X_test.shape[1]
y_test = np.concatenate([[0]*255,[1]*45],axis=0)

### 0.3 normalize the data(not use)



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
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 2), iterations=500, d=d)
    ### only for simple model credit fraud
    # ano_score = model.fit(test_img.reshape(1,2),test_img.reshape(1,2),epochs=50,batch_size=1)
    ano_score = ano_score.history['loss'][-1]
    # plot_model(model, to_file='anomaly_detector.png', show_shapes=True, show_layer_names=True)
    # model.save_weights('weights/test_3_299.h5',True)
    # model.load_weights('weights/test_3_299.h5')
    # similar_img = model.predict(test_img.reshape(1,2))


    ### only for simple model credit fraud
    np_residual = test_img.reshape(2,1) - similar_img.reshape(2,1)

    # np_residual = (np_residual + 2)/4

    return test_img.reshape(2,1), similar_img.reshape(2,1), np_residual, ano_score



### compute anomaly score - sample from strange image
########### change!!!


n_test = X_test.shape[0]
m = range(n_test)  # X_test.shape[0]
score = np.zeros((n_test, 1))
# qurey = np.zeros((n_test, X_test_l, X_test_w, 1))
qurey = np.zeros((n_test, X_test_l, 1))
pred = np.zeros((n_test, X_test_l, 1))
diff = np.zeros((n_test, X_test_l, 1))
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

np.save('result_cluster_u/test_qurey', qurey)
np.save('result_cluster_u/test_pred', pred)
np.save('result_cluster_u/test_diff', diff)
np.save('result_cluster_u/test_score', score)
np.save('result_cluster_u/X_test', X_test)
np.save('result_cluster_u/y_test', y_test)
np.save('result_cluster_u/X_train', X_train)
np.save('result_cluster_u/y_train', y_train)
