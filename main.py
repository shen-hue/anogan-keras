from __future__ import print_function

import matplotlib

matplotlib.use('PDF')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
import anogan
# from keras.layers import Input,Dense,Activation,Subtract
# from keras.models import Model
# from keras import backend as K

from tensorflow.keras.layers import Input,Dense,Activation,Subtract
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import pandas as pd
from load_data import sine_data_generation, anomaly_sine_data_generation
from sklearn.datasets import make_blobs,make_moons
from scipy.io import loadmat
import os


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=str, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()
#
def main():
### 0. prepare data
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)   # anomaly data
    n_inliers = n_samples - n_outliers                # normal data

    ##  0.1 load artificial data
    np.random.seed(10)
    X = np.random.uniform(0,6,(n_inliers,4))
    X = np.insert(X,4,values=X[:,0]+X[:,1],axis=1)
    X = np.insert(X,5,values=X[:,2]+X[:,3],axis=1)
    X_test = np.concatenate([X, np.random.uniform(0,6,(n_outliers,6))], axis=0)
    #
    #
    y_train = np.asarray([0]*n_inliers)
    X_test_l = X_test.shape[1]
    y_test = np.concatenate([[0]*n_inliers,[1]*n_outliers],axis=0).astype(float)

    ### 0.2 normalize the data(not use)
    X_train = (X-np.min(X))/(np.max(X)-np.min(X))
    X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))


    print('train shape:', X_train.shape)

#### training
    ### 1. train generator & discriminator
    if args.mode == 'train':
        Model_d, Model_g = anogan.train(64, X_train)

    ### 2. test generator
    generated_img = anogan.generate(25)





    ### 3. class anomaly detection

    def anomaly_detection(test_img, g=None, d=None):
        model = anomaly_score()
        model.load_weights('weights/artificial_classification_decode.h5')
        ano_score = model.predict(test_img.reshape(1,-1))

        test_img = test_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)


        return test_img.reshape(1,-11), ano_score






    # train the encode of GAN model on test data
    model_d = anogan.anomaly_detector(g=None, d=None)
    ano_score = model_d.fit(X_test_standard,X_test_standard,epochs=100,batch_size=1)
    model_d.save_weights('weights/artificial_classification_encode.h5', True)
    #
    def sum_of_residual(y_true, y_pred):
        return K.sum(K.abs(y_true - y_pred))
    # classification part of GAN model
    def anomaly_score():
        g = model_d
        model_d.load_weights('weights/artificial_classification_encode.h5')
        g.trainable = False
        input = Input(shape=(6,))
        G_out = g(input)
        input_l = Subtract()([G_out,input])
        layer_1 = Dense(256)(input_l)
        layer_1 = Activation('relu')(layer_1)
        layer_2 = Dense(1)(layer_1)
        output = Activation('sigmoid')(layer_2)
        model_anomaly = Model(inputs=input,outputs=output)
        model_anomaly.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
        return model_anomaly

    # train the classification part
    model_anomaly = anomaly_score()
    model_anomaly.fit(X_test_standard,y_test,epochs=100,batch_size=1)
    model_anomaly.save_weights('weights/artificial_classification_decode.h5', True)

# test data on GAN model
    n_test = X_test_standard.shape[0]
    m = range(n_test)  # X_test.shape[0]
    score = np.zeros((n_test, 1))
    # qurey = np.zeros((n_test, X_test_l, X_test_w, 1))
    qurey = np.zeros((n_test, X_test_l))
    pred = np.zeros((n_test, X_test_l))
    diff = np.zeros((n_test, X_test_l))

    for i in m:
        test_img = X_test_standard[i]
        start = cv2.getTickCount()
        qurey[i], score[i] = anomaly_detection(test_img)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print("number: ", i, "score:", score[i])

    if os.path.exists('result_artificial_classification')==False:
        os.mkdirs('result_artificial_classification')
    np.save('result_artificial_classification/test_qurey', qurey)
    np.save('result_artificial_classification/test_pred', pred)
    np.save('result_artificial_classification/test_diff', diff)
    np.save('result_artificial_classification/test_score', score)
    np.save('result_artificial_classification/X_test', X_test)
    np.save('result_artificial_classification/y_test', y_test)
    np.save('result_artificial_classification/X_train', X)
    np.save('result_artificial_classification/y_train', y_train)
#

if __name__ == '__main__':
    main()
