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

iris = datasets.load_iris()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=str, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

### 0. prepare data
### 0.1 load mnist data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.astype(np.float32) - 127.5) / 127.5
#
# X_train = X_train[:,:,:,None]
# X_test = X_test[:,:,:,None]
# X_train_l, X_train_w = X_train.shape[1], X_train.shape[2]
# X_test_l, X_test_w = X_test.shape[1], X_test.shape[2]

### 0.2 load credit fraud data
tmp = pd.read_csv("creditcard.csv", encoding='gbk', header=None)
X_train = tmp.iloc[1:10000, 1:-2].values.astype(np.float)
y_train = tmp.iloc[1:10000, -1].values.astype(np.float)
X_test = tmp.iloc[10000:, 1:-2].values.astype(np.float)
y_test = tmp.iloc[10000:, -1].values.astype(np.float)

# ### 0.3 load sine data
# sequence = 28       # sequence length
# dimension = 28      # dimension of each sequence
# X_train = sine_data_generation(10000, sequence, dimension)
# y_train = np.zeros((10000,))
# n_test_n = 800      # number of normal data in test data set
# n_test_a = 200      #number of anomaly data in test data set
# X_test_n = sine_data_generation(n_test_n, sequence, dimension)
# X_test_a = anomaly_sine_data_generation(200, sequence, dimension)
# X_test = np.concatenate((X_test_n,X_test_a), axis=0)
# y_test_n = np.zeros((n_test_n),)
# y_test_a = np.ones((n_test_a),)
# y_test = np.concatenate((y_test_n,y_test_a), axis=0)
#
# ### 0.3.1 show the training data and test data
#
# [plt.plot(np.arange(sequence), X_train[1][:,s]) for s in range(sequence)]
# plt.title('training data')
# plt.savefig('result_sin/train_data')
# [plt.plot(np.arange(sequence), X_test[801][:,s]) for s in range(sequence)]
# plt.title('test data')
# plt.savefig('result_sin/test_data')

# X_test_original = X_test.copy()

# X_train = X_train[y_train == 1]       # in mnist data number"1" as normal data
# X_train = X_train[y_train == 0]         # in credit fraud data 0 as normal data

### 0.3 normalize the data(not use)
# plt.hist(X_train, 40)
# plt.savefig('X_train')
# X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
# X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))


### 0.4 reshape the data(only for credit fraud)
# X_train = np.repeat(X_train[:, np.newaxis], 28, axis=1)
# X_test = np.repeat(X_test[:, np.newaxis], 28, axis=1)
#
# X_train = X_train[:, :, :, None]
# X_test = X_test[:, :, :, None]
#
# X_train_l, X_train_w = X_train.shape[1], X_train.shape[2]
# X_test_l, X_test_w = X_train.shape[1], X_test.shape[2]

### 0.4 lenth of the data(for simple model credit fraud)
X_train_l = X_train.shape[1]
X_test_l = X_train.shape[1]

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


### opencv view
# cv2.namedWindow('generated', 0)
# cv2.resizeWindow('generated', 256, 256)
# cv2.imshow('generated', img)
# cv2.imwrite('result_latent_10/generator.png', img)
# cv2.waitKey()

### plt view
# plt.figure(num=0, figsize=(4, 4))
# plt.title('trained generator')
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# exit()

### 3. other class anomaly detection

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    # ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    ### only for simple model credit fraud
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28), iterations=500, d=d)

    # anomaly area, 255 normalization
    # np_residual = test_img.reshape(28, 28, 1) - similar_img.reshape(28, 28, 1)
    ### only for simple model credit fraud
    np_residual = test_img.reshape(28,1) - similar_img.reshape(28,1)

    # np_residual = (np_residual + 2)/4

    # np_residual = (np_residual*(np.max(X_test)-np.min(X_test))+np.min(X_test)).astype(np.uint8)         # inverse the normalization
    # original_x = (test_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)).astype(np.uint8)       # inverse the normalization
    # similar_x = (similar_img*(np.max(X_test)-np.min(X_test))+np.min(X_test)).astype(np.uint8)       # inverse the normalization

    # original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)    # 将灰度图像转为彩色
    # original_x = cv2.applyColorMap(original_x, cv2.COLORMAP_JET)
    # similar_x = cv2.applyColorMap(similar_x,cv2.COLORMAP_JET)
    # residual_color = cv2.applyColorMap(np_residual.astype(np.uint8), cv2.COLORMAP_JET)      #热力图
    # show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)      #融合图像

    return ano_score, test_img.reshape(28,1), similar_img.reshape(28,1), np_residual


### compute anomaly score - sample from test set
# test_img = X_test_original[y_test==1][30]

### compute anomaly score - sample from strange image
# test_img = X_test_original[y_test==0][30]

### compute anomaly score - sample from strange image
########### change!!!
### select the test data(only for credit fraud)
X_test1 = X_test[y_test == 0]
X_test1 = X_test1[:800]
y_test1 = y_test[y_test == 0]
y_test1 = y_test1[:800]
X_test2 = X_test[y_test == 1]
X_test2 = X_test2[:200]
y_test2 = y_test[y_test == 1]
y_test2 = y_test2[:200]
X_test = np.concatenate((X_test1, X_test2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)

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
    score[i], qurey[i], pred[i], diff[i] = anomaly_detection(test_img)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    # print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
    print("number: ", i, "score:", score[i])

np.save('result_credit_NN/credit_test_score', score)
np.save('result_credit_NN/credit_test_qurey', qurey)
np.save('result_credit_NN/credit_test_pred', pred)
np.save('result_credit_NN/credit_test_diff', diff)
np.save('result_credit_NN/X_test', X_test)
np.save('result_credit_NN/y_test', y_test)
np.save('result_credit_NN/X_train', X_train)
np.save('result_credit_NN/y_train', y_train)

# cv2.imwrite('./qurey.png', qurey)
# cv2.imwrite('./pred.png', pred)
# cv2.imwrite('./diff.png', diff)

###### histogram

plt.hist(score, 40)
plt.savefig('result_credit_NN/histogram')
#
# ## matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(qurey.reshape(28,28), cmap=plt.cm.gray)
#
# print("anomaly score : ", score)
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(pred.reshape(28,28), cmap=plt.cm.gray)
#
# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
# plt.show()


### 4. tsne feature view

### t-SNE embedding
### generating anomaly image for test (radom noise image)

# from sklearn.manifold import TSNE
#
# random_image = np.random.uniform(0, 1, (100, 28, 28, 1))
# print("random noise image")
# # plt.figure(4, figsize=(2, 2))
# # plt.title('random noise image')
# # plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
#
# # intermidieate output of discriminator
# model = anogan.feature_extractor()
# feature_map_of_random = model.predict(random_image, verbose=1)
# feature_map_of_minist = model.predict(X_test[y_test != 1][:300], verbose=1)
# feature_map_of_minist_1 = model.predict(X_test[:100], verbose=1)
#
# # t-SNE for visulization
# output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
# output = output.reshape(output.shape[0], -1)
# anomaly_flag = np.array([1]*100+ [0]*300)
#
# X_embedded = TSNE(n_components=2).fit_transform(output)
# plt.figure(5)
# plt.title("t-SNE embedding on the feature representation")
# plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
# plt.scatter(X_embedded[100:400,0], X_embedded[100:400,1], label='mnist(anomaly)')
# plt.scatter(X_embedded[400:,0], X_embedded[400:,1], label='mnist(normal)')
# plt.legend()
# plt.show()
