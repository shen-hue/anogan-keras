import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib
matplotlib.use('PDF')
from visualization import visualization, visualization_confusion_matrix

# load test result
score = np.load('result_credit_NN/credit_test_score.npy')
qurey= np.load('result_credit_NN/credit_test_qurey.npy')
pred = np.load('result_credit_NN/credit_test_pred.npy')
diff = np.load('result_credit_NN/credit_test_diff.npy')

threshold = 100

# order prediction result(anomaly:1, normal:0)
score = score.flatten()
pred_y = np.zeros((score.shape[0]))
pred_y[score > threshold] = 1            #anomaly 1, normal 0
X_test = np.load('result_credit_NN/X_test.npy')
y_test = np.load('result_credit_NN/y_test.npy')



# evaluation
precision = len(pred_y[(pred_y == 1) & (y_test == 1)])/len(pred_y[pred_y == 1])
recall = len(y_test[(y_test == 1) & (pred_y == 1)])/len(y_test[y_test == 1])
F1 = 2/((1/precision)+(1/recall))
print("F1 score: ", F1)
# confusion matrix
sns.set()
# f, ax = plt.subplot()
C1 = confusion_matrix(y_test, pred_y, labels=[0, 1])
print(C1)

# plt.hist(score, 40, density=True)
# plt.xlabel("score")
# plt.ylabel("probability of each score")
# plt.savefig('histogram_')


false_positive_index = np.where((pred_y == 1) & (y_test == 0))
false_negative_index = np.where((pred_y ==0) & (y_test == 1))
true_positive_index = np.where((pred_y == 1) & (y_test == 1))
true_negative_index = np.where((pred_y ==0) & (y_test == 0))
### plot credit data
sequence = X_test.shape[1]
plt.figure(1)
plt.plot(np.arange(sequence), X_test[false_positive_index[0][0]])
plt.title('original data')
plt.savefig('result_credit_NN/false_positive_original')
plt.figure(2)
plt.plot(np.arange(sequence), pred[false_positive_index[0][0]])
plt.title('generated data')
plt.savefig('result_credit_NN/false_positive_generated')
plt.figure(3)
plt.plot(np.arange(sequence), X_test[false_negative_index[0][0]])
plt.title('original data')
plt.savefig('result_credit_NN/false_negative_original')
plt.figure(4)
plt.plot(np.arange(sequence), pred[false_negative_index[0][0]])
plt.title('generated data')
plt.savefig('result_credit_NN/false_negative_generated')
plt.figure(5)
plt.plot(np.arange(sequence), X_test[true_positive_index[0][0]])
plt.title('original data')
plt.savefig('result_credit_NN/true_positive_original')
plt.figure(6)
plt.plot(np.arange(sequence), pred[true_positive_index[0][0]])
plt.title('generated data')
plt.savefig('result_credit_NN/true_positive_generated')
plt.figure(7)
plt.plot(np.arange(sequence), X_test[true_negative_index[0][0]])
plt.title('original data')
plt.savefig('result_credit_NN/true_negative_original')
plt.figure(8)
plt.plot(np.arange(sequence), pred[true_negative_index[0][0]])
plt.title('generated data')
plt.savefig('result_credit_NN/true_negative_generated')

### plot sin data
# sequence = X_test.shape[1]
# plt.figure(1)
# [plt.plot(np.arange(sequence), X_test[false_positive_index[0][5]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('original data')
# plt.savefig('result_sin/false_positive_original')
# plt.figure(2)
# [plt.plot(np.arange(sequence), pred[false_positive_index[0][5]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('generated data')
# plt.savefig('result_sin/false_positive_generated')
# plt.figure(3)
# [plt.plot(np.arange(sequence), X_test[true_positive_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('original data')
# plt.savefig('result_sin/true_positive_original')
# plt.figure(4)
# [plt.plot(np.arange(sequence), pred[true_positive_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('generated data')
# plt.savefig('result_sin/true_positive_generated')
# plt.figure(5)
# [plt.plot(np.arange(sequence), X_test[true_negative_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('original data')
# plt.savefig('result_sin/true_negative_original')
# plt.figure(6)
# [plt.plot(np.arange(sequence), pred[true_negative_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
# plt.title('generated data')
# plt.savefig('result_sin/true_negative_generated')
# X_test = X_test.reshape(X_test.shape[:3])
# pred = pred.reshape(pred.shape[:3])
# visualization(X_test, pred,'tsne')
# visualization_confusion_matrix(np.concatenate((X_test[true_negative_index],pred[true_negative_index])),
#                                np.concatenate((X_test[true_positive_index],pred[true_positive_index])),
#                                np.concatenate((X_test[false_positive_index],pred[false_positive_index])),
#                                np.concatenate((X_test[false_negative_index],pred[false_negative_index])))
#
#
# ## plot heatmap image
# query_heatmap = qurey[false_positive_index[0][1]].reshape(28,28)
# pred_heatmap = pred[false_positive_index[0][1]].reshape(28,28)
# diff_heatmap = diff[false_positive_index[0][1]].reshape(28,28)
# diff_heatmap[np.abs(diff_heatmap) <= 10] = 0
# diff_heatmap[np.abs(diff_heatmap) > 10] = 1
#
#
#
# # ax = sns.heatmap(diff_heatmap, linewidth=0.5)          # sns automatically will change the scale to 0~255
# plt.figure(1, figsize=(3,3))
# plt.title('query image')
# plt.imshow(query_heatmap, cmap='Greys')
# plt.colorbar()
# plt.figure(2, figsize=(3,3))
# plt.title('generated image')
# plt.imshow(pred_heatmap, cmap='Greys')
# plt.colorbar()
# plt.figure(3, figsize=(3,3))
# plt.title('anomaly detection')
# plt.imshow(query_heatmap, cmap='Greys')
# plt.imshow(diff_heatmap, cmap='Reds', alpha=0.5)
# # plt.colorbar()
# plt.show()

# plt.imshow(query_heatmap, cmap='hot', interpolation='nearest')
# plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
# plt.imshow(diff_heatmap, cmap='hot', interpolation='nearest')
# ## matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(query_heatmap, cmap='hot', interpolation='nearest')
#
#
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
#
# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(diff_heatmap, cmap='hot', interpolation='nearest')
# plt.show()
# cv2.imwrite('./qurey.png', query_heatmap)
# cv2.imwrite('./pred.png', pred_heatmap)
# cv2.imwrite('./diff.png', diff_heatmap)
#
# ## matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(cv2.cvtColor(query_heatmap,cv2.COLOR_BGR2RGB))
#
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(cv2.cvtColor(pred_heatmap,cv2.COLOR_BGR2RGB))
#
# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(cv2.cvtColor(diff_heatmap,cv2.COLOR_BGR2RGB))
# plt.show()

