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
score = np.load('result_sin/sin_test_score.npy')
qurey= np.load('result_sin/sin_test_qurey.npy')
pred = np.load('result_sin/sin_test_pred.npy')
diff = np.load('result_sin/sin_test_diff.npy')

threshold = 180

# order prediction result(anomaly:1, normal:0)
score = score.flatten()
pred_y = np.zeros((score.shape[0]))
pred_y[score > threshold] = 1            #anomaly 1, normal 0
X_test = np.load('result_sin/X_test.npy')
y_test = np.load('result_sin/y_test.npy')



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
## plot sin data
sequence = X_test.shape[1]
plt.figure(1)
[plt.plot(np.arange(sequence), X_test[false_positive_index[0][5]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('original data')
plt.savefig('result_sin/false_positive_original')
plt.figure(2)
[plt.plot(np.arange(sequence), pred[false_positive_index[0][5]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('generated data')
plt.savefig('result_sin/false_positive_generated')
plt.figure(3)
[plt.plot(np.arange(sequence), X_test[true_positive_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('original data')
plt.savefig('result_sin/true_positive_original')
plt.figure(4)
[plt.plot(np.arange(sequence), pred[true_positive_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('generated data')
plt.savefig('result_sin/true_positive_generated')
plt.figure(5)
[plt.plot(np.arange(sequence), X_test[true_negative_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('original data')
plt.savefig('result_sin/true_negative_original')
plt.figure(6)
[plt.plot(np.arange(sequence), pred[true_negative_index[0][0]][:,s].reshape(sequence)) for s in range(sequence)]
plt.title('generated data')
plt.savefig('result_sin/true_negative_generated')
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

