import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# load test result
# score = np.load('result_artificial/test_score.npy')
qurey= np.load('result_cluster_4/test_qurey.npy')
pred = np.load('result_cluster_4/test_pred.npy')
score = np.load('result_cluster_4/test_diff.npy')

threshold = 4

# order prediction result(anomaly:1, normal:0)
score = score.flatten()
pred_y = np.zeros((score.shape[0])).astype(int)
pred_y[score > threshold] = 1            #anomaly 1, normal 0
# X_train = np.load('result_artificial/X_train.npy')
X_test = np.load('result_cluster_4/X_test.npy')
y_test = np.load('result_cluster_4/y_test.npy').reshape(345)



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


plt.figure()
colors = np.array(['#377eb8', '#ff7f00'])
# plt.scatter(X_train[:, 0], X_train[:, 1])
plt.scatter(X_test[:, 0], X_test[:, 1], s=10, color=colors[(pred_y + 1) // 2])
plt.show()

false_positive_index = np.where((pred_y == 1) & (y_test == 0))
false_negative_index = np.where((pred_y ==0) & (y_test == 1))
true_positive_index = np.where((pred_y == 1) & (y_test == 1))
true_negative_index = np.where((pred_y ==0) & (y_test == 0))
## plot sin data


#
#

