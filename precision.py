import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.metrics import confusion_matrix
import pandas as pd
from plotly.offline import iplot


# load test result
score = np.load('result_artificial/test_score.npy')
qurey= np.load('result_artificial/test_qurey.npy')
pred = np.load('result_artificial/test_pred.npy')
diff = np.load('result_artificial/test_diff.npy')

lime_value = np.load('result_artificial/limevalue.npy')
mean_shap = np.mean(np.abs(lime_value),axis=0)

threshold = 2

# order prediction result(anomaly:1, normal:0)
score = score.flatten()
pred_y = np.zeros((score.shape[0])).astype(int)
pred_y[score > threshold] = 1            #anomaly 1, normal 0
# X_train = np.load('result_artificial/X_train.npy')
X_test = np.load('result_artificial/X_test.npy')
y_test = np.load('result_artificial/y_test.npy')



# evaluation
precision = len(pred_y[(pred_y == 1) & (y_test == 1)])/len(pred_y[pred_y == 1])
recall = len(y_test[(y_test == 1) & (pred_y == 1)])/len(y_test[y_test == 1])
F1 = 2/((1/precision)+(1/recall))
print("F1 score: ", F1)
# confusion matrix
C1 = confusion_matrix(y_test, pred_y, labels=[0, 1])
print(C1)


# true positive and false positive index
# false_positive_index = np.where((pred_y == 1) & (y_test == 0))
# false_negative_index = np.where((pred_y ==0) & (y_test == 1))
# true_positive_index = np.where((pred_y == 1) & (y_test == 1))
# true_negative_index = np.where((pred_y ==0) & (y_test == 0))
