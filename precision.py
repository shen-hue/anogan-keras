import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.metrics import confusion_matrix
import pandas as pd
from plotly.offline import iplot


# load test result
score = np.load('result_cluster_1/test_score.npy')
qurey= np.load('result_cluster_1/test_qurey.npy')
pred = np.load('result_cluster_1/test_pred.npy')
diff = np.load('result_cluster_1/test_diff.npy')



threshold = 0.0185

# order prediction result(anomaly:1, normal:0)
score = score.flatten()
pred_y = np.zeros((score.shape[0])).astype(int)
pred_y[score > threshold] = 1            #anomaly 1, normal 0
# X_train = np.load('result_artificial/X_train.npy')
X_test = np.load('result_cluster_1/X_test.npy')
y_test = np.load('result_cluster_1/y_test.npy').reshape(300)



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

### high dimension
# methods = manifold.TSNE(n_components=2,init='pca',random_state=0)
# Y = methods.fit_transform(X_test)
plt.figure()
colors = np.array(['#377eb8', '#ff7f00'])
# plt.scatter(X_train[:, 0], X_train[:, 1])
plt.scatter(X_test[:, 0], X_test[:, 1], s=10, color=colors[(pred_y + 1) // 2])
plt.show()
#
# false_positive_index = np.where((pred_y == 1) & (y_test == 0))
# false_negative_index = np.where((pred_y ==0) & (y_test == 1))
# true_positive_index = np.where((pred_y == 1) & (y_test == 1))
# true_negative_index = np.where((pred_y ==0) & (y_test == 0))
## plot shap value

import plotly.graph_objs as go
result = pd.read_csv("result_cluster_1/shapvalue.csv")
dfnormal = result.loc[pred_y==0]
dfanomaly = result.loc[pred_y==1]
trace1 = go.Scatter(x=X_test[pred_y==0][:,0],y=X_test[pred_y==0][:,1],mode="markers",
                    name="normal",marker=dict(color='rgba(255, 128, 255, 0.8)'),
                    text=dfnormal.shapvalue)
trace2 = go.Scatter(x=X_test[pred_y==1][:,0],y=X_test[pred_y==1][:,1],mode="markers",
                    name="anomaly",marker=dict(color='rgba(255, 128, 2, 0.8)'),
                    text=dfanomaly.shapvalue)
data = [trace1,trace2]
layout = dict(title='shap value of all the data',
              xaxis=dict(title='feature 1'),
              yaxis=dict(title='feature 2'))
fig = dict(data=data,layout=layout)

import dash
import dash_core_components as dcc
import dash_html_components as html
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
app.run_server(debug=True)

