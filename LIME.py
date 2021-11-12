import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import anogan

n=6
X_test = np.load('result_artificial/test_qurey.npy').reshape(-1,n)
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
y_test = np.load('result_artificial/test_qurey.npy').reshape(-1,n)
y_test_standard = (y_test-np.min(y_test))/(np.max(y_test)-np.min(y_test))
# test = pd.DataFrame(data=test,index=range(len(test)),columns=range(2))
test_pred = np.load('result_artificial/test_pred.npy').reshape(-1,n)
test_pred_standard = (test_pred-np.min(X_test))/(np.max(X_test)-np.min(X_test))
rec_err = np.linalg.norm(X_test_standard-test_pred_standard, axis=1)
# idx = list(rec_err).index(max(rec_err))
result = np.load('result_artificial/limevalue.npy')
# result = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[1]))
loaded_model = anogan.anomaly_detector()
loaded_model.load_weights('weights/artificial_encode.h5')
for j in range(len(X_test_standard)):
    print("number of example: ", j)
    idx = j
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test_standard[:510],feature_names=range(6),
                class_names=range(6),categorical_features=range(6),verbose=True,mode='regression')
    lime_value = explainer.explain_instance(X_test_standard[j],loaded_model.predict,num_features=6)
    for i in range(6):
        number = int(lime_value.as_list()[i][0][0])
        result[j][5][number] = lime_value.as_list()[i][1]

np.save('result_artificial/limevalue', result)
