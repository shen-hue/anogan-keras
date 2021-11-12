import numpy as np
import pandas as pd
import shap
import anogan

n=6
X_test = np.load('result_artificial/test_qurey.npy').reshape(-1,n)
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
# test = pd.DataFrame(data=test,index=range(len(test)),columns=range(2))
test_pred = np.load('result_artificial/test_pred.npy').reshape(-1,n)
test_pred_standard = (test_pred-np.min(X_test))/(np.max(X_test)-np.min(X_test))
rec_err = np.linalg.norm(X_test_standard-test_pred_standard, axis=1)
# idx = list(rec_err).index(max(rec_err))
result = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[1]))
loaded_model = anogan.anomaly_detector()
loaded_model.load_weights('weights/artificial_encode.h5')
for j in range(len(X_test_standard)):
    print("number of example: ", j)
    idx = j
    df = pd.DataFrame(data=test_pred_standard[idx], index= range(n),columns=['reconstruction_loss'])

    explainer = shap.DeepExplainer(loaded_model, X_test)
    shap_values = explainer.shap_values(X_test[idx,:].reshape(-1,n))
    result[j] = shap_values

np.save('result_artificial/shapvalue_unnormalized', result)
