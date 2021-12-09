import numpy as np
import pandas as pd
import shap
import anogan

n=6
# load result of GAN model
X_test = np.load('result_artificial/test_qurey.npy').reshape(-1,n)
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
test_pred = np.load('result_artificial/test_pred.npy').reshape(-1,n)
test_pred_standard = (test_pred-np.min(X_test))/(np.max(X_test)-np.min(X_test))
rec_err = np.linalg.norm(X_test_standard-test_pred_standard, axis=1)
result = pd.DataFrame(data=None,index=range(len(X_test_standard)),columns=['shapvalue'])
# calculate SHAP value
for j in range(len(X_test_standard)):
    idx = j
    df = pd.DataFrame(data=test_pred_standard[idx], index= range(n),columns=['reconstruction_loss'])
    loaded_model = anogan.anomaly_detector()
    loaded_model.load_weights('result_artificial/weights/test_1_' + str(j) + '.h5')
    explainer = shap.DeepExplainer(loaded_model, X_test_standard[:250,:])
    shap_values = explainer.shap_values(X_test_standard[idx,:].reshape(-1,n))
    result.loc[j, 'shapvalue'] = shap_values

result.to_csv("result_artificial/shapvalue.csv")

