import numpy as np
import pandas as pd
import shap
import anogan
from main import anomaly_score


n=6
X_test = np.load('result_artificial_classification/test_qurey.npy').reshape(-1,n)
X_test_standard = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
test_pred = np.load('result_artificial_classification/test_pred.npy').reshape(-1,n)
test_pred_standard = (test_pred-np.min(X_test))/(np.max(X_test)-np.min(X_test))
rec_err = np.linalg.norm(X_test_standard-test_pred_standard, axis=1)

result = np.zeros((X_test.shape[0],X_test.shape[1],X_test.shape[1]))
loaded_model = anomaly_score()
loaded_model.load_weights('weights/artificial_classification_decode.h5')
## calcaluate SHAP value of each data point
for j in range(len(X_test_standard)):
    print("number of example: ", j)
    idx = j
    df = pd.DataFrame(data=test_pred_standard[idx], index= range(n),columns=['reconstruction_loss'])

    explainer = shap.DeepExplainer(loaded_model, X_test_standard)
    shap_values = explainer.shap_values(X_test_standard[idx,:].reshape(-1,n))
    ## plot SHAP value
    # shap.force_plot(base_value=explainer.expected_value[0].numpy(), shap_values=shap_values[0],
    #                                               features=X_test[idx,:],show=False,matplotlib=True).savefig('r.png')
    # X_test_label = pd.DataFrame(data=X_test_standard,columns=['feature 0','feature 1','feature 2','feature 3','feature 4','feature 5'])
    # shap.save_html('./test.html', shap.force_plot(base_value=explainer.expected_value[4].numpy(), shap_values=shap_values[4],
    #                                               features=X_test_label.iloc[idx,:],show=False))
    result[j,:,:] = shap_values

np.save('result_artificial_classification/shapvalue', result)
