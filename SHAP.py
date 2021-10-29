import numpy as np
import pandas as pd
import shap
import anogan

n=6
X_test = np.load('result_artificial/test_qurey.npy').reshape(-1,n)
test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
# test = pd.DataFrame(data=test,index=range(len(test)),columns=range(2))
test_pred = np.load('result_artificial/test_pred.npy').reshape(-1,n)
test_pred = (test_pred-np.min(X_test))/(np.max(X_test)-np.min(X_test))
rec_err = np.linalg.norm(test-test_pred, axis=1)
# idx = list(rec_err).index(max(rec_err))
result = pd.DataFrame(data=None,index=range(len(test)),columns=['shapvalue'])
# for j in [200]:
for j in range(len(test)):
    idx = j
    df = pd.DataFrame(data=test_pred[idx], index= range(n),columns=['reconstruction_loss'])

    def sort_by_absolute(df, index):
        df_abs = df.apply(lambda x: abs(x))
        df_abs = df_abs.sort_values('reconstruction_loss', ascending=False)
        df = df.loc[df_abs.index, :]
        return df


    top_5_features = sort_by_absolute(df, idx).iloc[:6,:]
    # data_summary = test[idx].reshape(1,test.shape[1])
    # data_summary = data_summary.repeat(10,axis=0)
    data_summary = shap.kmeans(test[:250,:], 100)

    shaptop5features = pd.DataFrame(data=None)
    shap_value_ordered = np.zeros((n, n))
    for i in top_5_features.index:
        # load weights into new model
        loaded_model = anogan.anomaly_detector()
        loaded_model.load_weights('result_artificial/weights/test_1_'+str(j)+'.h5')
        weights = loaded_model.get_weights()

        ## make sure the weight for the specific one input feature is set to 0
        # feature_index = list(df.index).index(i)
        # print(feature_index, i)
        feature_index = list(df.index).index(i)
        updated_weights = weights[:][0]
        updated_weights[feature_index] = [0] * len(updated_weights[feature_index])
        loaded_model.get_layer(index=1).set_weights([updated_weights, weights[:][1]])

        ## determine the SHAP values
        explainer_autoencoder = shap.KernelExplainer(loaded_model.predict, data_summary)
        shap_values = explainer_autoencoder.shap_values(test[idx, :])

        ## build up pandas dataframe
        # shaptop5features[str(i)] = pd.Series(shap_values[feature_index])
        shap_value_ordered[:,i] = shap_values[feature_index]
    result.loc[j, 'shapvalue'] = shap_value_ordered
result.to_csv("result_artificial/shapvalue.csv")
# columns = ['0','1']
# shaptop5features.index = columns
# shaptop5features.index = df.index
# print(shaptop5features)

