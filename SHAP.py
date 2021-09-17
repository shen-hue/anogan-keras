import numpy as np
import pandas as pd
import shap
import anogan

test = np.load('result_wine_NN/wine_test_qurey.npy').reshape(-1,13)
test = pd.DataFrame(data=test,index=range(len(test)),columns=range(13))
test_pred = np.load('result_wine_NN/wine_test_pred.npy').reshape(-1,13)
rec_err = np.linalg.norm(test-test_pred, axis=0)
# idx = list(rec_err).index(max(rec_err))
idx = 0
df = pd.DataFrame(data=rec_err, index= range(13),columns=['reconstruction_loss'])

def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending=False)
    df = df.loc[df_abs.index, :]
    return df


top_5_features = sort_by_absolute(df, idx).iloc[:5,:]
data_summary = shap.kmeans(test, 10)
data = np.asarray(test)
data = data[11].reshape(1,13)

shaptop5features = pd.DataFrame(data=None)
for i in top_5_features.index:
    # load weights into new model
    loaded_model = anogan.anomaly_detector()
    loaded_model.load_weights('weights/test_11.h5')
    weights = loaded_model.get_weights()

    ## make sure the weight for the specific one input feature is set to 0
    feature_index = list(df.index).index(i)
    print(feature_index, i)
    updated_weights = weights[:][0]
    updated_weights[feature_index] = [0] * len(updated_weights[feature_index])
    loaded_model.get_layer(index=1).set_weights([updated_weights, weights[:][1]])

    ## determine the SHAP values
    explainer_autoencoder = shap.KernelExplainer(loaded_model.predict, data)
    shap_values = explainer_autoencoder.shap_values(test.loc[idx, :].values)

    ## build up pandas dataframe
    shaptop5features[str(i)] = pd.Series(shap_values[feature_index])
    shaptop5features[str(i)] = pd.Series(shap_values[feature_index])

columns = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'color intensity', 'Hue',
            'OD280/OD315 of diluted wines', 'Proline']
shaptop5features.index = columns
shaptop5features.index = df.index
print()