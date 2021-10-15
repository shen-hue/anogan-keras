import numpy as np
import pandas as pd
import shap

# load JS visualization code to notebook
from sklearn.datasets import make_blobs, make_moons

shap.initjs()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import model_from_json

import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os

# os.chdir("../")     # 不需要再改变路径到上一级文件夹

from autoencoder_specifications import *


def autoencoder(X_train, X_test):
    input_shape = X_train.shape[1]
    input_layer = Input(shape=(input_shape,))
    hid_layer1 = Dense(dim_hid1, activation=activation, name='hid_layer1')(input_layer)
    hid_layer2 = Dense(dim_hid2, activation=activation, name='hid_layer2')(hid_layer1)
    hid_layer3 = Dense(dim_hid3, activation=activation, name='hid_layer3')(hid_layer2)
    output_layer = Dense(input_shape)(hid_layer3)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimiser = Adam(lr=learning_rate)
    model.compile(optimizer=optimiser, loss='mean_squared_error')
    model.fit(x=X_train, y=X_train, batch_size=batch_size, shuffle=True,
              epochs=epochs, verbose=0, validation_data=[X_test, X_test],
              callbacks=[early_stop])
    return model


def main():
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)  # anomaly data
    n_inliers = n_samples - n_outliers  # normal data

    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    datasets = [
        make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
                   **blobs_params)[0],
        4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
              np.array([0.5, 0.25])),
        14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

    X = datasets[4]
    rng = np.random.RandomState(42)
    X_test = np.concatenate([X, rng.uniform(low=-6, high=6,
                                            size=(n_outliers, 2))], axis=0)

    ### load artificial data
    # np.random.seed(10)
    # X = np.random.uniform(-6,6,(n_inliers,4))
    # X = np.insert(X,4,values=X[:,0]+X[:,1],axis=1)
    # X = np.insert(X,5,values=X[:,2]+X[:,3],axis=1)
    # X_test = np.concatenate([X, np.random.uniform(-6, 6, (n_outliers, 6))], axis=0)


    X_train = (X - np.min(X)) / (np.max(X) - np.min(X))
    y_train = np.asarray([0] * n_inliers)
    X_test_standard = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
    y_test = np.concatenate([[0] * n_samples, [1] * n_outliers], axis=0)

    ## Train Autoencoder
    model = autoencoder(X_train, X_test_standard)
    print(model.summary())

    ## Save the Autoencoder
    model_json = model.to_json()
    with open("log/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('log/model.h5')
    print("Saved model to disk")
    # load json and create model
    json_file = open('log/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    return model, X_test, pd.DataFrame(data=X_test_standard, columns=['0','1']), y_test, loaded_model_json


if __name__ == "__main__":
    model, X, X_standard, y_test, loaded_model_json = main()
    print(
        "Autoencoder available as model, original DataFrame available as X and normalised DataFrame available as X_standard.")


model, X, X_standard, y_test, loaded_model_json = main()
X_reconstruction_standard = model.predict(X_standard)
X_reconstruction = X_reconstruction_standard*(np.max(X)-np.min(X))+np.min(X)
np_residual = X-X_reconstruction
diff = np.sum(abs(np_residual),axis=1)

np.save('result_cluster_5/test_qurey', X)
np.save('result_cluster_5/test_pred', X_reconstruction)
np.save('result_cluster_5/test_diff', diff)
# np.save('result_cluster_3/test_score', score)
np.save('result_cluster_5/X_test', X)
np.save('result_cluster_5/y_test', y_test)


rec_err = np.linalg.norm(X - X_reconstruction, axis = 1)
idx = list(rec_err).index(max(rec_err))
df = pd.DataFrame(data = X_reconstruction_standard[idx], index = range(2), columns = ['reconstruction_loss'])

def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending = False)
    df = df.loc[df_abs.index,:]
    return df

top_5_features = sort_by_absolute(df, idx).iloc[:5,:]
data_summary = shap.kmeans(X_standard, 100)
shaptop5features = pd.DataFrame(data=None)

for i in top_5_features.index:
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('log/model.h5')
    weights = loaded_model.get_weights()

    ## make sure the weight for the specific one input feature is set to 0
    feature_index = list(df.index).index(i)
    print(feature_index, i)
    updated_weights = weights[:][0]
    updated_weights[feature_index] = [0] * len(updated_weights[feature_index])
    model.get_layer('hid_layer1').set_weights([updated_weights, weights[:][1]])

    ## determine the SHAP values
    explainer_autoencoder = shap.KernelExplainer(model.predict, data_summary)
    shap_values = explainer_autoencoder.shap_values(X_standard.loc[idx, :].values)

    ## build up pandas dataframe
    shaptop5features[str(i)] = pd.Series(shap_values[feature_index])

columns = ['0','1']
shaptop5features.index = columns
shaptop5features.index = df.index
print(shaptop5features)