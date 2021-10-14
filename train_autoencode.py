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

    X_train = datasets[2]
    y_train = np.asarray([0] * 255)
    rng = np.random.RandomState(42)
    X_test = np.concatenate([X_train, rng.uniform(low=-6, high=6,
                                                  size=(n_outliers, 2))], axis=0)

    ## Train Autoencoder
    model = autoencoder(X_train, X_test)
    print(model.summary())

    ## Save the Autoencoder
    model_json = model.to_json()
    with open("log/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("log/model.h5")
    print("Saved model to disk")
    # load json and create model
    json_file = open('log/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    return model, X_train, pd.DataFrame(data=X_test, columns=['1','2']), loaded_model_json


if __name__ == "__main__":
    model, X, X_standard, loaded_model_json = main()
    print(
        "Autoencoder available as model, original DataFrame available as X and normalised DataFrame available as X_standard.")


model, X, X_standard, loaded_model_json = main()
X_reconstruction_standard = model.predict(X_standard)
rec_err = np.linalg.norm(X_standard - X_reconstruction_standard, axis = 1)
idx = list(rec_err).index(max(rec_err))
df = pd.DataFrame(data = X_reconstruction_standard[idx], index = X.columns, columns = ['reconstruction_loss'])
df.T
os.getcwd()