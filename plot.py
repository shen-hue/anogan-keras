import pandas as pd
from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_dataset():
  # Create sample data with sklearn make_regression function
  X, y = make_regression(n_samples=1000, n_features=10, n_informative=7, n_targets=5, random_state=0)

  # Convert the data into Pandas Dataframes for easier maniplution and keeping stored column names
  # Create feature column names
  feature_cols = ['feature_01', 'feature_02', 'feature_03', 'feature_04',
                  'feature_05', 'feature_06', 'feature_07', 'feature_08',
                  'feature_09', 'feature_10']

  df_features = pd.DataFrame(data = X, columns = feature_cols)

  # Create lable column names and dataframe
  label_cols = ['labels_01', 'labels_02', 'labels_03', 'labels_04', 'labels_05']

  df_labels = pd.DataFrame(data = y, columns = label_cols)

  return df_features, df_labels

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mae', optimizer='adam')
    return model

# Create the datasets
X, y = get_dataset()

# Get the number of inputs and outputs from the dataset
n_inputs, n_outputs = X.shape[1], y.shape[1]

model = get_model(n_inputs, n_outputs)
model.fit(X, y, verbose=0, epochs=100)
model.evaluate(x = X, y = y)
model.predict(X.iloc[0:1,:])

import shap

# print the JS visualization code to the notebook
shap.initjs()

explainer = shap.KernelExplainer(model = model.predict, data = X.head(50), link = "identity")
# Set the index of the specific example to explain
X_idx = 0

shap_value_single = explainer.shap_values(X = X.iloc[X_idx:X_idx+1,:], nsamples = 100)
X.iloc[X_idx:X_idx+1,:]
import ipywidgets as widgets
# Create the list of all labels for the drop down list
list_of_labels = y.columns.to_list()

# Create a list of tuples so that the index of the label is what is returned
tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

# Create a widget for the labels and then display the widget
current_label = widgets.Dropdown(options=tuple_of_labels,
                              value=0,
                              description='Select Label:'
                              )

# Display the dropdown list (Note: access index value with 'current_label.value')
current_label
# print the JS visualization code to the notebook
shap.initjs()

print(f'Current label Shown: {list_of_labels[current_label.value]}')

shap.force_plot(base_value = explainer.expected_value[current_label.value],
                shap_values = shap_value_single[current_label.value],
                features = X.iloc[X_idx:X_idx+1,:]
                )