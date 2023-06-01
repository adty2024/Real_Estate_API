import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import h5py
import pickle
import base64

# Fetch the Data Set
data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['Target'] = data.target

train_features_df = df.drop('Target', axis=1)
train_label_df = df['Target'].copy()

model_M = DecisionTreeRegressor()
model_M.fit(train_features_df, train_label_df)

# Convert the model to a string representation
model_string = base64.b64encode(pickle.dumps(model_M)).decode('utf-8')

# Save the model string to an HDF5 file
with h5py.File('model.h5', 'w') as f:
    f.create_dataset('model', data=model_string)
