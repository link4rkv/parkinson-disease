import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# Read the data
X = pd.read_csv("../input/parkinson-disease-detection/Parkinsson disease.csv")
X.head()

# Remove rows with missing target, separate target from predictors
X.dropna(axis = 0, subset = ['status'], inplace = True)
y = X.status
X.drop(['status'], axis = 1, inplace = True)
X.head()

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [column for column in X.columns if X[column].nunique() < 10 and 
                        X[column].dtype == "object"]

# Select numeric columns
numeric_cols = [column for column in X.columns if X[column].dtype in ['int64', 'float64']]

# Keep selected columns only
cols = low_cardinality_cols + numeric_cols
X = X[cols]

# One-hot encode the data (to shorten the code, we use pandas)
X = pd.get_dummies(X)
X.head()

# Scale the data from -1 to 1
minmax_scaling(X, columns = X.columns, min_val = -1, max_val = 1)
X.head()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Define the model
model = XGBClassifier(random_state = 0, disable_default_eval_metric = True, use_label_encoder = False)

# Fit the model
model.fit(X_train, y_train)

# Predict on X_valid
predictions = model.predict(X_valid)

# Error and accuracy analysis
print(mean_absolute_error(predictions, y_valid))
print(accuracy_score(y_valid, predictions) * 100)

# RandomForestClassifier Implementation
forest_model = RandomForestClassifier(n_estimators = 100, random_state = 1)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_valid)

print(mean_absolute_error(forest_predictions, y_valid))
print(accuracy_score(y_valid, forest_predictions) * 100)

# Neural Nets Implementation
nn_model = keras.Sequential([
    layers.Dense(128, activation = 'relu', input_shape = [22]),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(1)
])

nn_model.compile(
    optimizer = 'adam',
    loss = 'mae'
)

early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 5,
    restore_best_weights = True
)

nn_model.fit(
    X_train, y_train,
    batch_size = 30, 
    epochs = 50,
    callbacks = [early_stopping]
)

nn_predictions = nn_model.predict(X_valid)

print(mean_absolute_error(nn_predictions, y_valid))
