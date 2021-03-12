import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

import xgboost import XGBClassifier

X = pd.read_csv("../input/parkinson-disease-detection/Parkinsson disease.csv")
X.head()


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

minmax_scaling(X, columns = X.columns, min_val = -1, max_val = 1)
X.head()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

model = XGBClassifier(random_state = 0, disable_default_eval_metric = True, use_label_encoder = False)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

print(mean_absolute_error(predictions, y_valid))
print(accuracy_score(y_valid, predictions) * 100)
