# Titanic Survival Classification

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Training Values
X_train = train.iloc[:, [2, 4, 5, 6, 7, 9]].values
y_train = train.iloc[:, 1].values

# Encode the data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])

# Impute missing data
from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3])

# One Hot Encode Pclasses
from sklearn.preprocessing import OneHotEncoder
encoder_train = OneHotEncoder(categorical_features=[0])
X_train = encoder_train.fit_transform(X_train).toarray()
X_train = X_train[:, 1:8]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
X_train = sc_train.fit_transform(X_train)

# Training Values
X_test = test.iloc[:, [1, 3, 4, 5, 6, 8]].values
y_test = test.iloc[:, 0].values

# Encode the data
labelencoder_test = LabelEncoder()
X_test[:, 1] = labelencoder_test.fit_transform(X_test[:, 1])

# Impute missing data
from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_train = imputer_train.fit(X_test[:, [2,5]])
X_test[:, [2,5]] = imputer_train.transform(X_test[:, [2,5]])

# One Hot Encode Pclasses
from sklearn.preprocessing import OneHotEncoder
encoder_train = OneHotEncoder(categorical_features=[0])
X_test = encoder_train.fit_transform(X_test).toarray()
X_test = X_test[:, 1:8]

# Feature Scaling
sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = y_pred.reshape(418, 1)

predictions = np.empty((418,2))
predictions[:, 1:2] = y_pred
predictions[:, 0:1] = y_test.reshape(418, 1)

predictions = predictions.astype(int)

np.savetxt("titanic_predictions.csv", predictions, fmt='%i', delimiter=',')