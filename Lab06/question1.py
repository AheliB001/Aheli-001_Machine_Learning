import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

#LOAD DATASET
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

#SEPARATE THE FEATURES AND THE TARGET
X = data.drop(columns=["disease_score_fluct","disease_score"]).values
y = data["disease_score_fluct"].values.reshape(-1, 1)

#LINEAR REGRESSION FROM SCRATCH
def train_linear_regression(X_train, y_train):
    X_b = np.c_[np.ones((len(X_train), 1)), X_train]
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train
    return theta

#PREDICTION FUNCTION
def predict(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b @ theta

#K-FOLD CROSS VALIDATION
def k_fold_cross_validation(X, y, k=10):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, val_index in kf.split(X):

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        theta = train_linear_regression(X_train, y_train)
        predictions = predict(X_val, theta)

        mse = mean_squared_error(y_val, predictions)
        mse_scores.append(mse)

    return np.array(mse_scores)

#RUN CROSS VALIDATION
scratch_mse = k_fold_cross_validation(X, y, k=10)
print("MSE for each fold:", scratch_mse)
print("Average MSE:", scratch_mse.mean())
print("Standard Deviation:", scratch_mse.std())

#USING SCIKIT LEARN METHOD
print("\nUSING SCIKIT-LEARN")

model = LinearRegression()
#This creates a ready made model

kf = KFold(n_splits=10, shuffle=True, random_state=42)
#Creates 10 folds

sklearn_scores = cross_val_score(model,X,y.ravel(),cv=kf,scoring="neg_mean_squared_error")

sklearn_mse = -sklearn_scores

print("MSE for each fold:", sklearn_mse)
print("Average MSE:", sklearn_mse.mean())
print("Standard Deviation:", sklearn_mse.std())

