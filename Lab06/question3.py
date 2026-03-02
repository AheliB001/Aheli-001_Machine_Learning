import numpy as np
import pandas as pd

data=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.drop(columns=["disease_score_fluct"]).values

mean=np.mean(X, axis=0)
std=np.std(X, axis=0)
X_standardized = (X - mean)/std
print(X_standardized)