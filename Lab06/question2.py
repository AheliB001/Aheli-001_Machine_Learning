import pandas as pd

data=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
X=data.drop(columns=["disease_score_fluct"]).values

x_min = X.min(axis=0)
x_max = X.max(axis=0)

x_new = (X-x_min)/(x_max-x_min)
print(x_new)