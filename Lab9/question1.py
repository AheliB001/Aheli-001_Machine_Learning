import pandas as pd

data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

thresholds=[80,78,82] #thrshold values

for t in thresholds: #loops through each threshold one by one
  left=data[data['BP']<=t] #subset for data where data is less than threshold
  right=data[data['BP']>t] #subset for data where data is more than threshold

  print("\nThreshold: ",t)
  print("\nLeft Data: ",len(left))
  print("\nRight Data: ",len(right))
