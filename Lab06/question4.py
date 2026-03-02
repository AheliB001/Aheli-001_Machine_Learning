import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
x=data.drop(columns=["disease_score_fluct"])
y=data["disease_score_fluct"]

#split the dataset into training set and validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#trying different models for model selection
model1=LinearRegression()
model2=DecisionTreeRegressor()

#train models
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

#predict on validation set
prediction1=model1.predict(x_val)
prediction2=model2.predict(x_val)

#calculate validation error
mse=mean_squared_error(y_val,prediction1)
mse2=mean_squared_error(y_val,prediction2)

print("linear regression mse=", mse)
print("decision tree regression mse2=",mse2)

if mse < mse2:
    print ("best model is linear regression")

else:
    print ("best model is decision tree")