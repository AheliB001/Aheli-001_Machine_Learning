from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Implement XGBoost classifier using scikit-learn

data = load_iris()
X, Y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=999)

model = XGBClassifier()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))

#Implement XGBoost regressor using scikit-learn

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

data1 = load_diabetes()
x=data1.data
y=data1.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=999)
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)

model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=2,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

model.fit(x_train_sc, y_train)
y_pred = model.predict(x_test_sc)
print("r2 score:", r2_score(y_test, y_pred))



