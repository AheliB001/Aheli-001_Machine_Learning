from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

data = load_iris()
X= data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# depth=1 = decision stump
base_estimator = DecisionTreeClassifier(max_depth=1)

ada = AdaBoostClassifier(estimator=base_estimator,  n_estimators=100, learning_rate=1.0, random_state=999)

ada.fit(X_train, y_train)
y_pred=ada.predict(X_test)

print("Accuracy score:", accuracy_score(y_test, y_pred))

