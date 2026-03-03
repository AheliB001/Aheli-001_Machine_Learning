from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale manually
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge
ridge = RidgeClassifier()
ridge.fit(X_train, y_train)
ridge_acc = accuracy_score(y_test, ridge.predict(X_test))

# Lasso
lasso = LogisticRegression(penalty='l1', solver='liblinear')
lasso.fit(X_train, y_train)
lasso_acc = accuracy_score(y_test, lasso.predict(X_test))

print("Ridge Accuracy:", ridge_acc)
print("Lasso Accuracy:", lasso_acc)