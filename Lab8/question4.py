import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
data = pd.read_csv(url, header=None, na_values='?')

#Fill missing values
data = data.fillna(data.mode().iloc[0])

#Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Encode target
y = LabelEncoder().fit_transform(y)

#Split once
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#ORDINAL
X_train_o = OrdinalEncoder().fit_transform(X_train)
X_test_o = OrdinalEncoder().fit(X_train).transform(X_test)

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train_o, y_train)
print("Ordinal Accuracy:", accuracy_score(y_test, model1.predict(X_test_o)))

#ONE HOT
encoder = OneHotEncoder(sparse_output=False)
X_train_h = encoder.fit_transform(X_train)
X_test_h = encoder.transform(X_test)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_h, y_train)
print("One-Hot Accuracy:", accuracy_score(y_test, model2.predict(X_test_h)))