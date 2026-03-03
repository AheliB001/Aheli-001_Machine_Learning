import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
    data = pd.read_csv(url, header=None, na_values='?')
    data = data.fillna(data.mode().iloc[0])

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = X.apply(LabelEncoder().fit_transform)
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42,
        max_depth=5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    #Plotting tree
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.show()


if __name__ == "__main__":
    main()