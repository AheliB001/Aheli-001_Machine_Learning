import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_stump(X, y, weights):
    n_samples, n_features = X.shape

    best_error = float('inf')
    best_feature = None
    best_threshold = None
    best_polarity = 1

    for f in range(n_features):
        thresholds = np.unique(X[:, f])

        for t in thresholds:
            for polarity in [1, -1]:

                preds = np.ones(n_samples)
                preds[X[:, f] < t] = -1
                preds *= polarity

                error = np.sum(weights[preds != y])

                if error < best_error:
                    best_error = error
                    best_feature = f
                    best_threshold = t
                    best_polarity = polarity

    return best_feature, best_threshold, best_polarity, best_error

def train_adaboost(X, y, n_estimators=10):
    n_samples = len(y)

    weights = np.ones(n_samples) / n_samples

    models = []
    alphas = []

    for i in range(n_estimators):
        f, t, polarity, error = train_stump(X, y, weights)

        # Avoid division by zero
        error = max(error, 1e-10)

        alpha = 0.5 * np.log((1 - error) / error)

        # Predictions
        preds = np.ones(n_samples)
        preds[X[:, f] < t] = -1
        preds *= polarity

        # Update weights
        weights *= np.exp(-alpha * y * preds)
        weights /= np.sum(weights)

        models.append((f, t, polarity))
        alphas.append(alpha)

        print(f"Round {i + 1}: Feature={f}, Threshold={t:.2f}, Error={error:.4f}, Alpha={alpha:.4f}")

    return models, alphas

def predict(X, models, alphas):
    final = np.zeros(X.shape[0])

    for (f, t, polarity), alpha in zip(models, alphas):
        preds = np.ones(X.shape[0])
        preds[X[:, f] < t] = -1
        preds *= polarity

        final += alpha * preds

    return np.sign(final)

models, alphas = train_adaboost(X_train, y_train, n_estimators=10)

train_preds = predict(X_train, models, alphas)
test_preds = predict(X_test, models, alphas)

train_acc = np.mean(train_preds == y_train) * 100
test_acc = np.mean(test_preds == y_test) * 100

print("\n=== Final Results ===")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy : {test_acc:.2f}%")